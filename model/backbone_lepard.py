#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 22:44
# @Author  : 沈子明
# @File    : backbone_lepard.py
# @Software: PyCharm
import torch
import torch.nn as nn
from model.pointnet2.pointnet2_modules import PointNet2FPModule
from model.pointnet2_inverse_module import PTENet2FPMaskMtutalModule, FeatureFuse, PTENet2FPflowModule, BaseUpRelu, \
    MaskFuse, feature_mask_fuse, feature_onlymask_fuse, Upmask
from model.pointnet2_paconv_modules import PointNet2SAModuleCUDA as PointNet2SAModule
from lib.pointops.functions import pointops
import torch.nn.functional as F

from model.transformer import GeometryAttentionLayer, PositionEncoding


def batch_weighted_procrustes(X, Y, w, eps=0.0001):
    '''
    @param X: source frame [B, N,3]
    @param Y: target frame [B, N,3]
    @param w: weights [B, N,1]
    @param eps:
    @return:
    '''
    # https://ieeexplore.ieee.org/document/88573

    bsize = X.shape[0]
    device = X.device
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul((Y - mean_Y).transpose(1, 2), w_norm * (X - mean_X))
    Sxy = Sxy.cpu().double()
    U, D, V = Sxy.svd()  # small SVD runs faster on cpu
    condition = D.max(dim=1)[0] / D.min(dim=1)[0]
    S = torch.eye(3)[None].repeat(bsize, 1, 1).double()
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1, 1)
    svT = torch.matmul(S, V.transpose(1, 2))
    R = torch.matmul(U, svT).float().to(device)
    t = mean_Y.transpose(1, 2) - torch.matmul(R, mean_X.transpose(1, 2))
    return R, t, condition


class transformer_global_sample_trans(nn.Module):
    """"
    从最终的效果看，较为失败
    """

    def __init__(self, feature_dim, n_head, n_sample, procru_sample, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.procru_sample = procru_sample
        self.nsamples = n_sample
        self.src_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.tgt_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.maskmlp = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz1, xyz2):
        """
        Parameters
        ----------
        feature1: B, C, N
        feature2: B, C, N
        xyz1: B, N, 3
        xyz2: B, N, 3
        """
        B, N, _ = xyz1.shape
        src_feat = feature1.transpose(1, 2).contiguous()
        tgt_feat = feature2.transpose(1, 2).contiguous()
        sim_matrix_1 = F.cosine_similarity(src_feat.unsqueeze(2), tgt_feat.unsqueeze(1), dim=-1)
        topk_v1, topk_idx1 = F.softmax(sim_matrix_1, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask = self.maskmlp(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # sample
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        toptop_v1, toptop_idx1 = mask.topk(dim=-2, k=self.procru_sample, sorted=True)
        xyz2 = torch.gather(xyz2, 1, toptop_idx1.unsqueeze(-1).expand(B, self.procru_sample, self.nsamples, 3))
        topk_v1_sample = torch.gather(topk_v1, 1, toptop_idx1.expand(B, self.procru_sample, self.nsamples))
        simik = self.DeSmooth_moudle(topk_v1_sample.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        xyz1_sample = torch.gather(xyz1, 1, toptop_idx1.expand(B, self.procru_sample, 3))
        R, t, condition = batch_weighted_procrustes(xyz1_sample, pred_xyz, toptop_v1)
        t = t.transpose(2, 1).contiguous()
        return [R, t]


class transformer_global_trans(nn.Module):
    def __init__(self, feature_dim, n_head, n_sample, procru_sample, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.procru_sample = procru_sample
        self.nsamples = n_sample
        self.src_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.tgt_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.maskmlp = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz1, xyz2):
        """
        Parameters
        ----------
        feature1: B, C, N
        feature2: B, C, N
        xyz1: B, N, 3
        xyz2: B, N, 3
        """
        B, N, _ = xyz1.shape
        src_feat = feature1.transpose(1, 2).contiguous()
        tgt_feat = feature2.transpose(1, 2).contiguous()
        sim_matrix_1 = F.cosine_similarity(src_feat.unsqueeze(2), tgt_feat.unsqueeze(1), dim=-1)
        topk_v1, topk_idx1 = F.softmax(sim_matrix_1, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask = self.maskmlp(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # sample
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        R, t, condition = batch_weighted_procrustes(xyz1, pred_xyz, mask)
        t = t.transpose(2, 1).contiguous()
        return [R, t]


class transformer_local_trans(nn.Module):
    """
    类似于光流与场景流，仅在近场进行搜索
    特征向量使用的是 减 出来的
    """

    def __init__(self, feature_dim, n_head, num_points, nsamples, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.num_points = num_points
        self.acti = nn.ReLU()
        self.nsamples = nsamples
        self.positional_encoding = PositionEncoding(feature_dim=self.d_model, pe_type="rotary")
        self.selfAttentionSrc = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.selfAttentionTgt = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.mask = nn.Sequential(nn.Conv1d(self.d_model, self.d_model // 2, 1, bias=False),
                                  nn.BatchNorm1d(self.d_model // 2, ),
                                  nn.ReLU(),
                                  nn.Conv1d(self.d_model // 2, self.d_model // 4, 1, bias=False),
                                  nn.BatchNorm1d(self.d_model // 4),
                                  nn.ReLU(),
                                  nn.Conv1d(self.d_model // 4, 1, 1, bias=False),
                                  nn.Sigmoid())
        self.cos_simi = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature1, feature2, xzy1_wrapped, xyz2):
        """
        Parameters
        ----------
        feature1: B, C, N
        feature2: B, C, N
        xzy1_wrapped: B, N, 3
        xyz2: B, N, 3

        Returns
        -------
        mask1: B, N, 1
        mask2: B, N, 1
        xyz1_pred: B, N, 3
        """
        bs, dims, num = feature1.shape
        feature1 = feature1.transpose(1, 2).contiguous()
        feature2 = feature2.transpose(1, 2).contiguous()
        src_pe = self.positional_encoding(xzy1_wrapped)
        tgt_pe = self.positional_encoding(xyz2)
        src_feat = self.selfAttentionSrc(feature1, feature1, src_pe, src_pe).transpose(1, 2).contiguous()
        tgt_feat = self.selfAttentionTgt(feature2, feature2, tgt_pe, tgt_pe).transpose(1, 2).contiguous()

        # root 2
        idx1 = pointops.knnquery(self.nsamples, xzy1_wrapped, xyz2)
        neigh_feature1 = pointops.grouping(src_feat, idx1.int()).contiguous()
        cat_feature2 = neigh_feature1.mean(dim=-1) - tgt_feat
        mask2 = self.mask(cat_feature2).transpose(1, 2).contiguous()
        # root 1
        idx2 = pointops.knnquery(self.nsamples, xyz2, xzy1_wrapped)
        neigh_xyz2 = pointops.grouping(xyz2.transpose(1, 2).contiguous(), idx2.int()).permute(0, 2, 3, 1).contiguous()
        neigh_feature2 = pointops.grouping(tgt_feat, idx2.int()).contiguous()
        cat_feature1 = neigh_feature2.mean(dim=-1) - src_feat
        mask1 = self.mask(cat_feature1).transpose(1, 2).contiguous()

        # pred xyz
        similarity = self.cos_simi(neigh_feature2, tgt_feat.unsqueeze(-1))
        similarity = similarity / similarity.sum(-1, keepdim=True)
        xyz1_pred = torch.sum(similarity.unsqueeze(-1) * neigh_xyz2, dim=2)
        return mask1, mask2, xyz1_pred


class transformer_local_flow_trans(nn.Module):
    """
    类似于光流与场景流，仅在近场进行搜索
    mask 特征向量是cat 出来的
    """

    def __init__(self, feature_dim, n_head, num_points, nsamples, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.num_points = num_points
        self.acti = nn.ReLU()
        self.nsamples = nsamples
        self.positional_encoding = PositionEncoding(feature_dim=self.d_model, pe_type="rotary")
        self.selfAttentionSrc = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.selfAttentionTgt = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.mask = nn.Sequential(nn.Conv1d(self.d_model * 2, self.d_model, 1, bias=False),
                                  nn.BatchNorm1d(self.d_model),
                                  nn.ReLU(),
                                  nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
                                  nn.BatchNorm1d(self.d_model),
                                  nn.ReLU(),
                                  nn.Conv1d(self.d_model, 1, 1, bias=False),
                                  nn.Sigmoid())
        self.cos_simi = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature1, feature2, xzy1_wrapped, xyz2):
        """
        Parameters
        ----------
        feature1: B, C, N
        feature2: B, C, N
        xzy1_wrapped: B, N, 3
        xyz2: B, N, 3

        Returns
        -------
        mask1: B, N, 1
        mask2: B, N, 1
        xyz1_pred: B, N, 3
        """
        bs, dims, num = feature1.shape
        feature1 = feature1.transpose(1, 2).contiguous()
        feature2 = feature2.transpose(1, 2).contiguous()
        src_pe = self.positional_encoding(xzy1_wrapped)
        tgt_pe = self.positional_encoding(xyz2)
        src_feat = self.selfAttentionSrc(feature1, feature1, src_pe, src_pe).transpose(1, 2).contiguous()
        tgt_feat = self.selfAttentionTgt(feature2, feature2, tgt_pe, tgt_pe).transpose(1, 2).contiguous()

        # # retation
        # src_pe_cos, src_pe_sin = src_pe[..., 0], src_pe[..., 1]
        # src_feat_pe = PositionEncoding.embed_rotary(src_feat, src_pe_cos, src_pe_sin).transpose(1, 2).contiguous()
        # tgt_pe_cos, tgt_pe_sin = tgt_pe[..., 0], tgt_pe[..., 1]
        # tgt_feat_pe = PositionEncoding.embed_rotary(tgt_feat, tgt_pe_cos, tgt_pe_sin).transpose(1, 2).contiguous()
        # root 2
        idx1 = pointops.knnquery(self.nsamples, xzy1_wrapped, xyz2)
        neigh_feature1 = pointops.grouping(src_feat, idx1.int()).contiguous()
        cat_feature2 = torch.cat([neigh_feature1.mean(dim=-1), tgt_feat], dim=1)
        mask2 = self.mask(cat_feature2).transpose(1, 2).contiguous()
        # root 1
        idx2 = pointops.knnquery(self.nsamples, xyz2, xzy1_wrapped)
        neigh_xyz2 = pointops.grouping(xyz2.transpose(1, 2).contiguous(), idx2.int()).permute(0, 2, 3, 1).contiguous()
        neigh_feature2 = pointops.grouping(tgt_feat, idx2.int()).contiguous()
        cat_feature1 = torch.cat([neigh_feature2.mean(dim=-1), src_feat], dim=1)
        mask1 = self.mask(cat_feature1).transpose(1, 2).contiguous()

        # pred xyz
        similarity = self.cos_simi(neigh_feature2, tgt_feat.unsqueeze(-1))
        similarity = similarity / similarity.sum(-1, keepdim=True)
        xyz1_pred = torch.sum(similarity.unsqueeze(-1) * neigh_xyz2, dim=2)
        return mask1, mask2, xyz1_pred


class NpCorrnetRt(nn.Module):
    def __init__(self, args=None, c=6):
        super().__init__()
        self.npoints = args.get('npoints', [1024, 512, 256, 64])  # C 1024 512 256 64
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.corr_nsample = args.get('corr_nsample', [8, 6, 4, 2])
        self.sa_mlps = args.get('sa_mlps',
                                [[c, 16, 32, 72], [72, 64, 128, 144], [144, 128, 128, 256],
                                 [256, 256, 256, 512]])  # 整除12
        self.radii = args.get('radii', [0.2, 0.2, 0.4, 0.6])
        self.filter = args.get('filter', False)
        self.SA_modules = nn.ModuleList()
        self.args = args
        self.corr_type = args.corr_type
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[0], nsample=self.nsamples[0], mlp=self.sa_mlps[0],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[1], nsample=self.nsamples[1], mlp=self.sa_mlps[1],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[2], nsample=self.nsamples[2], mlp=self.sa_mlps[2],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[3], nsample=self.nsamples[3], mlp=self.sa_mlps[3],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + 144, 256, self.sa_mlps[1][-1]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, self.sa_mlps[2][-1]]))

        self.global_registration_module = transformer_global_sample_trans(feature_dim=self.sa_mlps[1][-1], n_head=4,
                                                                   n_sample=16, procru_sample=8, args=args)
        self.local_corr_module = nn.ModuleList()
        self.local_corr_module.append(transformer_local_trans(feature_dim=self.sa_mlps[0][-1], n_head=4,
                                                              num_points=self.npoints[1], nsamples=32, args=args))
        self.local_corr_module.append(transformer_local_trans(feature_dim=self.sa_mlps[1][-1], n_head=4,
                                                              num_points=self.npoints[2], nsamples=16, args=args))

        self.FP_Final_modules = nn.ModuleList()
        self.FP_Final_modules.append(PTENet2FPMaskMtutalModule())
        self.FP_Final_modules.append(PTENet2FPMaskMtutalModule())

    def forward(self, pointxyz1, pointxyz2, colors1, colors2, trans_gt):
        """
        :param pointxyz1: source point cloud xyz [bs, num_points, 3]
        :param pointxyz2: target point cloud xyz
        :param colors1: source point cloud's color [bs, num_points, dims]
        :param colors2: target point cloud's color. The above two are used as feature
        """
        l_xyz1, l_features1 = [pointxyz1], [torch.cat([pointxyz1.permute(0, 2, 1), colors1.permute(0, 2, 1)], dim=1)]
        l_xyz2, l_features2 = [pointxyz2], [torch.cat([pointxyz2.permute(0, 2, 1), colors2.permute(0, 2, 1)], dim=1)]
        l_idx1 = []
        l_idx2 = []
        # calculate the feature of two points
        for i in range(len(self.SA_modules)):
            li_xyz1, li_features1, li_idx1 = self.SA_modules[i](l_xyz1[i], l_features1[i])
            # B N 3;   B C N; B N
            li_xyz2, li_features2, li_idx2 = self.SA_modules[i](l_xyz2[i], l_features2[i])
            l_xyz1.append(li_xyz1)  # [batch_size, num_points, 3]
            l_features1.append(li_features1)  # [batch_size, num_points, dims]
            l_xyz2.append(li_xyz2)
            l_features2.append(li_features2)
            l_idx1.append(li_idx1)
            l_idx2.append(li_idx2)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features1[i - 1] = self.FP_modules[i](l_xyz1[i - 1], l_xyz1[i], l_features1[i - 1], l_features1[i])
        #     l_features2[i - 1] = self.FP_modules[i](l_xyz2[i - 1], l_xyz2[i], l_features2[i - 1], l_features2[i])

        # global registration
        trans_pred = self.global_registration_module(l_features1[2], l_features2[2], l_xyz1[2], l_xyz2[2])
        for i in range(len(l_xyz1)):
            temp_xyz1 = l_xyz1[i]  # B N 3
            if trans_gt is not None:
                xzy1_wrapped = torch.matmul(trans_gt[0], temp_xyz1.transpose(1, 2)).transpose(1, 2) + trans_gt[1]
            else:
                xzy1_wrapped = torch.matmul(trans_pred[0], temp_xyz1.transpose(1, 2)).transpose(1, 2) + trans_pred[1]
            l_xyz1[i] = xzy1_wrapped.contiguous()
        # local registration
        l_flow = [None]
        l_mask1 = [None]
        l_mask2 = [None]
        for i in range(len(self.local_corr_module)):
            temp_mask1, temp_mask2, temp_pred = self.local_corr_module[i](l_features1[i + 1], l_features1[i + 1],
                                                                          l_xyz1[i + 1], l_xyz2[i + 1])
            temp_flow = temp_pred - l_xyz1[i + 1]
            l_flow.append(temp_flow)
            l_mask1.append(temp_mask1)
            l_mask2.append(temp_mask2)

        for i in range(-1, -(len(self.FP_Final_modules) + 1), -1):
            l_flow[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_Final_modules[i](unknown_xyz1=l_xyz1[i - 3],
                                                                                     known_xyz1=l_xyz1[i - 2],
                                                                                     unknown_xyz2=l_xyz2[i - 3],
                                                                                     known_xyz2=l_xyz2[i - 2],
                                                                                     unknown_new_xyz=l_flow[i - 1],
                                                                                     known_new_xyz=l_flow[i],
                                                                                     unknown_mask1=l_mask1[i - 1],
                                                                                     known_mask1=l_mask1[i],
                                                                                     unknown_mask2=l_mask2[i - 1],
                                                                                     known_mask2=l_mask2[i],
                                                                                     )
        l_new_xyz = []
        l_new_xyz.append(l_flow[0] + l_xyz1[0])
        l_new_xyz.append(l_flow[1] + l_xyz1[1])
        l_new_xyz.append(l_flow[2] + l_xyz1[2])
        return l_xyz1, l_new_xyz, l_idx1, l_idx2, l_mask1, l_mask2, trans_pred
