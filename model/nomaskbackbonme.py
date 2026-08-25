#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/2 10:31
# @Author  : 沈子明
# @File    : nomaskbackbonme.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from model.pointnet2_paconv_modules import PointNet2SAModuleCUDA as PointNet2SAModule

class norm(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis, keepdim=True)
        std = torch.std(x, self.axis, keepdim=True)
        x = (x - mean) / (std + 1e-6)
        return x


class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 8

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis=axis)

    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x


class pairwise_dist(nn.Module):
    def __init__(self):
        super(pairwise_dist, self).__init__()
        self.activation = nn.ReLU()

    def forward(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        pc1^T * pc2 = xn * xm + yn * ym + zn * zm；
        sum(pc1^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(pc2^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2 = sum(pc1**2,dim=-1)+sum(pc2**2,dim=-1)-2*pc1^T*pc2
        Args:
            src: source point cloud
            dst: target point cloud
        Returns:
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        dist = self.activation(dist)
        similarity = 1 / (dist + 1e-6)
        return similarity
class DeSmooth(nn.Module):
    def __init__(self, num_points, nsamples=13, topk=True):
        super(DeSmooth, self).__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.acti = nn.ReLU()
        self.nsamples = nsamples
        self.topk = topk
        if self.topk:
            self.DeSmooth_moudle = nn.Sequential(
                nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1,
                          bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1,
                          bias=False),
                nn.Softmax(dim=1))
        else:
            self.DeSmooth_moudle = nn.Sequential(
                nn.Conv1d(in_channels=num_points, out_channels=num_points + 128, kernel_size=1, stride=1, bias=False),
                nn.ReLU(),
                norm(axis=1),
                nn.Conv1d(in_channels=num_points + 128, out_channels=num_points, kernel_size=1, stride=1, bias=False),
                Modified_softmax(axis=2)
            )

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        # similarity = cosine_simi(feature1, feature2)
        if self.topk:
            topk_v, topk_idx = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
            simik = self.DeSmooth_moudle(topk_v.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
            pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        else:
            similarity = self.DeSmooth_moudle(similarity.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            pred_xyz = torch.matmul(similarity, xyz2)
        # indices = torch.topk(similarity, k=int(self.num_points / 4), dim=2)[1].int().contiguous().squeeze()
        # pred_xyz = pointops.gathering(xyz2.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        return pred_xyz
class PointNet2FPModule(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self):
        super().__init__()
        self.fpmlp = torch.nn.Linear(6, 3)

    def forward(self, unknown_xyz, known_xyz, unknown_new_xyz, known_new_xyz):
        """
        all parameter [bs, num_points, 3]
        @param unknown_xyz:  N0 xyz positions of the source
        @param known_xyz:  N1 xyz positions of the target  N0>N1
        @param unknown_new_xyz: N0 xyz positions of the source
        @param known_new_xyz:  N1 xyz positions of the target  N0>N1
        @return:
        """
        dist, idx = pointops.nearestneighbor(unknown_xyz, known_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_xyzs = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx, weight)
        interpolated_xyzs = interpolated_xyzs.transpose(2, 1).contiguous()

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyzs, unknown_new_xyz], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyzs.repeat(1, 1, 2)

        new_xyz = self.fpmlp(new_xyzs)

        return new_xyz

class PointNet2Corr(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Parameters
        ----------
        c: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in a Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether to use the xyz position of a point as a feature
    """

    def __init__(self, c=6, use_xyz=True, args=None):
        super().__init__()
        self.npoints = args.get('npoints', [2048, 512, 256, 64])
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.sa_mlps = args.get('sa_mlps',
                                [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
        self.radii = args.get('radii', [0.2, 0.2, 0.4, 0.6])
        self.use_color = args.use_color
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[0], nsample=self.nsamples[0], mlp=self.sa_mlps[0],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[1], nsample=self.nsamples[1], mlp=self.sa_mlps[1],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[2], nsample=self.nsamples[2], mlp=self.sa_mlps[2],
                                                 use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[3], nsample=self.nsamples[3], mlp=self.sa_mlps[3],
                                                 use_xyz=False, use_paconv=True, args=args))

        self.corr_modules = nn.ModuleList()
        self.corr_modules.append(DeSmooth(self.npoints[0], nsamples=8, topk=args.top_k))
        self.corr_modules.append(DeSmooth(self.npoints[1], nsamples=6, topk=args.top_k))
        self.corr_modules.append(DeSmooth(self.npoints[2], nsamples=4, topk=args.top_k))
        self.corr_modules.append(DeSmooth(self.npoints[3], nsamples=2, topk=args.top_k))

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule())
        self.FP_modules.append(PointNet2FPModule())
        self.FP_modules.append(PointNet2FPModule())
        self.FP_modules.append(PointNet2FPModule())

    def forward(self, pointxyz1, pointxyz2, colors1, colors2):
        """
        :param pointxyz1: source point cloud xyz [bs, num_points, 3]
        :param pointxyz2: target point cloud xyz
        :param colors1: source point cloud's color [bs, num_points, dims]
        :param colors2: target point cloud's color. The above two are used as feature
        :param gt:
        @return:

        """
        l_xyz1, l_features1 = [pointxyz1], [torch.cat([pointxyz1.permute(0, 2, 1), colors1.permute(0, 2, 1)], dim=1)]
        l_xyz2, l_features2 = [pointxyz2], [torch.cat([pointxyz2.permute(0, 2, 1), colors2.permute(0, 2, 1)], dim=1)]
        l_idx = []
        # calculate the feature of two points
        for i in range(len(self.SA_modules)):
            li_xyz1, li_features1, li_idx = self.SA_modules[i](l_xyz1[i], l_features1[i])
            # B N 3;   B C N; B N
            li_xyz2, li_features2, _ = self.SA_modules[i](l_xyz2[i], l_features2[i])
            l_xyz1.append(li_xyz1)  # [batch_size, num_points, 3]
            l_features1.append(li_features1)  # [batch_size, num_points, dims]
            l_xyz2.append(li_xyz2)
            l_features2.append(li_features2)
            l_idx.append(li_idx)

        # calculate the corresponding points of different layers
        l_tempnewxyz = [None]
        for i in range(len(self.corr_modules)):
            temp_newxyz = self.corr_modules[i](l_features1[i + 1].permute(0, 2, 1).contiguous(),
                                               l_features2[i + 1].permute(0, 2, 1).contiguous(),
                                               l_xyz2[i + 1])
            l_tempnewxyz.append(temp_newxyz)

        # upsamlping
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_tempnewxyz[i - 1] = self.FP_modules[i](unknown_xyz=l_xyz1[i - 1], known_xyz=l_xyz1[i],
                                                     unknown_new_xyz=l_tempnewxyz[i - 1], known_new_xyz=l_tempnewxyz[i])
        return l_xyz1, l_tempnewxyz, l_idx