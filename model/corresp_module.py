# -*- coding: utf-8 -*-
# @Time : 2022/4/17 9:12
# @Author : 8515
# @File : correspondence.py
# @Project : corr_cgcpa
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import PositionEncoding, GeometryAttentionLayer, GeometryMaskAttentionLayer


# -----------------------------
# The correspondence
# -----------------------------


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


def cosine_simi(x, y):
    # batch_size, Num_points, C_feature
    B, N1, _ = x.size()
    N2 = y.size(1)
    result = torch.matmul(x, torch.transpose(y, 1, 2))
    norm_a = torch.linalg.norm(x, dim=2).reshape(B, -1, 1)
    norm_b = torch.linalg.norm(y, dim=2).reshape(B, 1, -1)
    borm_a = norm_a.repeat(1, 1, N2)
    borm_b = norm_b.repeat(1, N1, 1)
    similarity = result / borm_a
    similarity = similarity / borm_b
    return similarity


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


class SortPoint_SortMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.maskmlp2 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.sort(dim=-1, descending=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().sort(dim=-1, descending=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = similarity.sort(dim=-1, descending=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TopkPoint_TopkMask_Sort(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().topk(dim=-1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Softmax_SortPoint_SortMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.maskmlp2 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = F.softmax(similarity, -1).sort(dim=-1, descending=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(similarity, 1).permute(0, 2, 1).contiguous().sort(dim=-1, descending=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = F.softmax(similarity, -1).sort(dim=-1, descending=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Softmax_SortPoint_topMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().topk(dim=-1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = F.softmax(similarity, -1).sort(dim=-1, descending=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Softmax_TopkPoint_TopkMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = F.softmax(similarity, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(similarity, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = F.softmax(similarity, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class New(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = F.softmax(similarity, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(similarity, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp(topk_v2.contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = F.softmax(similarity, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Softmaxmask_TopkPoint_TopkMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = F.softmax(similarity, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(similarity, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Dual_Softmax_Topk(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        similarity = F.softmax(similarity, -1) * F.softmax(similarity, 1) * self.num_points

        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().topk(dim=-1, k=self.nsamples, sorted=False)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            similarity = F.softmax(similarity, -1) * F.softmax(similarity, 1) * self.num_points
            topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class Dual_Softmax(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.maskmlp2 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points * 2, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_points * 2, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        feature1_norm = feature1 / feature1.norm(dim=-1)[:, :, None]
        feature2_norm = feature2 / feature2.norm(dim=-1)[:, :, None]
        similarity = torch.bmm(feature1_norm, feature2_norm.transpose(1, 2))
        similarity = F.softmax(similarity, -1) * self.num_points * F.softmax(similarity, 1)
        topk_v1, topk_idx1 = similarity.sort(dim=-1, descending=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().sort(dim=-1, descending=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            similarity = F.softmax(similarity, -1) * F.softmax(similarity, 1) * self.num_points
            topk_v1, topk_idx1 = similarity.sort(dim=-1, descending=True)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TopkPoint_TopkMask(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().topk(dim=-1, k=self.nsamples, sorted=False)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TopkPoint(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
                      bias=False),
            nn.Sigmoid(),
        )
        self.maskmlp2 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=self.num_points, kernel_size=1, stride=1,
                      bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.num_points, out_channels=1, kernel_size=1, stride=1,
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        mask1 = self.maskmlp1(similarity).permute(0, 2, 1).contiguous()
        mask2 = self.maskmlp2(similarity.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TopkPoint_SortMask(TopkPoint):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super(TopkPoint_SortMask, self).__init__(num_points, args, nsamples, topk)

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.sort(dim=-1, descending=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().sort(dim=-1, descending=True)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TopkPoint_TopkMask_Bais(nn.Module):
    def __init__(self, num_points, args, nsamples=13, topk=True):
        super().__init__()
        self.num_points = num_points
        self.distance = pairwise_dist()
        self.nsamples = nsamples
        self.topk = topk
        self.fuse = args.fuse_type
        self.maskmlp1 = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.maskmlp2 = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.nsamples, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.DeSmooth_moudle = nn.Sequential(
            nn.Conv1d(in_channels=self.nsamples, out_channels=self.nsamples * 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.nsamples * 2, out_channels=self.nsamples, kernel_size=1, stride=1),
            nn.Softmax(dim=1))

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        similarity = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = similarity.permute(0, 2, 1).contiguous().topk(dim=-1, k=self.nsamples, sorted=False)
        mask2 = self.maskmlp2(topk_v2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.fuse == False:
            pass
        elif self.fuse == True:
            similarity = similarity * mask2.permute(0, 2, 1).contiguous() * mask1
            topk_v1, topk_idx1 = similarity.topk(dim=-1, k=self.nsamples, sorted=False)
        simik = self.DeSmooth_moudle(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        xyz2 = xyz2.unsqueeze(1).repeat(1, N, 1, 1).gather(2, topk_idx1.unsqueeze(-1).repeat(1, 1, 1, 3))  # B N K 3
        pred_xyz = torch.matmul(simik.unsqueeze(2), xyz2).squeeze(2)  # 1,5,3
        return pred_xyz, mask1, mask2


class TransformerMacthing(nn.Module):
    def __init__(self, feature_dim, n_head, num_points, nsamples, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.num_points = num_points
        self.acti = nn.ReLU()
        self.nsamples = nsamples
        self.corr_type = args.corr_type
        self.positional_encoding = PositionEncoding(feature_dim=self.d_model, pe_type="rotary")
        self.selfAttentionSrc = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.selfAttentionTgt = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.crossAttention = GeometryMaskAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz1, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        src_pe = self.positional_encoding(xyz1)
        tgt_pe = self.positional_encoding(xyz2)
        src_feat = self.selfAttentionSrc(feature1, feature1, src_pe, src_pe)
        tgt_feat = self.selfAttentionTgt(feature2, feature2, tgt_pe, tgt_pe)
        sim_matrix_1 = torch.einsum("bsc,btc->bst", src_feat, tgt_feat)
        topk_v1, topk_idx1 = F.softmax(sim_matrix_1, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(sim_matrix_1, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.contiguous()).permute(0, 2, 1).contiguous()

        pred_xyz = self.crossAttention(src_feat, tgt_feat, src_pe, tgt_pe, mask1, mask2, xyz2)
        return pred_xyz, mask1, mask2


class TransformerMacthingNoPosition(nn.Module):
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
        self.crossAttention = GeometryMaskAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz1, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        src_feat = self.selfAttentionSrc(feature1, feature1, None, None)
        tgt_feat = self.selfAttentionTgt(feature2, feature2, None, None)
        sim_matrix_1 = torch.einsum("bsc,btc->bst", src_feat, tgt_feat)
        topk_v1, topk_idx1 = F.softmax(sim_matrix_1, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(sim_matrix_1, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.contiguous()).permute(0, 2, 1).contiguous()
        pred_xyz = self.crossAttention(x=src_feat, tgt=tgt_feat, x_pe=None, tgt_pe=None, mask1=mask1, mask2=mask2,
                                       xyz2=xyz2)
        return pred_xyz, mask1, mask2


class Softmax_Transforme_SortMask(nn.Module):
    def __init__(self, feature_dim, n_head, num_points, nsamples, args, pe_type='rotary'):
        super().__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.num_points = num_points
        self.acti = nn.ReLU()
        self.nsamples = nsamples
        self.corr_type = args.corr_type
        self.distance = pairwise_dist()
        self.positional_encoding = PositionEncoding(feature_dim=self.d_model, pe_type="rotary")
        self.selfAttentionSrc = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.selfAttentionTgt = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.crossAttention = GeometryMaskAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.maskmlp1 = nn.Sequential(
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
        self.maskmlp2 = nn.Sequential(
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

    def forward(self, feature1, feature2, xyz2):
        """
        @param feature1: [ bs, num_points, dims]
        @param feature2 [ bs, num_points, dims]
        @return:
        """
        N = feature1.shape[1]
        sim_matrix_1 = self.distance(feature1, feature2)
        topk_v1, topk_idx1 = F.softmax(sim_matrix_1, -1).topk(dim=-1, k=self.nsamples, sorted=True)
        mask1 = self.maskmlp1(topk_v1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        topk_v2, topk_idx2 = F.softmax(sim_matrix_1, 1).topk(dim=1, k=self.nsamples, sorted=True)
        mask2 = self.maskmlp2(topk_v2.contiguous()).permute(0, 2, 1).contiguous()
        pred_xyz = self.crossAttention(x=feature1, tgt=feature2, x_pe=None, tgt_pe=None, mask1=mask1, mask2=mask2,
                                       xyz2=xyz2)
        return pred_xyz, mask1, mask2


if __name__ == "__main__":
    xyz1 = torch.randn(2, 2048, 3, requires_grad=True).cuda()
    xyz1_feats = torch.randn(2, 2048, 24, requires_grad=True).cuda()
    xyz2 = torch.randn(2, 2048, 3, requires_grad=True).cuda()
    xyz2_feats = torch.randn(2, 2048, 24, requires_grad=True).cuda()
    test_model = TopkPoint_TopkMask(24, 2048, 13).cuda()
    result = test_model(xyz1_feats, xyz2_feats, xyz2)

    print("Hello world!")
