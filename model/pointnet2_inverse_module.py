import torch
from torch import nn
from lib.pointops.functions import pointops
from model import paconv


class PTENet2FPflowModule(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self, args):
        super().__init__()
        self.up_type = args.up_type

        if args.up_type == "mask_fuse":
            self.f_mask1pmlp = nn.Sequential(nn.Linear(8, 32), nn.PReLU(init=0.5), nn.Linear(32, 16),
                                             nn.PReLU(init=0.5),
                                             nn.Linear(16, 4))
            self.maskmlp2 = nn.Sequential(nn.Linear(2, 12), nn.PReLU(), nn.Linear(12, 6), nn.PReLU(), nn.Linear(6, 1),
                                          nn.PReLU())
        else:
            self.fpmlp = nn.Sequential(nn.Linear(6, 24), nn.PReLU(init=0.5), nn.Linear(24, 6), nn.PReLU(init=0.5),
                                       nn.Linear(6, 3))
            self.maskmlp1 = nn.Sequential(nn.Linear(2, 12), nn.PReLU(), nn.Linear(12, 6), nn.PReLU(), nn.Linear(6, 1),
                                          nn.PReLU())
            self.maskmlp2 = nn.Sequential(nn.Linear(2, 12), nn.PReLU(), nn.Linear(12, 6), nn.PReLU(), nn.Linear(6, 1),
                                          nn.PReLU())

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz,
                unknown_mask1, known_mask1, unknown_mask2, known_mask2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
        if self.up_type == "mask_fuse":
            xyzs_mask1 = torch.cat([new_xyzs, new_mask1s], dim=2)
            f_mask1pmlp = self.f_mask1pmlp(xyzs_mask1)
            new_xyz = f_mask1pmlp[:, :, :3]
            new_mask1 = f_mask1pmlp[:, :, 3:]
            new_mask2 = self.maskmlp2(new_mask2s)
        else:
            new_xyz = self.fpmlp(new_xyzs)
            new_mask1 = self.maskmlp1(new_mask1s)
            new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2


class PTENet2FPMaskMtutalModule(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self):
        super().__init__()
        self.fpmlp = nn.Sequential(nn.Linear(6, 24), nn.PReLU(init=0.5), nn.Linear(24, 6), nn.PReLU(init=0.5),
                                   nn.Linear(6, 3))
        self.maskmlp1 = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 12), nn.Tanh(), nn.Linear(12, 1))
        self.maskmlp2 = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz,
                unknown_mask1, known_mask1, unknown_mask2, known_mask2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1

        # interpolated_xyz 是从 known_new_xyz 这个形变流上插值而来到下一层，unknown_new_xyz则是下一层的真实形变流
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
        new_xyz = self.fpmlp(new_xyzs)
        new_mask1 = self.maskmlp1(new_mask1s)
        new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2

class Upmask(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self):
        super().__init__()
        self.fpmlp = nn.Sequential(nn.Linear(6, 24),
                                   nn.PReLU(init=0.5),
                                   nn.Linear(24, 6),
                                   nn.PReLU(init=0.5),
                                   nn.Linear(6, 3))
        self.maskmlp1 = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 12), nn.Tanh(), nn.Linear(12, 1))
        self.maskmlp2 = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz,
                unknown_mask1, known_mask1, unknown_mask2, known_mask2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()

        interpolated_xyz *= interpolated_mask1
        if unknown_new_xyz is not None:
            unknown_new_xyz *= unknown_mask1
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
        new_xyz = self.fpmlp(new_xyzs)
        new_mask1 = self.maskmlp1(new_mask1s)
        new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2
class MaskFuse(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self):
        super().__init__()
        self.f_mask_mlp1 = nn.Sequential(nn.Linear(8, 32), nn.PReLU(init=0.5), nn.Linear(32, 16), nn.PReLU(init=0.5),
                                         nn.Linear(16, 4))
        self.maskmlp2 = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz,
                unknown_mask1, known_mask1, unknown_mask2, known_mask2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
        xzy_mask = torch.cat([new_xyzs, new_mask1s], dim=2)  # (B, n, 6)
        new_xzy_mask = self.f_mask_mlp1(xzy_mask)
        new_xyz = new_xzy_mask[:, :, :3]
        new_mask1 = new_xzy_mask[:, :, 3:]
        new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2


class feature_mask_fuse(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self, mlps):
        super().__init__()
        self.mlp_first = mlps[0]
        self.mlp_last = mlps[-1]
        self.mask_input_dim = self.mlp_first + self.mlp_last + 2
        self.match_input_dim = self.mlp_first + self.mlp_last + 8
        self.f_mask_mlp1 = nn.Sequential(nn.Linear(self.match_input_dim, self.mlp_last + 6),
                                         nn.PReLU(init=0.5),
                                         nn.Linear(self.mlp_last + 6, self.mlp_first + 3),
                                         nn.PReLU(init=0.5),
                                         nn.Linear(self.mlp_first + 3, 4))
        self.maskmlp2 = nn.Sequential(nn.Linear(self.mask_input_dim, self.mlp_last + 2),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_last + 2, self.mlp_first + 1),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_first + 1, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz, unknown_mask1,
                known_mask1, unknown_mask2, known_mask2, unknown_feature1, known_feature1, unknown_feature2,
                known_feature2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()
        interpolated_feature1 = pointops.interpolation(known_feature1, idx1, weight1)
        interpolated_feature1 = interpolated_feature1

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()
        interpolated_feature2 = pointops.interpolation(known_feature2, idx2, weight1)
        interpolated_feature2 = interpolated_feature2

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
            new_feature1s = torch.cat([interpolated_feature1, unknown_feature1], dim=1)
            new_feature2s = torch.cat([interpolated_feature2, unknown_feature2], dim=1)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
            new_feature1s = interpolated_feature1.repeat(1, 2, 1)
            new_feature2s = interpolated_feature2.repeat(1, 2, 1)

        xzy_mask = torch.cat([new_feature1s.transpose(2, 1).contiguous(), new_xyzs, new_mask1s], dim=2)  # (B, n, 6)
        new_xzy_mask = self.f_mask_mlp1(xzy_mask)
        new_xyz = new_xzy_mask[:, :, :3]
        new_mask1 = new_xzy_mask[:, :, 3:]

        new_mask2s_feature = torch.cat([new_feature2s.transpose(2, 1).contiguous(), new_mask2s], dim=2)
        new_mask2 = self.maskmlp2(new_mask2s_feature)
        return new_xyz, new_mask1, new_mask2

class feature_onlymask_fuse(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self, mlps):
        super().__init__()
        self.mlp_first = mlps[0]
        self.mlp_last = mlps[-1]
        self.mask_input_dim = self.mlp_first + self.mlp_last + 2
        self.f_mlp1  = nn.Sequential(nn.Linear(6, 24), nn.PReLU(init=0.5), nn.Linear(24, 6), nn.PReLU(init=0.5),
                                   nn.Linear(6, 3))
        self.maskmlp1 = nn.Sequential(nn.Linear(self.mask_input_dim, self.mlp_last + 2),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_last + 2, self.mlp_first + 1),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_first + 1, 1))
        self.maskmlp2 = nn.Sequential(nn.Linear(self.mask_input_dim, self.mlp_last + 2),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_last + 2, self.mlp_first + 1),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_first + 1, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz, unknown_mask1,
                known_mask1, unknown_mask2, known_mask2, unknown_feature1, known_feature1, unknown_feature2,
                known_feature2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()
        interpolated_feature1 = pointops.interpolation(known_feature1, idx1, weight1)
        interpolated_feature1 = interpolated_feature1

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()
        interpolated_feature2 = pointops.interpolation(known_feature2, idx2, weight1)
        interpolated_feature2 = interpolated_feature2

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
            new_feature1s = torch.cat([interpolated_feature1, unknown_feature1], dim=1)
            new_feature2s = torch.cat([interpolated_feature2, unknown_feature2], dim=1)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
            new_feature1s = interpolated_feature1.repeat(1, 2, 1)
            new_feature2s = interpolated_feature2.repeat(1, 2, 1)

        new_xyz = self.f_mlp1(new_xyzs) # (B, n, 6)
        new_mask1s_feature = torch.cat([new_feature1s.transpose(2, 1).contiguous(), new_mask1s], dim=2)
        new_mask1 = self.maskmlp2(new_mask1s_feature)
        new_mask2s_feature = torch.cat([new_feature2s.transpose(2, 1).contiguous(), new_mask2s], dim=2)
        new_mask2 = self.maskmlp2(new_mask2s_feature)
        return new_xyz, new_mask1, new_mask2

class BaseUpRelu(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self):
        super().__init__()
        self.fpmlp = nn.Sequential(nn.Linear(6, 24), nn.PReLU(init=0.5), nn.Linear(24, 6), nn.PReLU(init=0.5),
                                   nn.Linear(6, 3))
        self.maskmlp1 = nn.Sequential(nn.Linear(2, 24), nn.PReLU(init=0.5), nn.Linear(24, 12), nn.PReLU(init=0.5),
                                      nn.Linear(12, 1))
        self.maskmlp2 = nn.Sequential(nn.Linear(2, 24), nn.PReLU(init=0.5), nn.Linear(24, 12), nn.PReLU(init=0.5),
                                      nn.Linear(12, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz,
                unknown_mask1, known_mask1, unknown_mask2, known_mask2):
        """
        all parameter [bs, num_points, 3]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_xyz = interpolated_xyz.transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1)
        interpolated_mask1 = interpolated_mask1.transpose(2, 1).contiguous()

        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2)
        interpolated_mask2 = interpolated_mask2.transpose(2, 1).contiguous()

        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_xyz, unknown_new_xyz], dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1], dim=2)  # (B, n, 6)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2], dim=2)  # (B, n, 6)
        else:
            new_xyzs = interpolated_xyz.repeat(1, 1, 2)
            new_mask1s = interpolated_mask1.repeat(1, 1, 2)
            new_mask2s = interpolated_mask2.repeat(1, 1, 2)
        new_xyz = self.fpmlp(new_xyzs)
        new_mask1 = self.maskmlp1(new_mask1s)
        new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2


class FeatureFuse(nn.Module):
    r"""
    Propagates the features of one set to another
    """

    def __init__(self, mlps):
        super().__init__()
        self.mlp_first = mlps[0]
        self.mlp_last = mlps[-1]
        self.mask_input_dim = self.mlp_first + self.mlp_last + 2
        self.match_input_dim = self.mlp_first + self.mlp_last + 6 # 上一层的输出和当前层的输入维度
        self.fpmlp = nn.Sequential(nn.Linear(self.match_input_dim, self.mlp_last + 6),
                                   nn.PReLU(init=0.5),
                                   nn.Linear(self.mlp_last + 6, self.mlp_first + 3),
                                   nn.PReLU(init=0.5),
                                   nn.Linear(self.mlp_first + 3, 3))
        self.maskmlp1 = nn.Sequential(nn.Linear(self.mask_input_dim, self.mlp_last + 2),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_last + 2, self.mlp_first + 1),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_first + 1, 1))
        self.maskmlp2 = nn.Sequential(nn.Linear(self.mask_input_dim, self.mlp_last + 2),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_last + 2, self.mlp_first + 1),
                                      nn.PReLU(init=0.5),
                                      nn.Linear(self.mlp_first + 1, 1))

    def forward(self, unknown_xyz1, known_xyz1, unknown_xyz2, known_xyz2, unknown_new_xyz, known_new_xyz, unknown_mask1,
                known_mask1, unknown_mask2, known_mask2, unknown_feature1, known_feature1, unknown_feature2,
                known_feature2):
        """
        all parameter [bs, num_points, 3]
        features [B C N]
        @return:
        """
        dist1, idx1 = pointops.nearestneighbor(unknown_xyz1, known_xyz1)
        dist_recip1 = 1.0 / (dist1 + 1e-8)
        norm1 = torch.sum(dist_recip1, dim=2, keepdim=True)
        weight1 = dist_recip1 / norm1
        interpolated_pred_xyz = pointops.interpolation(known_new_xyz.transpose(2, 1).contiguous(), idx1,
                                                       weight1).transpose(2, 1).contiguous()
        interpolated_mask1 = pointops.interpolation(known_mask1.transpose(2, 1).contiguous(), idx1, weight1).transpose(
            2, 1).contiguous()
        interpolated_feature1 = pointops.interpolation(known_feature1, idx1, weight1).transpose(2, 1).contiguous()
        
        dist2, idx2 = pointops.nearestneighbor(unknown_xyz2, known_xyz2)
        dist_recip2 = 1.0 / (dist2 + 1e-8)
        norm2 = torch.sum(dist_recip2, dim=2, keepdim=True)
        weight2 = dist_recip2 / norm2
        interpolated_mask2 = pointops.interpolation(known_mask2.transpose(2, 1).contiguous(), idx2, weight2).transpose(
            2, 1).contiguous()
        interpolated_feature2 = pointops.interpolation(known_feature2, idx2, weight2).transpose(2, 1).contiguous()

        unknown_feature1 = unknown_feature1.transpose(2, 1).contiguous()
        unknown_feature2 = unknown_feature2.transpose(2, 1).contiguous()
        if unknown_new_xyz is not None:
            new_xyzs = torch.cat([interpolated_pred_xyz, unknown_new_xyz, interpolated_feature1, unknown_feature1],
                                 dim=2)  # (B, n, 6)
            new_mask1s = torch.cat([interpolated_mask1, unknown_mask1, interpolated_feature1, unknown_feature1], dim=2)
            new_mask2s = torch.cat([interpolated_mask2, unknown_mask2, interpolated_feature2, unknown_feature2], dim=2)
        else:
            new_xyzs = torch.cat([interpolated_pred_xyz.repeat(1, 1, 2), interpolated_feature1.repeat(1, 1, 2)], dim=2)
            new_mask1s = torch.cat([interpolated_mask1.repeat(1, 1, 2), interpolated_feature1.repeat(1, 1, 2)], dim=2)
            new_mask2s = torch.cat([interpolated_mask2.repeat(1, 1, 2), interpolated_feature2.repeat(1, 1, 2)], dim=2)

        new_xyz = self.fpmlp(new_xyzs)
        new_mask1 = self.maskmlp1(new_mask1s)
        new_mask2 = self.maskmlp2(new_mask2s)
        return new_xyz, new_mask1, new_mask2


if __name__ == "__main__":
    print("Hello")
