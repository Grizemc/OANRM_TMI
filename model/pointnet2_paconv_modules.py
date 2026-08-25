from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pointops.functions import pointops
from util import block
from model import paconv


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        for i in range(len(self.groupers)):
            new_features, grouped_xyz, _ = self.groupers[i](xyz, new_xyz, features)
            # (B, Cin+3, N1, K), (B, 3, N1, K)
            if isinstance(self.mlps[i], paconv.SharedPAConv):
                new_features = self.mlps[i]((new_features, grouped_xyz))[0]  # (B, Cout, N1, K)
            else:
                new_features = self.mlps[i](new_features)  # (B, Cout, N1, K)
            if self.agg == 'max':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(-1)])  # (B, Cout, N1, 1)
            elif self.agg == 'sum':
                new_features = torch.sum(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            elif self.agg == 'avg':
                new_features = torch.mean(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            else:
                raise ValueError('Not implemented aggregation mode.')
            new_features = new_features.squeeze(-1)  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, voxel_size=None, args=None):
        # PointNet2SAModuleCUDA调用PointNet2SAModuleMSG父类的时候，会将mlps=[mlp]传入，即 mlps=[[6,16,,32,64]]等
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.use_xyz = use_xyz
        self.agg = args.get('agg', 'max')
        self.sampling = args.get('sampling', 'fps')
        self.voxel_size = voxel_size
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 添加不同的groupers
            self.groupers.append(
                pointops.QueryAndGroup(radius, nsample, use_xyz=use_xyz, return_idx=True)
                # if npoint is not None else pointops.GroupAll(use_xyz=use_xyz)
            )
            # mlp_spec = [6,16,32,64]
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if use_paconv:
                # mlps的添加唯一 此时 多层paconv模型已经实例化好并添加到了列表mlps中。
                self.mlps.append(paconv.SharedPAConv(mlp_spec, bn=bn, config=args))
            else:
                self.mlps.append(block.SharedMLP(mlp_spec, bn=bn))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, args=None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv, args=args)


class PointNet2SAModuleCUDA(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, args=None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv, args=args)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N0) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        # 从xyz原始点进行采样，采样到npoint的个数的点
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,   # B 3 N0
            new_xyz_idx  # B N1
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        new_features = features
        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                # 特征提取 -Paconv 寻找领域点
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features)
                # (B, Cin+3, N1, K), grouped_xyz：(B, 3, N1, K), grouped_idx：(B, N1, K)  new_xyz：B x N_1 x 3
                # use_xyz = false
                if self.use_xyz and j == 0:
                    new_features = torch.cat((xyz.permute(0, 2, 1), new_features), dim=1)
                if isinstance(self.mlps[i], paconv.SharedPAConv):
                    # 将new_features，即上层的特征f，以邻域点grouped_xyz等送入mlp进行处理
                    # j 对应c, 16, 32, 64，每次迭代，new_features更换一次特征维度，以此进行mlp的计算

                    '''class SharedPAConv(nn.Sequential):

                    def __init__(
                            self,
                            args: List[int],
                            *,
                            config,
                            bn: bool = False,
                            activation=nn.ReLU(inplace=True),
                            preact: bool = False,
                            first: bool = False,
                            name: str = "",
                    ):
                        super().__init__()
                
                        for i in range(len(args) - 1):
                            if config.get('cuda', False):
                                self.add_module(
                                    name + 'layer{}'.format(i),
                                    PAConvCUDA(
                                        # 6,16
                                        # 16,32
                                        # 32,64
                                        args[i],
                                        args[i + 1],
                                        bn=(not first or not preact or (i != 0)) and bn,
                                        activation=activation
                                        if (not first or not preact or (i != 0)) else None,
                                        config=config,
                                    )
                                )'''
                    # new_features为前一尺度的特征，grouped_new_features为池化前的后一尺度的特征. -> 6 16 32 64
                    # self.mlps[i][j] self.mlps[0]为SharedPAConv()网络，即包括了不同输入输出通道的PAConvCUDA
                    grouped_new_features = self.mlps[i][j]((new_features, grouped_xyz, grouped_idx))[0]  # (B, Cout, N1, K)
                else:
                    raise NotImplementedError
                if self.agg == 'max':
                    # 论文中的max操作 grouped_new_features.size(3) 代表 K，即邻域点的数量。
                    # 当 kernel_size 的第一个值大于 1 时，池化操作会在特征维度上进行，这样会减少特征通道数。
                    new_features = F.max_pool2d(grouped_new_features, kernel_size=[1, grouped_new_features.size(3)])  # (B, Cout, N1, 1)
                elif self.agg == 'sum':
                    new_features = torch.sum(grouped_new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                else:
                    raise ValueError('Not implemented aggregation mode.')
                # 将new_xyz的对象赋给xyz
                xyz = new_xyz
                new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
            #  [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]]
            # 每次j的循环，获得最后一层的特征，也就是64 或者 128 或者 256 或者 512
            # new_features_list 为每个grouper特征的列表。  一个grouper用一种radius和nsample  即采样的距离和点数。
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), new_xyz_idx

class PointNet2FPModule(nn.Module):
    r"""Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True,  use_paconv=False, args=None):
        super().__init__()
        self.use_paconv = use_paconv
        if self.use_paconv:
            self.mlp = paconv.SharedPAConv(mlp, bn=bn, config=args)
        else:
            self.mlp = block.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)



#  以下由沈子明撰写
class DaulPointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, voxel_size=None, args=None):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.use_xyz = use_xyz
        self.agg = args.get('agg', 'max')
        self.sampling = args.get('sampling', 'fps')
        self.voxel_size = voxel_size
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup(radius, nsample, use_xyz=use_xyz, return_idx=True)
                # if npoint is not None else pointops.GroupAll(use_xyz=use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if use_paconv:
                self.mlps.append(paconv.SharedDaulPAConv(mlp_spec, bn=bn, config=args))
            else:
                self.mlps.append(block.SharedMLP(mlp_spec, bn=bn))

class DualPointNet2Module(DaulPointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, args=None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv, args=args)

    def forward(self, xyz, features_xyz, features_rgb) :
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features_xyz : torch.Tensor
            (B, Cin, N0) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list_xyz = []
        new_features_list_rgb = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans, # B 3 N0
            new_xyz_idx # B N1
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        new_features_xyz = features_xyz
        new_features_rgb = features_rgb
        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features_xyz)
                # (B, Cin+3, N1, K), (B, 3, N1, K), (B, N1, K)
                if self.use_xyz and j == 0:
                    new_features_xyz = torch.cat((xyz.permute(0, 2, 1), new_features_xyz), dim=1)
                if isinstance(self.mlps[i], paconv.SharedDaulPAConv):
                    grouped_new_features_xyz,  grouped_new_features_rgb,_,_ = self.mlps[i][j]((
                        new_features_xyz, new_features_rgb, grouped_xyz, grouped_idx))  # (B, Cout, N1, K)
                else:
                    raise NotImplementedError
                if self.agg == 'max':
                    new_features_xyz = F.max_pool2d(grouped_new_features_xyz, kernel_size=[1, grouped_new_features_xyz.size(3)])  # (B, Cout, N1, 1)
                    new_features_rgb = F.max_pool2d(grouped_new_features_rgb, kernel_size=[1, grouped_new_features_rgb.size(3)])  # (B, Cout, N1, 1)
                elif self.agg == 'sum':
                    new_features_xyz = torch.sum(grouped_new_features_xyz, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                    new_features_rgb = torch.sum(grouped_new_features_rgb, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                else:
                    raise ValueError('Not implemented aggregation mode.')
                xyz = new_xyz
                new_features_xyz = new_features_xyz.squeeze(-1).contiguous()  # (B, Cout, N1)
                new_features_rgb = new_features_rgb.squeeze(-1).contiguous()  # (B, Cout, N1)
            new_features_list_xyz.append(new_features_xyz)
            new_features_list_rgb.append(new_features_rgb)
        return new_xyz, torch.cat(new_features_list_xyz, dim=1), torch.cat(new_features_list_rgb, dim=1), new_xyz_idx




if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(2, 9, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 9, 6, requires_grad=True).cuda()

    test_module = PointNet2SAModuleMSG(npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]])
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    # test_module = PointNet2FPModule(mlp=[6, 6])
    # test_module.cuda()
    # from torch.autograd import gradcheck
    # inputs = (xyz, xyz, None, xyz_feats)
    # test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    # print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
