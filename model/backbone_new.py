import torch
import torch.nn as nn
from lib.pointops.functions.pointops import knnquery_heap, grouping
from model.backbone_lepard import batch_weighted_procrustes
from model.corresp_module import TopkPoint_TopkMask, TopkPoint_TopkMask_Bais, TopkPoint, SortPoint_SortMask, \
    TopkPoint_TopkMask_Sort, TopkPoint_SortMask, Softmax_TopkPoint_TopkMask, Dual_Softmax_Topk, Dual_Softmax, \
    Softmax_SortPoint_SortMask, Softmaxmask_TopkPoint_TopkMask, Softmax_SortPoint_topMask, TransformerMacthing, \
    TransformerMacthingNoPosition, Softmax_Transforme_SortMask, New
from model.loss import one_loss
from model.pointnet2.pointnet2_modules import PointNet2FPModule
from model.pointnet2_inverse_module import PTENet2FPMaskMtutalModule, FeatureFuse, PTENet2FPflowModule, BaseUpRelu, \
    MaskFuse, feature_mask_fuse, feature_onlymask_fuse, Upmask
from model.pointnet2_paconv_modules import PointNet2SAModuleCUDA as PointNet2SAModule
from lib.pointops.functions import pointops
import torch.nn.functional as F


class PTFlowmean(nn.Module):
    r"""
    """

    def __init__(self, args=None, c=6):
        super().__init__()
        self.npoints = args.get('npoints', [1024, 512, 256, 64])
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.corr_nsample = args.get('corr_nsample', [8, 6, 4, 2])
        self.sa_mlps = args.get('sa_mlps',
                                [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
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
        self.corr_modules = nn.ModuleList()
        if args.corr_type == "topkpoint_topkmask":
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_topkmask":
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing":
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0], nsamples=16,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1], nsamples=8,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2], nsamples=4,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3], nsamples=2,
                                    args=args))
        elif args.corr_type == "softmax_transformer_topkmask":
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                            nsamples=16, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                            nsamples=8, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                            nsamples=4, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                            nsamples=2, args=args))
        elif args.corr_type == "softmax_topkpoint_topkmask":
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "New":
            self.corr_modules.append(New(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing_no_position":
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                              nsamples=16,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                              nsamples=8,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                              nsamples=4,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                              nsamples=2,
                                              args=args))
        elif args.corr_type == "softmaxmask_topkpoint_topkmask":
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_sortmask":
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_sort":
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[0], nsamples=self.corr_nsample[0], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[1], nsamples=self.corr_nsample[1], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[2], nsamples=self.corr_nsample[2], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[3], nsamples=self.corr_nsample[3], topk=args.top_k, args=args))
        elif args.corr_type == "sortpoint_sortmask":
            self.corr_modules.append(SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_bais":
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint":
            self.corr_modules.append(TopkPoint(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax":
            self.corr_modules.append(Dual_Softmax(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax_topk":
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_sortmask":
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        else:
            raise Exception("Corr module no imple")
        self.FP_modules = nn.ModuleList()
        if args.up_moudle_type == "feature_fuse":
            self.FP_modules.append(FeatureFuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "feature_mask_fuse":
            self.FP_modules.append(feature_mask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "up_mask":
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
        elif args.up_moudle_type == "feature_onlymask_fuse":
            self.FP_modules.append(feature_onlymask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "source":
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
        elif args.up_moudle_type == "mask_fuse":
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
        else:
            raise Exception("up module no imple")

    def forward(self, pointxyz1, pointxyz2, colors1, colors2):
        """
        :param pointxyz1: source point cloud xyz [bs, num_points, 3]
        :param pointxyz2: target point cloud xyz
        :param colors1: source point cloud's color [bs, num_points, dims]
        :param colors2: target point cloud's color. The above two are used as feature
        """
        mask_point1_mean = torch.mean(pointxyz1, dim=1, keepdim=True)
        mask_point2_mean = torch.mean(pointxyz2, dim=1, keepdim=True)
        pointxyz1 = pointxyz1 - mask_point1_mean
        pointxyz2 = pointxyz2 - mask_point2_mean
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

        # calculate the corresponding points of different layers
        l_new_xyz = [None]
        l_mask1 = [None]
        l_mask2 = [None]
        for i in range(len(self.corr_modules)):
            if self.corr_type == "TransformerMacthing" or self.corr_type == "TransformerMacthing_no_position" or self.corr_type == "softmax_topkpoint_topkmask_coordinate":
                temp_newxyz, temp_mask1, temp_mask2 = self.corr_modules[i](
                    l_features1[i + 1].permute(0, 2, 1).contiguous(),
                    l_features2[i + 1].permute(0, 2, 1).contiguous(),
                    l_xyz1[i + 1],
                    l_xyz2[i + 1])
            else:
                temp_newxyz, temp_mask1, temp_mask2 = self.corr_modules[i](
                    l_features1[i + 1].permute(0, 2, 1).contiguous(),
                    l_features2[i + 1].permute(0, 2, 1).contiguous(),
                    l_xyz2[i + 1])

            # 这里使用temp_flow形变场而不是直接用temp_newxyz，是为了方便后面计算上采样模块。
            temp_flow = temp_newxyz - l_xyz1[i + 1]
            l_new_xyz.append(temp_flow)
            l_mask1.append(temp_mask1)
            l_mask2.append(temp_mask2)

        # upsamlping
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            if self.args.up_moudle_type == "feature_fuse" or self.args.up_moudle_type == "feature_mask_fuse" or self.args.up_moudle_type == "feature_onlymask_fuse":
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      unknown_feature1=l_features1[
                                                                                          i - 1],
                                                                                      known_feature1=l_features1[i],
                                                                                      unknown_feature2=l_features2[
                                                                                          i - 1],
                                                                                      known_feature2=l_features1[i]
                                                                                      )
            else:
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      )
        # for i in range(len(l_new_xyz)):
        #     l_new_xyz[i] = l_new_xyz[i] + l_xyz1[i]
        # l_new_xyz中存储的是形变流，然后加在原始点l_xyz1上，就得到了最终的对应点。
        l_new_xyz[0] = l_new_xyz[0] + l_xyz1[0]
        l_new_xyz[1] = l_new_xyz[1] + l_xyz1[1]
        l_new_xyz[2] = l_new_xyz[2] + l_xyz1[2]
        l_new_xyz[3] = l_new_xyz[3] + l_xyz1[3]
        l_new_xyz[4] = l_new_xyz[4] + l_xyz1[4]
        l_new_xyz[0] = l_new_xyz[0] + mask_point2_mean
        return l_xyz1, l_new_xyz, l_idx1, l_idx2, l_mask1, l_mask2


class PTFlow(nn.Module):
    r"""
    """

    def __init__(self, args=None, c=6):
        super().__init__()
        self.npoints = args.get('npoints', [1024, 512, 256, 64])
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.corr_nsample = args.get('corr_nsample', [8, 6, 4, 2])
        self.sa_mlps = args.get('sa_mlps',
                                [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
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
        self.corr_modules = nn.ModuleList()
        if args.corr_type == "topkpoint_topkmask":
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_topkmask":
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Softmax_SortPoint_topMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing":
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0], nsamples=16,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1], nsamples=8,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2], nsamples=4,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3], nsamples=2,
                                    args=args))
        elif args.corr_type == "softmax_transformer_topkmask":
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                            nsamples=16, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                            nsamples=8, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                            nsamples=4, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                            nsamples=2, args=args))
        elif args.corr_type == "softmax_topkpoint_topkmask":
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "New":
            self.corr_modules.append(New(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing_no_position":
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                              nsamples=16,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                              nsamples=8,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                              nsamples=4,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                              nsamples=2,
                                              args=args))
        elif args.corr_type == "softmaxmask_topkpoint_topkmask":
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_sortmask":
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_sort":
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[0], nsamples=self.corr_nsample[0], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[1], nsamples=self.corr_nsample[1], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[2], nsamples=self.corr_nsample[2], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[3], nsamples=self.corr_nsample[3], topk=args.top_k, args=args))
        elif args.corr_type == "sortpoint_sortmask":
            self.corr_modules.append(SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_bais":
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint":
            self.corr_modules.append(TopkPoint(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax":
            self.corr_modules.append(Dual_Softmax(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax_topk":
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_sortmask":
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        else:
            raise Exception("Corr module no imple")
        self.FP_modules = nn.ModuleList()
        if args.up_moudle_type == "feature_fuse":
            self.FP_modules.append(FeatureFuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "feature_mask_fuse":
            self.FP_modules.append(feature_mask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "up_mask":
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
        elif args.up_moudle_type == "feature_onlymask_fuse":
            self.FP_modules.append(feature_onlymask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "source":
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
        elif args.up_moudle_type == "mask_fuse":
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
        else:
            raise Exception("up module no imple")
    
    def forward(self, pointxyz1, pointxyz2, colors1, colors2):
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
            l_xyz1.append(li_xyz1)  # [batch_size, num_points, 3] # [8192点,下采样后的1024点,再下采样后的512,256,64]点 第一个是输入的pointxyz1 8192点
            l_features1.append(li_features1)  # [batch_size, num_points, dims]
            l_xyz2.append(li_xyz2)
            l_features2.append(li_features2)
            l_idx1.append(li_idx1)
            l_idx2.append(li_idx2)

        # calculate the corresponding points of different layers
        l_new_xyz = [None]
        l_mask1 = [None]
        l_mask2 = [None]
        for i in range(len(self.corr_modules)):
            if self.corr_type == "TransformerMacthing" or self.corr_type == "TransformerMacthing_no_position" or self.corr_type == "softmax_topkpoint_topkmask_coordinate":
                temp_newxyz, temp_mask1, temp_mask2 = self.corr_modules[i](
                    l_features1[i + 1].permute(0, 2, 1).contiguous(),
                    l_features2[i + 1].permute(0, 2, 1).contiguous(),
                    l_xyz1[i + 1],
                    l_xyz2[i + 1])
            else:
                temp_newxyz, temp_mask1, temp_mask2 = self.corr_modules[i](
                    l_features1[i + 1].permute(0, 2, 1).contiguous(),
                    l_features2[i + 1].permute(0, 2, 1).contiguous(),
                    l_xyz2[i + 1])
            temp_flow = temp_newxyz - l_xyz1[i + 1]
            l_new_xyz.append(temp_flow)
            l_mask1.append(temp_mask1)
            l_mask2.append(temp_mask2)

        # upsamlping
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            if self.args.up_moudle_type == "feature_fuse" or self.args.up_moudle_type == "feature_mask_fuse" or self.args.up_moudle_type == "feature_onlymask_fuse":
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      unknown_feature1=l_features1[
                                                                                          i - 1],
                                                                                      known_feature1=l_features1[i],
                                                                                      unknown_feature2=l_features2[
                                                                                          i - 1],
                                                                                      known_feature2=l_features1[i]
                                                                                      )
            else:
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      )
        # for i in range(len(l_new_xyz)):
        #     l_new_xyz[i] = l_new_xyz[i] + l_xyz1[i]
        l_new_xyz[0] = l_new_xyz[0] + l_xyz1[0]
        l_new_xyz[1] = l_new_xyz[1] + l_xyz1[1]
        l_new_xyz[2] = l_new_xyz[2] + l_xyz1[2]
        l_new_xyz[3] = l_new_xyz[3] + l_xyz1[3]
        l_new_xyz[4] = l_new_xyz[4] + l_xyz1[4]

        # l_new_xyz 取第一个 8192个点
        return l_xyz1, l_new_xyz, l_idx1, l_idx2, l_mask1, l_mask2


class temp_test(nn.Module):
    r"""
    用训练好的网络测试预测R t 变换矩阵的性能， 与 New_Rt_test.py同时使用
    """

    def __init__(self, args=None, c=6):
        super().__init__()
        self.npoints = args.get('npoints', [1024, 512, 256, 64])
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.corr_nsample = args.get('corr_nsample', [8, 6, 4, 2])
        self.sa_mlps = args.get('sa_mlps',

                                [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
        self.radii = args.get('radii', [0.2, 0.2, 0.4, 0.6])
        self.filter = args.get('filter', False)
        self.SA_modules = nn.ModuleList()
        self.args = args
        self.corr_type = args.corr_type
        self.SA_modules.append(
            PointNet2SAModule(npoint=self.npoints[0], nsample=self.nsamples[0], mlp=self.sa_mlps[0],
                              use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(
            PointNet2SAModule(npoint=self.npoints[1], nsample=self.nsamples[1], mlp=self.sa_mlps[1],
                              use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(
            PointNet2SAModule(npoint=self.npoints[2], nsample=self.nsamples[2], mlp=self.sa_mlps[2],
                              use_xyz=False, use_paconv=True, args=args))
        self.SA_modules.append(
            PointNet2SAModule(npoint=self.npoints[3], nsample=self.nsamples[3], mlp=self.sa_mlps[3],
                              use_xyz=False, use_paconv=True, args=args))
        self.corr_modules = nn.ModuleList()
        if args.corr_type == "topkpoint_topkmask":
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_topkmask":
            self.corr_modules.append(
                Softmax_SortPoint_topMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_topMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_topMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_topMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing":
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                    nsamples=16,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                    nsamples=8,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                    nsamples=4,
                                    args=args))
            self.corr_modules.append(
                TransformerMacthing(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                    nsamples=2,
                                    args=args))
        elif args.corr_type == "softmax_transformer_topkmask":
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                            nsamples=16, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                            nsamples=8, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                            nsamples=4, args=args))
            self.corr_modules.append(
                Softmax_Transforme_SortMask(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                            nsamples=2, args=args))
        elif args.corr_type == "softmax_topkpoint_topkmask":
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "New":
            self.corr_modules.append(New(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(New(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "TransformerMacthing_no_position":
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[0][-1], n_head=4, num_points=self.npoints[0],
                                              nsamples=16,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[1][-1], n_head=4, num_points=self.npoints[1],
                                              nsamples=8,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[2][-1], n_head=4, num_points=self.npoints[2],
                                              nsamples=4,
                                              args=args))
            self.corr_modules.append(
                TransformerMacthingNoPosition(feature_dim=self.sa_mlps[3][-1], n_head=4, num_points=self.npoints[3],
                                              nsamples=2,
                                              args=args))
        elif args.corr_type == "softmaxmask_topkpoint_topkmask":
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmaxmask_TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "softmax_sortpoint_sortmask":
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                Softmax_SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_sort":
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[0], nsamples=self.corr_nsample[0], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[1], nsamples=self.corr_nsample[1], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[2], nsamples=self.corr_nsample[2], topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Sort(self.npoints[3], nsamples=self.corr_nsample[3], topk=args.top_k, args=args))
        elif args.corr_type == "sortpoint_sortmask":
            self.corr_modules.append(SortPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(SortPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_bais":
            self.corr_modules.append(
                TopkPoint_TopkMask_Bais(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Bais(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Bais(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(
                TopkPoint_TopkMask_Bais(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint":
            self.corr_modules.append(TopkPoint(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax":
            self.corr_modules.append(Dual_Softmax(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "dual_sotftmax_topk":
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(Dual_Softmax_Topk(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_sortmask":
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_SortMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        else:
            raise Exception("Corr module no imple")
        self.FP_modules = nn.ModuleList()
        if args.up_moudle_type == "feature_fuse":
            self.FP_modules.append(FeatureFuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "feature_mask_fuse":
            self.FP_modules.append(feature_mask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_mask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "up_mask":
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
            self.FP_modules.append(Upmask())
        elif args.up_moudle_type == "feature_onlymask_fuse":
            self.FP_modules.append(feature_onlymask_fuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(feature_onlymask_fuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "source":
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
        elif args.up_moudle_type == "mask_fuse":
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
            self.FP_modules.append(MaskFuse())
        else:
            raise Exception("up module no imple")

    def forward(self, pointxyz1, pointxyz2, colors1, colors2, trans_gt):
        """
        :param pointxyz1: source point cloud xyz [bs, num_points, 3]
        :param pointxyz2: target point cloud xyz
        :param colors1: source point cloud's color [bs, num_points, dims]
        :param colors2: target point cloud's color. The above two are used as feature
        """
        l_xyz1, l_features1 = [pointxyz1], [
            torch.cat([pointxyz1.permute(0, 2, 1), colors1.permute(0, 2, 1)], dim=1)]
        l_xyz2, l_features2 = [pointxyz2], [
            torch.cat([pointxyz2.permute(0, 2, 1), colors2.permute(0, 2, 1)], dim=1)]
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
        src_feat = l_features1[2].transpose(1, 2).contiguous()
        tgt_feat = l_features1[2].transpose(1, 2).contiguous()
        sim_matrix_1 = F.cosine_similarity(src_feat.unsqueeze(2), tgt_feat.unsqueeze(1), dim=-1)
        conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_1, 2)
        bsize, N, M = conf_matrix.shape

        self.sample_n_points = 128
        conf, idx = conf_matrix.view(bsize, -1).sort(descending=True, dim=1)
        xyz1 = l_xyz1[2]
        xyz2 = l_xyz2[2]
        weight = conf[:, :self.sample_n_points]
        idx = idx[:, :self.sample_n_points]
        idx_src = idx // M  # torch.div(idx, M, rounding_mode='trunc')
        idx_tgt = idx % M
        b_index = torch.arange(bsize).view(-1, 1).repeat((1, self.sample_n_points)).view(-1)
        src_pcd_sampled = xyz1[b_index, idx_src.view(-1)].view(bsize, self.sample_n_points, -1)
        tgt_pcd_sampled = xyz2[b_index, idx_tgt.view(-1)].view(bsize, self.sample_n_points, -1)
        pred_R, pred_t, condition = batch_weighted_procrustes(src_pcd_sampled, tgt_pcd_sampled, weight.unsqueeze(-1))
        Identity = []
        for i in range(pred_R.shape[0]):
            Identity.append(torch.eye(3, 3).cuda())
        Identity = torch.stack(Identity, dim=0)
        resi_R = torch.norm((torch.matmul(pred_R.transpose(2, 1).contiguous(), trans_gt[0]) - Identity), dim=(1, 2),
                            keepdim=False).mean()
        return l_xyz1


def mean_filtering(pred_point, source_point, nsample=8):
    nsample += 1
    idx = knnquery_heap(nsample, source_point, source_point)[:, :, 1:].contiguous()  # (b, m, nsample)
    pred_point_trans = pred_point.transpose(1, 2).contiguous()
    grouped_pred_xyz = grouping(pred_point_trans, idx)  # (b, 3, m, nsample)
    l2_distance = torch.norm(grouped_pred_xyz - pred_point_trans.unsqueeze(dim=-1), dim=1)
    norm = torch.sum(1 / (l2_distance + 1e-6), dim=2, keepdim=True)
    weight = 1 / (l2_distance + 1e-6) / norm
    group_pred_mean = (pred_point.unsqueeze(dim=-1) * weight.unsqueeze(dim=-2)).sum(dim=-1)
    return group_pred_mean


class PTEnetBase(nn.Module):
    r"""
    """

    def __init__(self, c=6, args=None):
        super().__init__()
        self.npoints = args.get('npoints', [1024, 512, 256, 64])
        self.nsamples = args.get('nsamples', [33, 25, 13, 13])
        self.sa_mlps = args.get('sa_mlps',
                                [[c, 16, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
        self.radii = args.get('radii', [0.2, 0.2, 0.4, 0.6])
        self.filter = args.get('filter', False)
        self.up_moudle_type = args.up_moudle_type
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
        if args.corr_type == "topkpoint_topkmask":
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint_topkmask_bais":
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint_TopkMask_Bais(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        elif args.corr_type == "topkpoint":
            self.corr_modules.append(TopkPoint(self.npoints[0], nsamples=8, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[1], nsamples=6, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[2], nsamples=4, topk=args.top_k, args=args))
            self.corr_modules.append(TopkPoint(self.npoints[3], nsamples=2, topk=args.top_k, args=args))
        else:
            raise Exception("Corr module no imple")

        self.FP_modules = nn.ModuleList()
        if args.up_moudle_type == "feature_fuse":
            self.FP_modules = nn.ModuleList()
            self.FP_modules.append(FeatureFuse(mlps=[self.sa_mlps[0][-1], self.sa_mlps[0][-1]]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[1]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[2]))
            self.FP_modules.append(FeatureFuse(mlps=self.sa_mlps[3]))
        elif args.up_moudle_type == "source":
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
            self.FP_modules.append(PTENet2FPMaskMtutalModule())
        elif args.up_moudle_type == "relu":
            self.FP_modules.append(BaseUpRelu())
            self.FP_modules.append(BaseUpRelu())
            self.FP_modules.append(BaseUpRelu())
            self.FP_modules.append(BaseUpRelu())
        else:
            raise Exception("up module no imple")

    def forward(self, pointxyz1, pointxyz2, colors1, colors2):
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

        # calculate the corresponding points of different layers
        l_new_xyz = [None]
        l_mask1 = [None]
        l_mask2 = [None]
        for i in range(len(self.corr_modules)):
            temp_newxyz, temp_mask1, temp_mask2 = self.corr_modules[i](l_features1[i + 1].permute(0, 2, 1).contiguous(),
                                                                       l_features2[i + 1].permute(0, 2, 1).contiguous(),
                                                                       l_xyz2[i + 1])
            l_new_xyz.append(temp_newxyz)
            l_mask1.append(temp_mask1)
            l_mask2.append(temp_mask2)

        # upsamlping
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            if self.up_moudle_type == "feature_fuse":
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      unknown_feature1=l_features1[
                                                                                          i - 1],
                                                                                      known_feature1=l_features1[i],
                                                                                      unknown_feature2=l_features2[
                                                                                          i - 1],
                                                                                      known_feature2=l_features1[i]
                                                                                      )
            else:
                l_new_xyz[i - 1], l_mask1[i - 1], l_mask2[i - 1] = self.FP_modules[i](unknown_xyz1=l_xyz1[i - 1],
                                                                                      known_xyz1=l_xyz1[i],
                                                                                      unknown_xyz2=l_xyz2[i - 1],
                                                                                      known_xyz2=l_xyz2[i],
                                                                                      unknown_new_xyz=l_new_xyz[i - 1],
                                                                                      known_new_xyz=l_new_xyz[i],
                                                                                      unknown_mask1=l_mask1[i - 1],
                                                                                      known_mask1=l_mask1[i],
                                                                                      unknown_mask2=l_mask2[i - 1],
                                                                                      known_mask2=l_mask2[i],
                                                                                      )
        if self.filter:
            l_new_xyz[0] = mean_filtering(pred_point=l_new_xyz[0], source_point=l_xyz1[0])

        return l_xyz1, l_new_xyz, l_idx1, l_idx2, l_mask1, l_mask2


if __name__ == "__main__":
    from util.util import CfgNode

    args = CfgNode()
    args.cuda = True
    args.top_k = True
    xyz1 = torch.randn(5, 4096, 3, requires_grad=True).cuda()
    xyz1_feats = torch.randn(5, 4096, 3, requires_grad=True).cuda()
    xyz2 = torch.randn(5, 4096, 3, requires_grad=True).cuda()
    xyz2_feats = torch.randn(5, 4096, 3, requires_grad=True).cuda()
    mean_filtering(pred_point=xyz1, source_point=xyz2)

    # test_model = PTEnetBase(args=args).cuda()
    # result = test_model(xyz1, xyz2, xyz1_feats, xyz2_feats)
