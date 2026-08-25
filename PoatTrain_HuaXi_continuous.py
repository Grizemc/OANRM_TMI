#  无监督过拟合两帧图像
import os

from sklearn.neighbors import NearestNeighbors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch import nn
from lib.pointops.functions import pointops
from model.backbone_new import PTEnetBase, PTFlow
import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from util.data import HuaXiFpfh, HuaXiFpfhContinuous
from util.util import load_cfg_from_cfg_file, IOStream
import lib.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
import open3d as o3d
import sys
from BiaoZhu_single_test_value import Gaussian_filter_single_inter


class Human_SelfSupervise_mask:
    def __init__(self, mask_truncation, gt_factor, smooth_factor, damp_ratio, damp_step, mask_factor=10.0,
                 plane_true=False):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.mask_factor = mask_factor
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()
        self.plan_true = plane_true
        self.damp_ratio = damp_ratio
        self.damp_step = damp_step
        self.mask_truncation = mask_truncation
        self.mask_loss1 = nn.BCEWithLogitsLoss()
        self.mask_loss2 = nn.BCEWithLogitsLoss()

    def flow_smooth_loss(self, pred, xyz1, smooth_weight):
        B = pred.shape[0]
        flow = pred - xyz1
        # idx1 = pointops.knnquery(6, xyz1, xyz1)
        idx1 = pointops.ballquery(3, 6, xyz1, xyz1)
        idx1 = idx1[:, :, 1:].contiguous()
        neigh_flow = pointops.grouping(flow.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3,
                                                                                              1).contiguous()
        neigh_xyz1 = pointops.grouping(xyz1.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3,
                                                                                              1).contiguous()
        relative_xyz = neigh_xyz1 - xyz1.unsqueeze(-2)
        source_distance = torch.norm(relative_xyz, dim=-1)
        # Reciprocal
        dist_recip = 1.0 / (source_distance + 1e-10)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        relative_flow = torch.norm(neigh_flow - flow.unsqueeze(-2), dim=-1) * weight
        smooth_loss = relative_flow.mean() * smooth_weight
        return smooth_loss, smooth_loss

    def eye_like(self, tensor):
        """Create an identity matrix of the same size as a given tensor.

        Args:
            tensor (torch.Tensor): Tensor to match the size of.

        Returns:
            torch.Tensor: Identity matrix.
        """
        assert tensor.shape[-1] == tensor.shape[-2]
        if tensor.ndim == 4:
            b = tensor.shape[0]
            n = tensor.shape[1]
            eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
            expanded_eye = eyed_tensor.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1)
        elif tensor.ndim == 3:
            n = tensor.shape[0]
            eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
            expanded_eye = eyed_tensor.unsqueeze(0).expand(n, -1, -1)
        elif tensor.ndim == 5:
            b = tensor.shape[0]
            n = tensor.shape[1]
            m = tensor.shape[2]
            eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
            expanded_eye = eyed_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n, m, -1, -1)

        return expanded_eye.to(tensor.dtype)

    def gradient_and_det_loss(self, pred, xyz1, smooth_weight):
        B = pred.shape[0]
        flow = pred - xyz1
        # idx1 = pointops.ballquery(0.05, 6, xyz1, xyz1)
        idx1 = pointops.knnquery(6, xyz1, xyz1)
        idx1 = idx1[:, :, 1:].contiguous()
        neigh_flow = pointops.grouping(flow.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3,
                                                                                              1).contiguous()
        neigh_xyz1 = pointops.grouping(xyz1.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3,
                                                                                              1).contiguous()
        relative_xyz = neigh_xyz1 - xyz1.unsqueeze(-2)
        relative_flow = neigh_flow - flow.unsqueeze(-2)
        graident_list = []
        error_sum = 0.
        for i in range(3):
            flow_neigh_single = relative_flow[:, :, :, i].reshape(B, relative_flow.shape[1], relative_flow.shape[2],
                                                                  1)
            xyz_relative_trans = relative_xyz.permute(0, 1, 3, 2).contiguous()
            coefficients_front = torch.linalg.pinv(torch.matmul(xyz_relative_trans, relative_xyz))
            coefficients = torch.matmul(torch.matmul(coefficients_front, xyz_relative_trans), flow_neigh_single)
            flow_neigh_single_restore = torch.matmul(relative_xyz, coefficients)
            single_error = (flow_neigh_single_restore - flow_neigh_single).mean()
            graident_list.append(coefficients)
            error_sum += single_error
        jacob_mat = torch.cat((graident_list[0], graident_list[1], graident_list[2]), dim=3).transpose(2,
                                                                                                       3).contiguous()

        smooth_gradient_loss = torch.abs(jacob_mat.norm(dim=-1)).mean() * smooth_weight
        #  neighbour jacob loos
        # jacob_mat_reshape =jacob_mat.reshape(B, jacob_mat.shape[1], 9)
        # neighbour_jacob = pointops.grouping(jacob_mat_reshape.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3,
        #                                                                                                 1).contiguous()
        # smooth_gradient_loss = torch.norm(neighbour_jacob - jacob_mat_reshape.unsqueeze(-2), dim=-1).mean() * smooth_weight

        # AS RIGID AS POSSIBLE LOSS
        # anlyt_grad_inner = torch.matmul(jacob_mat.transpose(2, 3), jacob_mat)
        # smooth_gradient_loss = torch.linalg.norm(torch.abs(anlyt_grad_inner - self.eye_like(anlyt_grad_inner) + 1e-7),
        #                   ord="fro", dim=(-1, -2)).mean() * smooth_weight
        # det_loss
        flow_gradient_det = torch.linalg.det(jacob_mat)
        det_loss = torch.clamp(torch.abs(1 - torch.linalg.det(jacob_mat)), -10., 10.)
        det_loss = (flow_gradient_det - 1).abs().mean() * smooth_weight * 0.5
        return smooth_gradient_loss, det_loss

    def chamfer_distance(self, xyz1, xyz2, normal2, plane_true=False):
        if plane_true:
            idx1 = pointops.knnquery(2, xyz2, xyz1)[:, :, 1:].contiguous()
            idx2 = pointops.knnquery(2, xyz1, xyz2)[:, :, 1:].contiguous()
            normal1_group = pointops.grouping(normal2.transpose(1, 2).contiguous(), idx1.int())
            normal1_group = normal1_group.transpose(1, 2).squeeze().contiguous()
            xyz1_group = pointops.grouping(xyz2.transpose(1, 2).contiguous(), idx1.int())
            xyz1_group = xyz1_group.transpose(1, 2).squeeze().contiguous()
            single_pred = xyz1[0].detach().cpu().numpy()
            single_pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(single_pred))
            single_pred_pcd.estimate_normals()
            pred_normal = torch.from_numpy(np.array(single_pred_pcd.normals)).unsqueeze(0).float().to(xyz1.device)
            normal2_group = pointops.grouping(pred_normal.transpose(1, 2).contiguous(), idx2.int())
            normal2_group = normal2_group.transpose(1, 2).squeeze().contiguous()
            xyz2_group = pointops.grouping(xyz1.transpose(1, 2).contiguous(), idx2.int())
            xyz2_group = xyz2_group.transpose(1, 2).contiguous().squeeze()
            chamfer_distance1 = (((xyz1_group - xyz1.squeeze()) * normal1_group) ** 2).sum(dim=-1)
            chamfer_distance2 = (((xyz2_group - xyz2.squeeze()) * normal2_group) ** 2).sum(dim=-1)
            nn_loss_sum = (chamfer_distance1.sqrt()).mean() + (chamfer_distance2.sqrt()).mean()
        else:
            # dist = torch.cdist(xyz1, xyz2, p=2.)
            # nn_loss_sum = dist.min(-1)[0].mean() + dist.min(-2)[0].mean()
            dist1, dist2, idx1, idx2 = self.chamLoss(xyz1, xyz2)
            dist1 = torch.relu(dist1 - 0.00001)
            dist2 = torch.relu(dist2 - 0.00001)
            nn_loss_sum = (dist1.sqrt()).mean() + (dist2.sqrt()).mean()
        return nn_loss_sum

    def rgb_target(self, xyz):
        xyz += 1
        # idx1 = pointops.knnquery( 5, single_xyz2.unsqueeze(0), single_pred.unsqueeze(0))[:, :, 1:].contiguous()
        #     rgb2_mean = pointops.grouping(single_color2.unsqueeze(0).transpose(1, 2).contiguous(), idx1.int()
        #                                   ).transpose(1, 2).contiguous().mean(dim=3).squeeze()
        #     idx1 = pointops.knnquery(5, single_pred.unsqueeze(0), single_xyz2.unsqueeze(0))[:, :, 1:].contiguous()
        #     rgb1_mean = pointops.grouping(single_color1.unsqueeze(0).transpose(1, 2).contiguous(), idx1.int()
        #                                   ).transpose(1, 2).contiguous().mean(dim=3).squeeze()
        #     rgb_error1 = torch.norm(rgb2_mean - single_color1, dim=1)
        #     rgb_error2 = torch.norm(rgb1_mean - single_color2, dim=1)
        #     target1 = torch.where(rgb_error1 < 0.3, 1., 0.)
        #     target2 = torch.where(rgb_error2 < 0.3, 1., 0.)
        return xyz

    def post_losscal_mask_huamn_contiuous(self, l_xyz1, l_new_xyz, xyz2, color1, color2, l_mask1, l_mask2, mask_pseudo1,
                                          mask_pseudo2, normal2, gt_pseudo, loss_epoch):
        mask1 = l_mask1[0]
        mask2 = l_mask2[0]
        pred = l_new_xyz[0]
        xyz1 = l_xyz1[0]
        # smooth loss
        # smooth_distance_loss, smooth_angle_loss = AngleAndDistanceLoss(pc1=single_xyz1.unsqueeze(dim=0),
        #                                                                smooth_num=20, distance_num=10,
        #                                                                pred=single_pred.unsqueeze(dim=0),
        #                                                                distance_factor=2000,
        #                                                                angle_facor=0.5)
        if loss_epoch % self.damp_step == 0 and loss_epoch != 0:
            self.smooth_a = self.smooth_a * self.damp_ratio

        nn_loss_sum = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true) * self.gt_a
        nn_loss_sum_eval = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true)
        if mask_pseudo1.sum == 0:
            print("No mask loss")
        else:
            mask_gt_1 = torch.ones_like(mask_pseudo1).float()
            pred_mask_1 = mask1[:, mask_pseudo1[0, :].long(), 0]
            mask_loss_sum = self.mask_loss1(pred_mask_1, mask_gt_1)
            pred_pseudo = pred[:, mask_pseudo1[0, :].long(), 0:]
            gt_pseudo_loss = torch.norm((pred_pseudo - gt_pseudo), dim=2).mean()
            mask_gt_2 = torch.ones_like(mask_pseudo2).float()
            pred_mask_2 = mask2[:, mask_pseudo2[0, :].long(), 0]
            mask_loss_sum += self.mask_loss2(pred_mask_2, mask_gt_2)
        # overlap area
        single_mask1 = torch.relu(torch.sigmoid(mask1) - self.mask_truncation)
        single_mask2 = torch.relu(torch.sigmoid(mask2) - self.mask_truncation)
        single_mask1 = single_mask1.bool()
        single_mask2 = single_mask2.bool()
        normal2 = normal2[single_mask2.repeat(1, 1, 3)].reshape(1, -1, 3)
        # smooth_gradient_loss, det_loss = self.gradient_and_det_loss(pred, xyz1, self.smooth_a)
        smooth_gradient_loss, det_loss = self.flow_smooth_loss(pred, xyz1, self.smooth_a)
        if single_mask1.sum() / single_mask1.shape[0] < 0.2 or single_mask2.sum() / single_mask2.shape[0] < 0.2:
            # loss = nn_loss_sum * 2 + smooth_gradient_loss
            loss = nn_loss_sum + smooth_gradient_loss + gt_pseudo_loss
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss
        else:
            # NN true loss;
            overlap_pred = pred[single_mask1.repeat(1, 1, 3)].reshape(1, -1, 3)
            overlap_xyz2 = xyz2[single_mask2.repeat(1, 1, 3)].reshape(1, -1, 3)
            smooth_true_gradient_loss, det_loss = self.flow_smooth_loss(overlap_pred,
                                                                   xyz1[single_mask1.repeat(1, 1, 3)].reshape(1, -1, 3),
                                                                   self.smooth_a)
            nn_true_loss_sum = self.chamfer_distance(overlap_pred, overlap_xyz2, normal2,
                                                     plane_true=self.plan_true) * self.gt_a
            loss = nn_true_loss_sum +  smooth_gradient_loss + gt_pseudo_loss
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_true_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss


def PostTrainInit(type_name):
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='config/Source_Flow_softmax_topkpoint_topmask_fuse_8192.yaml',
                        help='config file')
    args_l = parser.parse_args()
    assert args_l.config is not None
    args = load_cfg_from_cfg_file(args_l.config)
    # -----------------------------------------------------------------------------
    # post_train
    # -----------------------------------------------------------------------------
    if not os.path.exists("checkpoints/" + args.exp_name + '/saved_model'):
        raise SystemExit('No model')
    if not os.path.exists("checkpoints/" + '/fpfh_' + type_name):
        os.makedirs("checkpoints/" + '/fpfh_' + type_name)
        os.makedirs("checkpoints/" + '/fpfh_' + type_name + "/post_train")
        os.makedirs("checkpoints/" + '/fpfh_' + type_name + "/merge_result")
    io = IOStream("checkpoints/" + '/fpfh_' + type_name + '/post_train.log')
    file_name = os.path.basename(sys.argv[0])
    os.system('cp -r {} '.format(file_name) +
              "checkpoints/" + '/fpfh_' + type_name + '/{}.txt'.format(file_name))
    # -----------------------------------------------------------------------------
    # set random seed
    # -----------------------------------------------------------------------------
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.type_name = type_name
    return args, io


def init_model(args):
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.model_type == "Base":
        model = PTEnetBase(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow":
        model = PTFlow(c=6, args=args).to(args.device)
    else:
        raise SystemExit('Not impletion')
    return model


def post_train_main(model, args, io, index, data, loss_fn):
    # ============= Model ===================
    io.cprint("dataset index is ".format(index))
    # ============= load ================
    model_path = os.path.join('checkpoints/', args.exp_name, "saved_model/best_model.t7")
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    # ============= Optimizer ================
    if args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.post_train_lr, momentum=args["MOMENTUM"])
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.post_train_lr, weight_decay=1e-4)
    else:
        optimizer = None
        exit(0)
    # ============= scheduler ================
    scheduler = StepLR(optimizer, step_size=5, gamma=0.2)
    model.train()
    points1, points2, colors1, colors2, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo = data
    points1 = points1.to(args.device)
    points2 = points2.to(args.device)
    colors1 = colors1.to(args.device)
    colors2 = colors2.to(args.device)
    mask_pseudo1 = mask_pseudo1.to(args.device)
    mask_pseudo2 = mask_pseudo2.to(args.device)
    gt_pseudo = gt_pseudo.to(args.device)
    normal2 = normal2.to(args.device)
    best_loss = 1e10
    best_result = []
    for epoch in range(0, args.pt_epoch):
        optimizer.zero_grad()
        l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(points1, points2, colors1,
                                                                               colors2)
        nn_loss_sum_eval, loss, nnloss, nn_true_loss, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss = loss_fn.post_losscal_mask_huamn_contiuous(
            l_xyz1=l_xyz1,
            l_new_xyz=l_pred_xyz,
            xyz2=points2,
            color1=colors1,
            color2=colors2,
            l_mask1=l_pred_mask1,
            l_mask2=l_pred_mask2,
            mask_pseudo1=mask_pseudo1,
            mask_pseudo2=mask_pseudo2,
            normal2=normal2,
            gt_pseudo=gt_pseudo,
            loss_epoch=epoch)
        io.cprint("index is {}, nn_loss_sum_eval is {}, loss is {}, nnloss is {}, nn_true_loss is {}"
                  ", smooth_gradient_loss is {}, mask_loss_sum is {} , gt_pseudo_loss is {} ".format(index,
                                                                                                     nn_loss_sum_eval,
                                                                                                     loss, nnloss,
                                                                                                     nn_true_loss,
                                                                                                     smooth_gradient_loss,
                                                                                                     mask_loss_sum,
                                                                                                     gt_pseudo_loss
                                                                                                     ))

        pred_mask1 = torch.sigmoid(l_pred_mask1[0]) > args.mask_truncation
        pred_mask2 = torch.sigmoid(l_pred_mask2[0]) > args.mask_truncation
        mask1_sum = pred_mask1.sum() / pred_mask1[0].shape[1] / pred_mask1[0].shape[0]
        mask2_sum = pred_mask2.sum() / pred_mask2[0].shape[1] / pred_mask2[0].shape[0]
        string = "index is {}, mask1_sum is {}, mask2_sum is {}".format(index, mask1_sum, mask2_sum)
        io.cprint(string)
        loss.backward()
        optimizer.step()
        sum_nn_loss = nnloss.item()
        if best_loss >= sum_nn_loss or epoch == 0:
            best_loss = sum_nn_loss
        scheduler.step()
    save_path = 'checkpoints/fpfh_{}/{}'.format(str(args.type_name), 'post_train/index_') + "{}.npz".format(index)
    np.savez(save_path,
             points2=points2.cpu(),
             points1=points1.cpu(),
             colors1=colors1.cpu(),
             colors2=colors2.cpu(),
             pred_xyz=l_pred_xyz[0].detach().cpu(),
             pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
             pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())


def post_process(index, args, mask_trun):
    pred_path = 'checkpoints/fpfh_{}/{}'.format(str(args.type_name), 'post_train/index_') + "{}.npz".format(index)
    with np.load(pred_path) as npz:
        point1 = npz['points1'].squeeze()
        point2 = npz['points2'].squeeze()
        pred_xyz = npz['pred_xyz'].squeeze()
        pred_mask1 = npz['pred_mask1'].squeeze()
        pred_mask2 = npz['pred_mask2'].squeeze()
        color1 = npz['colors1'].squeeze()
        color2 = npz['colors2'].squeeze()
    # 使用ransac算法进行整体位移变换
    ransac_pred_mask1 = pred_mask1 > 0.9
    ransac_source_cloud = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
    ransac_pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
    ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
    ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
    distance_threshold = 2
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
    )
    ratation_point1 = np.dot(ransac_result.transformation,
                             np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]
    # 使用Flow剔除
    pred_mask1 = pred_mask1 > 0.99
    flow = pred_xyz - ratation_point1
    flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
    flow_true_mean = flow_true.mean(axis=0)
    dot_products = np.dot(flow_true, flow_true_mean)  # 计算向量的点积
    cosine_angles = dot_products / (np.linalg.norm(flow_true, axis=1) * np.linalg.norm(flow_true_mean))  # 计算余弦夹角
    angles_radians = np.degrees(np.arccos(cosine_angles))  # 计算角度（弧度）
    new_true = angles_radians < 90
    # new_true = np.ones_like(angles_radians).astype(np.bool_)
    true_true_point = ratation_point1[pred_mask1][new_true]
    true_true_color = color1[pred_mask1][new_true]
    true_true_flow = flow[pred_mask1][new_true]
    true_true_pred_xyz = pred_xyz[pred_mask1][new_true]
    false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]), axis=0)
    false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
    distances, nearest_indices = nbrs.kneighbors(false_point)
    neigh_flow = true_true_flow[nearest_indices]
    result_flow = neigh_flow
    weight = 1 / distances
    result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
    result_point = false_point + result_flow.sum(axis=1)
    new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1][new_true]), axis=0)
    new_color = np.concatenate((false_color, true_true_color), axis=0)
    source_fusion_point = np.concatenate((point2, new_pred_point), axis=0)
    source_fusion_color = np.concatenate((color2, new_color), axis=0)
    source_fusion_point_sample = source_fusion_point.copy()
    source_fusion_point_sample[:, 2] = 0
    sample_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((source_fusion_point_sample)))
    sample_pcd.colors = o3d.pybind.utility.Vector3dVector(source_fusion_color)
    sample_pcd, _, pd_list = sample_pcd.voxel_down_sample_and_trace(voxel_size=0.5, min_bound=sample_pcd.get_min_bound(),
                                                                    max_bound=sample_pcd.get_max_bound())
    result_list = []
    for temp_list in pd_list:
        temp_list = np.array(temp_list)
        if temp_list.size > 1:
            result_list.append(temp_list[-1])
        result_list.append(temp_list[0])
    result_index = np.array(result_list)
    final_downsample = source_fusion_point[result_index]
    final_downsample_color = source_fusion_color[result_index]
    savez_path = 'checkpoints/fpfh_{}/{}'.format(str(args.type_name), 'merge_result/index_') + "{}.npz".format(index)
    np.savez(savez_path, point1=final_downsample, color1=final_downsample_color)


if __name__ == "__main__":
    mask_truncation = 0.9
    g_type_name = "Huaxi_continuous_10thousand"
    args_g, io_g = PostTrainInit(g_type_name)

    args_g.mask_truncation = mask_truncation
    args_g.pt_epoch = 10  # 5 0.0001 0.1 20
    merge_result = 'checkpoints/fpfh_{}/{}'.format(str(args_g.type_name), 'merge_result')
    train_loader = DataLoader(
        HuaXiFpfhContinuous(root=r"/big_data/szm/szm_MICCAI_Hamlyn/continuous_HuaXi_10thousand/", temp_result=merge_result),
        batch_size=1)
    args_g.train_loader = train_loader
    # post train code
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    args_g.result = []
    model_g = init_model(args_g)
    for index, data in loop:
        args_g.post_train_lr = 0.001
        loss_fn = Human_SelfSupervise_mask(mask_truncation, smooth_factor=5, gt_factor=1, damp_ratio=0.5,
                                           damp_step=2)
        start_time = datetime.datetime.now()
        post_train_main(model_g, args_g, io_g, index, data, loss_fn)
        end_time = datetime.datetime.now()
        io_g.cprint("all time consumption is {}".format((end_time - start_time).seconds))
        post_process(index, args_g, mask_truncation)
