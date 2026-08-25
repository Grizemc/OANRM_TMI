#  无监督过拟合两帧图像
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from scipy.spatial import KDTree
from torch import nn
from lib.pointops.functions import pointops
from model.backbone_new import PTEnetBase, PTFlow, PTFlowmean
import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.data import HumanMarkDataSingle, HumanMarkDataSingleFilterFpfh, HumanMarkDataSingleFilterFpfhSL
from util.util import load_cfg_from_cfg_file, IOStream
import lib.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
import open3d as o3d
import sys


def Gaussian_filter_single_inter(xyz1, source_pcd2, pcd2, pred_pc, single_num, pred_mask, knn_num=5, sigma=0.2):
    # Gaussian_filter_gradient_inter
    single_xyz1 = xyz1[0, ::]
    single_pred_pc = pred_pc[0, ::]
    single_flow = single_pred_pc - single_xyz1
    # 构建KD树
    kdtree = KDTree(single_xyz1)
    # 使用KD树进行最近邻搜索，返回距离和索引
    distances, indices = kdtree.query(single_xyz1, k=knn_num)
    # 获取距离最近的7个点的坐标
    neigh_xyz1 = single_xyz1[indices]
    neigh_flow = single_flow[indices]
    relative_xyz = neigh_xyz1 - single_xyz1[:, np.newaxis, :]
    # gaussian
    gaussian_weight_up = np.exp(-(np.square(relative_xyz).sum(axis=-1)) / (sigma ** 2 * 2))
    gaussian_weight_down = np.power(2 * np.pi, 1.5) * np.power(sigma, 3)
    gaussian_weight = gaussian_weight_up / gaussian_weight_down
    gaussian_weight = gaussian_weight / gaussian_weight.sum(axis=-1, keepdims=True)
    single_new_flow = (neigh_flow * np.expand_dims(gaussian_weight, axis=-1)).sum(axis=-2)
    single_new_pred = single_new_flow + single_xyz1
    new_pred_pcd = single_new_pred[-single_num[0]:, :]

    gaussian_displace = np.sqrt(np.sum((new_pred_pcd - pcd2) ** 2, axis=2))
    gaussian_displace_gt = gaussian_displace.mean()

    if pred_mask.sum() == 0:
        gaussian_displace_pred = 0.
    else:
        gaussian_displace_pred = gaussian_displace[pred_mask.squeeze(2)].mean()
    return gaussian_displace_gt, gaussian_displace_pred


class Human_SelfSupervise_mask:
    def __init__(self, mask_truncation, gt_factor, charm_factor,smooth_factor, damp_ratio, damp_step, mask_factor=10.0,
                 plane_true=False):
        self.gt_a = gt_factor
        self.charm_factor = charm_factor
        self.smooth_a = smooth_factor
        self.mask_factor = mask_factor
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()
        self.plan_true = plane_true
        self.damp_ratio = damp_ratio
        self.damp_step = damp_step
        self.mask_truncation = mask_truncation
        self.mask_loss1 = nn.BCEWithLogitsLoss()
        self.mask_loss2 = nn.BCEWithLogitsLoss()

    def gradient_and_det_loss(self, pred, xyz1, smooth_weight):
        B = pred.shape[0]
        flow = pred - xyz1
        # idx1 = pointops.ballquery(0.05, 6, xyz1, xyz1)
        idx1 = pointops.knnquery(4, xyz1, xyz1)
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

        smooth_gradient_loss = torch.abs(jacob_mat.norm(dim=-1) - 0.5).mean() * smooth_weight
        flow_gradient_det = torch.linalg.det(jacob_mat)
        det_loss = torch.clamp(torch.abs(1 - torch.linalg.det(jacob_mat)), -10., 10.)
        det_loss = (flow_gradient_det - 1).abs().mean() * smooth_weight * 0.5
        return smooth_gradient_loss, det_loss

    def chamfer_distance(self, xyz1, xyz2, normal2, plane_true=False):
        dist1, dist2, idx1, idx2 = self.chamLoss(xyz1, xyz2)
        nn_loss_sum = (dist1.sqrt()).mean() + (dist2.sqrt()).mean()
        return nn_loss_sum

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

    def post_losscal_mask_huamn(self, l_xyz1, l_new_xyz, xyz2, color1, color2, l_mask1, l_mask2, mask_pseudo1,
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
        smooth_gradient_loss, det_loss = self.gradient_and_det_loss(pred, xyz1, self.smooth_a)
        # smooth_gradient_loss, det_loss = self.flow_smooth_loss(pred, xyz1, self.smooth_a)
        nn_loss_sum_eval = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true)
        nn_loss_sum = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true) * self.charm_factor
        if mask_pseudo1.sum == 0:
            print("No mask loss")
            mask_loss_sum = 0.
            gt_pseudo_loss = 0.

        else:
            mask_gt_1 = torch.ones_like(mask_pseudo1).float()
            pred_mask_1 = mask1[:, mask_pseudo1[0, :].long(), 0]
            mask_loss_sum = self.mask_loss1(pred_mask_1, mask_gt_1)
            pred_pseudo = pred[:, mask_pseudo1[0, :].long(), 0:]
            gt_pseudo_loss = torch.norm((pred_pseudo - gt_pseudo), dim=2).mean() * self.gt_a
            mask_gt_2 = torch.ones_like(mask_pseudo2).float()
            pred_mask_2 = mask2[:, mask_pseudo2[0, :].long(), 0]
            mask_loss_sum += self.mask_loss2(pred_mask_2, mask_gt_2)
        # overlap area
        single_mask1 = torch.relu(torch.sigmoid(mask1[0]) - self.mask_truncation)
        single_mask2 = torch.relu(torch.sigmoid(mask2[0]) - self.mask_truncation)
        single_mask1 = single_mask1.bool()
        single_mask2 = single_mask2.bool()
        overlap_pred = pred[single_mask1.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)
        overlap_xyz2 = xyz2[single_mask2.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)
        normal2 = normal2[single_mask2.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)

        if single_mask1.sum() / single_mask1.shape[0] < 0.5 or single_mask2.sum() / single_mask2.shape[0] < 0.5:
            # loss = nn_loss_sum + smooth_gradient_loss + gt_pseudo_loss
            loss = nn_loss_sum * 2 + smooth_gradient_loss + gt_pseudo_loss 
            # loss = nn_loss_sum
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss
        else:
            # NN true loss;
            nn_true_loss_sum = self.chamfer_distance(overlap_pred, overlap_xyz2, normal2,
                                                     plane_true=self.plan_true) * self.charm_factor
            # loss = nn_true_loss_sum+ nn_loss_sum+
            # loss = nn_true_loss_sum + nn_loss_sum + smooth_gradient_loss + gt_pseudo_loss
            loss = nn_true_loss_sum + nn_loss_sum + smooth_gradient_loss + gt_pseudo_loss 
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_true_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss


def SH_PostTrainInit(type_name):
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml',
                        help='config file')
    args_l = parser.parse_args()
    assert args_l.config is not None
    args = load_cfg_from_cfg_file(args_l.config)
    # -----------------------------------------------------------------------------
    # post_train
    # -----------------------------------------------------------------------------
    type_name = 'fpfh_' + type_name
    if not os.path.exists('checkpoints/' + args.exp_name + '/saved_model'):
        raise SystemExit('No model')
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + type_name):
        os.makedirs('checkpoints/' + args.exp_name + '/' + type_name)
        os.makedirs('checkpoints/' + args.exp_name + '/' + type_name + '/npz_result')
        os.makedirs('checkpoints/' + args.exp_name + '/' + type_name + '/eval_result')
    io = IOStream('checkpoints/' + args.exp_name + '/' + type_name + '/post_train.log')
    file_name = os.path.basename(sys.argv[0])
    os.system(
        'cp -r {} checkpoints/'.format(file_name) + args.exp_name + '/' + type_name + '/{}.txt'.format(file_name))
    os.system(
        'cp -r config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_80.yaml checkpoints/'.format(file_name) + args.exp_name + '/' + type_name + '/Source_Flow_softmax_topkpoint_topmask_fuse_8192_80.yaml')
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


def BiaoZhuTantTaiMaskGaussianPostTrainSource(source_pred_pcd, source_pcd2, source_pcd1, source_pred_mask1,
                                              source_pred_mask2, single_num):
    pred_mask1 = source_pred_mask1[:, -single_num:, :]
    pred_mask2 = source_pred_mask2[:, -single_num:, :]
    pred_pcd = source_pred_pcd.clone()
    pcd2 = source_pcd2.clone()
    # source
    pred_pcd = pred_pcd[:, -single_num:, :]
    pcd2 = pcd2[:, -single_num:, :]
    displace = torch.norm(pred_pcd - pcd2, dim=2)
    displace_gt = displace.mean()
    if pred_mask1.sum() == 0:
        displace_pred = torch.tensor(0.).cuda()
    else:
        displace_pred = displace[pred_mask1.squeeze(2)].mean()
    # gaussian_
    # gaussian_displace_gt, gaussian_displace_pred = Gaussian_filter_single_inter(np.array(source_pcd1.detach().cpu()),
    #                                                                             np.array(source_pcd2.detach().cpu()),
    #                                                                             np.array(pcd2.detach().cpu()),
    #                                                                             np.array(
    #                                                                                 source_pred_pcd.detach().cpu()),
    #                                                                             single_num.detach().cpu(),
    #                                                                             np.array(pred_mask1.detach().cpu()),
    #                                                                             knn_num=4,
    #                                                                             sigma=0.01)
    mask1_acc = pred_mask1.sum() / pred_mask1.shape[1] * 100
    mask2_acc = pred_mask2.sum() / pred_mask2.shape[1] * 100
    return ([np.array(mask1_acc.detach().cpu()), np.array(mask2_acc.detach().cpu()),
             np.array(displace_gt.detach().cpu()), np.array(displace_pred.detach().cpu())],
            [np.array(displace_gt.detach().cpu()), np.array(displace_pred.detach().cpu())])


def xyz_restore(xyz_in_all, relax_proportion_all):
    B = xyz_in_all.shape[0]
    result = []
    for i in range(B):
        xyz_in = xyz_in_all[i, :]
        relax_proportion = relax_proportion_all[i]
        len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
        x = xyz_in[:, 0].reshape(-1, 1)
        y = xyz_in[:, 1].reshape(-1, 1)
        z = xyz_in[:, 2].reshape(-1, 1)
        new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
        new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
        new_z = (z + 0.5) * len_z + z_min
        temp = torch.cat((new_x, new_y, new_z), axis=1).reshape(-1, 3)
        result.append(temp)
    result = torch.stack(result)
    return result


def init_model(args):
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.model_type == "Base":
        model = PTEnetBase(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow":
        model = PTFlow(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow_mean":
        model = PTFlowmean(c=6, args=args).to(args.device)
    else:
        raise SystemExit('Not impletion')
    return model


def post_train_main(model, args, io, sample_num, data, loss_fn):
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
        exit(0)
    # ============= scheduler ================
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

    model.train()
    points1, points2, colors1, colors2, mask_sum, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo = data
    points1 = points1.to(args.device)
    points2 = points2.to(args.device)
    colors1 = colors1.to(args.device)
    colors2 = colors2.to(args.device)
    mask_sum = mask_sum.to(args.device)
    mask_pseudo1 = mask_pseudo1.to(args.device)
    mask_pseudo2 = mask_pseudo2.to(args.device)
    gt_pseudo = gt_pseudo.to(args.device)
    normal2 = normal2.to(args.device)
    nn_sum_best = 30
    nn_sum_list = []
    nn_count = 0
    best_result = 0
    for epoch in range(0, args.pt_epoch):
        optimizer.zero_grad()
        l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(points1, points2, colors1,
                                                                               colors2)
        nn_loss_sum_eval, loss, nnloss, nn_true_loss, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss = loss_fn.post_losscal_mask_huamn(
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
        if epoch != args.pt_epoch - 1:
            loss.backward()
            optimizer.step()
            scheduler.step()
        pred_mask1 = torch.sigmoid(l_pred_mask1[0]) > args.mask_truncation
        pred_mask2 = torch.sigmoid(l_pred_mask2[0]) > args.mask_truncation
        mask1_sum = pred_mask1.sum() / pred_mask1[0].shape[1] / pred_mask1[0].shape[0]
        mask2_sum = pred_mask2.sum() / pred_mask2[0].shape[1] / pred_mask2[0].shape[0]
        nn_loss_sum_eval = nn_loss_sum_eval.item()
        result, gaussian_result = BiaoZhuTantTaiMaskGaussianPostTrainSource(l_pred_xyz[0], points2, points1,
                                                                            pred_mask1, pred_mask2, mask_sum)
        io.cprint(
            "nn_loss_sum_eval is {}, loss is {}, nnloss is {}, nn_true_loss is {},smooth_gradient_loss is {}, gt_pseudo_loss is {}".format(
                nn_loss_sum_eval, loss, nnloss, nn_true_loss, smooth_gradient_loss, gt_pseudo_loss
            ))
        io.cprint(
            "dataset sample_num is {},  Test epoch is {} , nn_loss_sum_eval is {}, mask_sum1 is {}, mask_sum2 is {}, "
            "mask1_acc_sum is {}, mask2_acc_sum is {}, displace_gt is {}, displace_pred is {},"
            "gaussian_displace_gt is {}, gaussian_displace_pred is {}".format(sample_num, epoch, nn_loss_sum_eval,
                                                                              mask1_sum,
                                                                              mask2_sum, result[0], result[1],
                                                                              result[2], result[3],
                                                                              gaussian_result[0],
                                                                              gaussian_result[1]))

        if nn_sum_best >= nn_loss_sum_eval:
            nn_sum_best = nn_loss_sum_eval
            # save_path = 'checkpoints/{}/{}/npz_result/{}'.format(str(args.exp_name), str(args.type_name),
            #                                                      'post_train') + "_sample_num{}_".format(
            #     sample_num) + "epoch_{}".format(epoch) + ".npz"
            # np.savez(save_path,
            #          points2=points2.cpu(),
            #          points1=points1.cpu(),
            #          colors1=colors1.cpu(),
            #          colors2=colors2.cpu(),
            #          nn_loss_sum_eval=nn_loss_sum_eval,
            #          mask_sum=mask_sum.cpu(),
            #          pred_xyz=l_pred_xyz[0].detach().cpu(),
            #          pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
            #          pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())
            best_result = np.array([mask1_sum.detach().cpu(), mask2_sum.detach().cpu(), result[0], result[1], result[2],
                                    result[3], gaussian_result[0], gaussian_result[1], nn_loss_sum_eval])
        if epoch != 0:
            if nn_loss_sum_eval > nn_sum_list[-1] * 1.5 and epoch == 1:
                nn_count += 4
            elif nn_loss_sum_eval > nn_sum_list[-1] and epoch <= 5:
                nn_count += 2
            elif nn_loss_sum_eval > nn_sum_list[-1] * 1.1 and epoch >= 6:
                nn_count += 1
        nn_sum_list.append(nn_loss_sum_eval)


    args.best_result = best_result

    # import pdb; pdb.set_trace()
    if nn_count < 3:
        save_path = 'checkpoints/{}/{}/npz_result/post_train_sample_num_{}_best.npz'.format(
            str(args.exp_name), str(args.type_name), sample_num)
        np.savez(save_path,
                 points2=points2.cpu(),
                 points1=points1.cpu(),
                 colors1=colors1.cpu(),
                 colors2=colors2.cpu(),
                 nn_loss_sum_eval=nn_loss_sum_eval,
                 mask_sum=mask_sum.cpu(),
                 pred_xyz=l_pred_xyz[0].detach().cpu(),
                 pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
                 pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())
        return False
    else:
        while count == 1:
            args_g.pt_epoch = 12
            return True
        while count == 2:
            args_g.pt_epoch = 12
            args_g.post_train_lr = 0.002
            return True
        while count == 3:
            args_g.pt_epoch = 10
                # args.need_fpfh = False
            args_g.post_train_lr = 0.004
            return True
        while count == 4:
            save_path = 'checkpoints/{}/{}/npz_result/post_train_sample_num_{}_best.npz'.format(
                str(args.exp_name), str(args.type_name), sample_num)
            np.savez(save_path,
                     points2=points2.cpu(),
                     points1=points1.cpu(),
                     colors1=colors1.cpu(),
                     colors2=colors2.cpu(),
                     nn_loss_sum_eval=nn_loss_sum_eval,
                     mask_sum=mask_sum.cpu(),
                     pred_xyz=l_pred_xyz[0].detach().cpu(),
                     pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
                     pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())
            return False


if __name__ == "__main__":
    dataset_name = "szmFilteNpz"  # szmNpz
    args_g, io_g = SH_PostTrainInit("Tantai_fitness_new16")
    # args_g, io_g = SH_PostTrainInit("Tantai_fitness")
    sample_num = "None"
    # sample_num = [1351, 2703]     # sample_num: 0 - 2703
    mask_truncation = 0.9
    dataset_Path = r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/"
    train_loader = DataLoader(HumanMarkDataSingleFilterFpfhSL(sample_num=sample_num, normalize=False,
    fpfh_path = r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/fpfhfitness"), batch_size=1)
    args_g.train_loader = train_loader
    args_g.mask_truncation = mask_truncation
    args_g.pt_epoch = 10  # 5 0.0001 0.1 20
    # cache = [args_g.gt_factor, args_g.charm_factor, args_g.smooth_factor]
    cache = [15.0, 4.0, 0.0004]
    # post train code
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    model_g = init_model(args_g)
    for index, data in loop:
        if isinstance(sample_num, list):
            index += sample_num[0]
        args_g.post_train_lr = 0.001
        args_g.gt_factor = cache[0]
        args_g.charm_factor = cache[1]
        args_g.smooth_factor = cache[2]
        save_path = 'checkpoints/{}/{}/eval_result/{}_index_{}'.format(str(args_g.exp_name), str(args_g.type_name),
                                                           'post_train', index) + ".npy"
        if os.path.exists(save_path):
            continue
        else:
            need_train = True
            count = 1
            start_time = datetime.datetime.now()
            try:
                while need_train:
                    io_g.cprint('sample_num is {}, post_train_lr is {}, '
                                'gt_factor is {}'.format(index, args_g.post_train_lr, args_g.gt_factor))
                    loss_fn = Human_SelfSupervise_mask(mask_truncation, smooth_factor=args_g.smooth_factor,
                                                       charm_factor=args_g.charm_factor,
                                                       gt_factor=args_g.gt_factor, damp_ratio=0.5,
                                                       damp_step=2)
                    need_train = post_train_main(model_g, args_g, io_g, index, data, loss_fn)
                    count = count + 1
                    args_g.post_train_lr = args_g.post_train_lr * 0.8
                    args_g.charm_factor = args_g.charm_factor * 0.8
                    args_g.gt_factor = args_g.gt_factor * 0.8
            except RuntimeError:
                with open("result.txt", "a") as f:
                    f.write(str(1))
            end_time = datetime.datetime.now()
            io_g.cprint("all time consumption is {}".format((end_time - start_time).seconds))
            np.save(save_path, args_g.best_result)
            args_g.pt_epoch = 10