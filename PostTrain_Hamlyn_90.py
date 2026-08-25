#  无监督过拟合两帧图像
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from util.data import HumanMarkDataSingle, HumanMarkDataSingleFilterFpfh, HumanMarkDataSingleFilterFpfhSL, \
    HamlynArtificialFPFH
from util.util import load_cfg_from_cfg_file, IOStream
import lib.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
import open3d as o3d
import sys

""""
无监督微调，验证Hamlyn数据集的精度，用于验证一个新生成的低重叠率的数据集，用于扩展最后实验结果的数量。
"""


def Miccai_absolute_strength_evaluate_result(pcd1, pcd2, mask):
    """
    predict is success when the predicted value is evaluate_indicator%(default 0.2) above or below the true value.
    Args:
        pcd1:  source point cloud xyz  [bs, num_points, 3]
        pcd2: target point cloud xyz
    Returns: Acc
    """
    pcd1 = pcd1[mask, :]
    pcd2 = pcd2[mask, :]
    num = mask.sum()
    displace = torch.norm(pcd1 - pcd2, dim=1)
    relax_error = [1, 2, 3, 5, 10]
    acc_list = []
    for error in relax_error:
        acc_list.append((displace < error).sum() / num)
    acc_list.append(displace.mean() / 100)
    acc_list = torch.tensor(acc_list).to(pcd1.device)
    return acc_list


def Mask_evaluate_result(pred, gt):
    pred = pred.squeeze()
    B, N = gt.shape[0], gt.shape[1]
    truncation_nums = [0.7, 0.75, 0.8, 0.85, 0.9]
    success_ratio = []
    for truncation_num in truncation_nums:
        pred_trun = pred > truncation_num
        success_ratio.append(pred_trun.eq(gt).sum() / (B * N))
    success_ratio = torch.tensor(success_ratio)
    return success_ratio


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
        smooth_gradient_loss = jacob_mat.norm(dim=-1).mean() * smooth_weight
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

    def post_losscal_hamlyn_mask(self, l_xyz1, l_new_xyz, xyz2, color1, color2, l_mask1, l_mask2, mask_pseudo1,
                                 mask_pseudo2, normal2, gt_pseudo, loss_epoch, need_fpfh):
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
        nn_loss_sum_eval = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true)
        nn_loss_sum = self.chamfer_distance(pred, xyz2, normal2, plane_true=self.plan_true) * self.gt_a
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
        single_mask1 = torch.relu(torch.sigmoid(mask1[0]) - self.mask_truncation)
        single_mask2 = torch.relu(torch.sigmoid(mask2[0]) - self.mask_truncation)
        single_mask1 = single_mask1.bool()
        single_mask2 = single_mask2.bool()
        overlap_pred = pred[single_mask1.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)
        overlap_xyz2 = xyz2[single_mask2.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)
        normal2 = normal2[single_mask2.unsqueeze(0).repeat(1, 1, 3)].reshape(1, -1, 3)

        if single_mask1.sum() / single_mask1.shape[0] < 0.5 or single_mask2.sum() / single_mask2.shape[0] < 0.5:
            if need_fpfh:
                loss = nn_loss_sum * 2 + smooth_gradient_loss + gt_pseudo_loss
            else:
                loss = nn_loss_sum * 2 + smooth_gradient_loss
            # loss = nn_loss_sum
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss
        else:
            # NN true loss;
            nn_true_loss_sum = self.chamfer_distance(overlap_pred, overlap_xyz2, normal2,
                                                     plane_true=self.plan_true) * self.gt_a

            # loss = nn_true_loss_sum+ nn_loss_sum+
            if need_fpfh:
                loss = nn_true_loss_sum + nn_loss_sum + smooth_gradient_loss + gt_pseudo_loss
            else:
                loss = nn_true_loss_sum + nn_loss_sum + smooth_gradient_loss
            return nn_loss_sum_eval, loss, nn_loss_sum, nn_true_loss_sum, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss


def SH_PostTrainInit(type_name):
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='config/Zall.yaml',
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
    io = IOStream('checkpoints/' + args.exp_name + '/' + type_name + '/post_train.log')
    file_name = os.path.basename(sys.argv[0])
    os.system(
        'cp -r {} checkpoints/'.format(file_name) + args.exp_name + '/' + type_name + '/{}.txt'.format(file_name))
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
    test_acc = 0.0
    mask_acc = 0.0
    precise_pred_mask_acc = 0.0
    model.train()
    points1, points2, colors1, colors2, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo, mask_gt1, mask_gt2, mask_gt_pc = data
    points1 = points1.to(args.device)
    points2 = points2.to(args.device)
    colors1 = colors1.to(args.device)
    colors2 = colors2.to(args.device)
    mask_pseudo1 = mask_pseudo1.to(args.device)
    mask_pseudo2 = mask_pseudo2.to(args.device)
    gt_pseudo = gt_pseudo.to(args.device)
    normal2 = normal2.to(args.device)
    mask_gt1 = mask_gt1.to(args.device)
    mask_gt2 = mask_gt2.to(args.device)
    mask_gt_pc = mask_gt_pc.to(args.device)
    nn_sum_best = 30
    nn_sum_list = []
    nn_count = 0

    for epoch in range(0, args.pt_epoch):
        optimizer.zero_grad()
        l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(points1, points2, colors1,
                                                                               colors2)
        # ============= loss ================
        nn_loss_sum_eval, loss, nnloss, nn_true_loss, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss = loss_fn.post_losscal_hamlyn_mask(
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
            loss_epoch=epoch,
            need_fpfh=args.need_fpfh)
        io.cprint("nn_loss_sum_eval is {}, loss is {}, nnloss is {}, nn_true_loss is {}"
                  ", smooth_gradient_loss is {}, mask_loss_sum is {} , gt_pseudo_loss is {} ".format(
            nn_loss_sum_eval, loss, nnloss, nn_true_loss, smooth_gradient_loss, mask_loss_sum, gt_pseudo_loss
        ))
        if epoch != args.pt_epoch - 1:
            loss.backward()
            optimizer.step()
            scheduler.step()
        test_acc = Miccai_absolute_strength_evaluate_result(l_pred_xyz[0], mask_gt_pc, mask_gt1) * 100
        mask_acc = Mask_evaluate_result(torch.sigmoid(l_pred_mask1[0]), mask_gt1)
        precise_mask_acc = Miccai_absolute_strength_evaluate_result(pcd1=l_pred_xyz[0], pcd2=mask_gt_pc, mask=(
                torch.sigmoid(l_pred_mask1[0]) > 0.9).squeeze(-1)) * 100

        if nn_sum_best >= nn_loss_sum_eval:
            nn_sum_best = nn_loss_sum_eval
            pred_mask1 = torch.sigmoid(l_pred_mask1[0]) > args.mask_truncation
            pred_mask2 = torch.sigmoid(l_pred_mask2[0]) > args.mask_truncation
            mask1_sum = pred_mask1.sum() / pred_mask1[0].shape[1] / pred_mask1[0].shape[0]
            mask2_sum = pred_mask2.sum() / pred_mask2[0].shape[1] / pred_mask2[0].shape[0]
            best_result = [test_acc.cpu(), mask_acc.cpu(), precise_mask_acc.cpu()]
        if epoch != 0:
            if nn_loss_sum_eval > nn_sum_list[-1] * 1.5 and epoch == 1:
                nn_count += 4
            elif nn_loss_sum_eval > nn_sum_list[-1] and epoch <= 5:
                nn_count += 2
            elif nn_loss_sum_eval > nn_sum_list[-1] * 1.1 and epoch >= 6:
                nn_count += 1
        nn_sum_list.append(nn_loss_sum_eval)

    string = "test_acc is {}, mask_acc is {}, precise_pred_mask_acc is {}".format(best_result[0], best_result[1],
                                                                                  best_result[2])
    io.cprint(string)
    args.best_result = best_result
    if nn_count > 3:
        return True
    else:
        temp_true_false = mask1_sum < 0.2 or mask1_sum == 1.0
        if temp_true_false and args.need_fpfh:
            args.need_fpfh = False
            args_g.post_train_lr = 0.004
            return True
        else:
            save_path = 'checkpoints/{}/{}/npz_result/post_train_sample_num_{}_best.npz'.format(
                str(args.exp_name), str(args.type_name), sample_num)
            np.savez(save_path,
                     points2=points2.cpu(),
                     points1=points1.cpu(),
                     colors1=colors1.cpu(),
                     colors2=colors2.cpu(),
                     nn_loss_sum_eval=nn_loss_sum_eval.detach().cpu(),
                     pred_xyz=l_pred_xyz[0].detach().cpu(),
                     mask_gt1=mask_gt1.cpu(),
                     mask_gt2=mask_gt2.cpu(),
                     mask_gt_pc=mask_gt_pc.detach().cpu(),
                     pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
                     pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())
            return False


def flatten_nested_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_nested_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


if __name__ == "__main__":
    args_g, io_g = SH_PostTrainInit("Post_Train_Hamlyn_no_rotation_90")
    # sample_num = [1630, 1926] # 1793
    # sample_num = [1926, 2222] # 1793
    # sample_num = [2222, 2518] # 1793
    # sample_num = [2518, 2821] # 1793
    sample_num = "None"  # 1793
    mask_truncation = 0.9
    test_hamlyn_loader = DataLoader(HamlynArtificialFPFH(
        root="/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_90/test",
        fpfh_path="/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_90/fpft_file",
        sample_num=sample_num, normalize=False, continuous=True),
        batch_size=1)

    args_g.train_loader = test_hamlyn_loader
    args_g.mask_truncation = mask_truncation
    args_g.pt_epoch = 10
    args_g.smooth_factor = 1.0
    args_g.mask_truncation = 0.8
    # post train code
    loop = tqdm(enumerate(test_hamlyn_loader), total=len(test_hamlyn_loader))
    model_g = init_model(args_g)
    for index, data in loop:
        save_path = 'checkpoints/{}/{}/npz_result/post_train_sample_num_{}_best.npz'.format(
            str(args_g.exp_name), str(args_g.type_name), index)
        if os.path.exists(save_path):
            pass
        else:
            args_g.post_train_lr = 0.001
            args_g.gt_factor = 2.0
            args_g.need_fpfh = True

            # if os.path.exists(save_path):
            #     continue
            # else:
            need_train = True
            start_time = datetime.datetime.now()
            try:
                while need_train:
                    io_g.cprint('sample_num is {}, post_train_lr is {}, '
                                'gt_factor is {}'.format(index, args_g.post_train_lr, args_g.gt_factor))
                    loss_fn = Human_SelfSupervise_mask(mask_truncation, smooth_factor=args_g.smooth_factor,
                                                       gt_factor=args_g.gt_factor, damp_ratio=0.5,
                                                       damp_step=2)
                    need_train = post_train_main(model_g, args_g, io_g, index, data, loss_fn)
                    args_g.post_train_lr = args_g.post_train_lr * 0.8
            except RuntimeError:
                with open("result.txt", "a") as f:
                    f.write(str(1))
            end_time = datetime.datetime.now()
            # io_g.cprint("all time consumption is {}".format((end_time - start_time).seconds))
            # asd1 = np.array(args_g.best_result[0])
            # asd2 = np.array(args_g.best_result[1])
            # asd3 = np.array(args_g.best_result[2])
            # save_all = np.concatenate((asd1, asd2, asd3), axis=0)
            # np.save(save_path, save_all)
