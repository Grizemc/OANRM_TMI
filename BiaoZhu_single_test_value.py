#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 16:36
# @Author  : 沈子明
# @File    : Big_main.py
# @Software: PyCharm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.neighbors import NearestNeighbors, KDTree  # 导入knn算法类
from lib.pointops.functions import pointops
from model.backbone_new import PTEnetBase, PTFlow
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.data import  HumanMarkDataSingle
from util.util import load_cfg_from_cfg_file, IOStream

"""
使用澹台佑彤师姐标注的数据集，进行评价
直接测试，不进行在线学习微调,该代码进行了数据的归一化，有错误
应该不需要进行归一化
"""


def init(type, root_directory_name):
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml',
                        help='config file')
    args_l = parser.parse_args()
    assert args_l.config is not None
    args = load_cfg_from_cfg_file(args_l.config)
    # -----------------------------------------------------------------------------
    # backup the running files

    if not os.path.exists('checkpoints/' + args.exp_name + '/' + root_directory_name):
        os.makedirs('checkpoints/' + args.exp_name + '/' + root_directory_name)
    source_io = IOStream(
        'checkpoints/' + args.exp_name + '/' + root_directory_name + '/BiaoZhu_test_{}_single_relax.log'.format(
            type))
    source_io.cprint(
        'checkpoints/' + args.exp_name + '/' + root_directory_name + '/BiaoZhu_test_{}_single_relax.log'.format(
            type))
    # -----------------------------------------------------------------------------
    # set random seed
    # -----------------------------------------------------------------------------
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        source_io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        source_io.cprint('Using CPU')
    source_io.cprint(str(args))
    args.source_io = source_io
    args.dataset_type = type
    args.directory = root_directory_name
    return args


def Gaussian_filter_single_inter(xyz1, source_pcd2, pcd2, pred_pc, relax_ratio2, single_num, pred_mask, knn_num=5,
                                 sigma=0.2):
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

    new_pred_pcd = np_xyz_restore(single_new_pred[np.newaxis, ::], relax_ratio2)
    new_pred_pcd = new_pred_pcd[:, -single_num[0]:, :]

    gaussian_displace = np.sqrt(np.sum((new_pred_pcd - pcd2) ** 2, axis=2))
    gaussian_displace_gt = gaussian_displace.mean()

    if pred_mask.sum() == 0:
        gaussian_displace_pred = 0.
    else:
        gaussian_displace_pred = gaussian_displace[pred_mask.squeeze(2)].mean()
    return gaussian_displace_gt, gaussian_displace_pred


def Gaussian_filter_all_inter(xyz1, pred, self_ratio, knn_num, sigma=0.1):
    # Gaussian_filter_gradient_inter no mask
    # xyz1 = xyz_restore(xyz1, relax_ratio1)
    B, t_num, _ = xyz1.shape
    flow = pred - xyz1
    idx1 = pointops.knnquery(knn_num, xyz1, xyz1)
    idx1 = idx1[:, :, 1:].contiguous()
    neigh_flow = pointops.grouping(flow.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3, 1).contiguous()
    neigh_xyz1 = pointops.grouping(xyz1.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3, 1).contiguous()
    relative_xyz = neigh_xyz1 - xyz1.unsqueeze(-2)
    gaussian_weight_up = torch.exp(-(relative_xyz ** 2).sum(dim=-1) / (sigma ** 2 * 2))
    gaussian_weight_down = torch.tensor(np.power(2 * np.pi, 1.5) * np.power(sigma, 3))
    gaussian_weight = gaussian_weight_up / gaussian_weight_down
    gaussian_weight = (gaussian_weight / gaussian_weight.sum(dim=-1, keepdims=True))
    new_flow = (neigh_flow * gaussian_weight.unsqueeze(-1)).sum(dim=-2) * (1 - self_ratio) + self_ratio * flow
    new_pred = new_flow + xyz1
    return new_pred


def BiaoZhuTantTaiMaskGaussianSingle(source_pred_pcd, source_pcd2, source_pcd1, relax_ratio1, relax_ratio2,
                                     source_pred_mask1,
                                     source_pred_mask2, single_num):
    pred_mask1 = source_pred_mask1[:, -single_num:, :]
    pred_mask2 = source_pred_mask2[:, -single_num:, :]
    pred_pcd = xyz_restore(source_pred_pcd, relax_ratio2)
    pcd2 = xyz_restore(source_pcd2, relax_ratio2)
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
    gaussian_displace_gt, gaussian_displace_pred = Gaussian_filter_single_inter(np.array(source_pcd1.cpu()),
                                                                                np.array(source_pcd2.cpu()),
                                                                                np.array(pcd2.cpu()),
                                                                                np.array(source_pred_pcd.cpu()),
                                                                                np.array(relax_ratio2.cpu()),
                                                                                single_num.cpu(),
                                                                                np.array(pred_mask1.cpu()),
                                                                                knn_num=4,
                                                                                sigma=0.01)
    mask1_acc = pred_mask1.sum() / pred_mask1.shape[1] * 100
    mask2_acc = pred_mask2.sum() / pred_mask2.shape[1] * 100
    return [np.array(mask1_acc.cpu()), np.array(mask2_acc.cpu()), np.array(displace_gt.cpu()),
            np.array(displace_pred.cpu())], [gaussian_displace_gt, gaussian_displace_pred]


def BiaoZhuTantTaiMask(pred_pcd, pcd2, pcd1, relax_ratio2, pred_mask1, pred_mask2, mask_gt):
    pred_mask1 = pred_mask1[:, -5:, :]
    pred_mask2 = pred_mask2[:, -5:, :]
    pred_mask1_true = pred_mask1[mask_gt]
    pred_mask2_true = pred_mask2[mask_gt]
    pred_mask = torch.eq(pred_mask1, mask_gt)
    pred_pcd = xyz_restore(pred_pcd, relax_ratio2)
    pcd2 = xyz_restore(pcd2, relax_ratio2)
    # source
    pred_pcd = pred_pcd[:, -5:, :]
    pcd2 = pcd2[:, -5:, :]
    displace = torch.norm(pred_pcd - pcd2, dim=2)
    displace_gt = displace[mask_gt.squeeze(2)].mean()
    displace_pred = displace[pred_mask.squeeze(2)].mean()
    # gaussian_
    mask1_acc = pred_mask1_true.sum() / pred_mask1_true.shape[0] * 100
    mask2_acc = pred_mask2_true.sum() / pred_mask2_true.shape[0] * 100
    return mask1_acc, mask2_acc, displace_gt, displace_pred


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


def np_xyz_restore(xyz_in_all, relax_proportion_all):
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
        temp = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
        result.append(temp)
    result = np.stack(result)
    return result


def Normalize_Mask_Data_run(dataloader, model, args):
    if args.model_test_type == "Train":
        model.train()
        args.source_io.cprint("model.train()")
    elif args.model_test_type == "eval":
        model.eval()
        args.source_io.cprint("model.eval()")
    mask1_sum = 0.0
    mask2_sum = 0.0
    mask1_acc_sum = 0.0
    mask2_acc_sum = 0.0
    displace_gt_sum = 0.0
    displace_pred_sum = 0.0
    gaussian_displace_gt_sum = 0.0
    gaussian_displace_pred_sum = 0.0
    result_list = []
    nan_sum = 0.0
    with torch.no_grad():
        for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, normal2, single_num = data
            points1 = points1.to(args.device)
            points2 = points2.to(args.device)
            colors1 = colors1.to(args.device)
            colors2 = colors2.to(args.device)
            single_num = single_num.to(args.device)
            relax_ratio1 = relax_ratio1.to(args.device)
            relax_ratio2 = relax_ratio2.to(args.device)
            l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(points1, points2, colors1,
                                                                                   colors2)
            pred_mask1 = torch.sigmoid(l_pred_mask1[0]) > args.mask_ratio
            pred_mask2 = torch.sigmoid(l_pred_mask2[0]) > args.mask_ratio
            result, gaussian_result = BiaoZhuTantTaiMaskGaussianSingle(l_pred_xyz[0], points2, points1, relax_ratio1,
                                                                       relax_ratio2,
                                                                       pred_mask1, pred_mask2, single_num)
            result_list.append(
                np.array([pred_mask1.float().mean().cpu(), pred_mask1.float().mean().cpu(),
                          result[0], result[1], result[2], result[3], gaussian_result[0], gaussian_result[1]]))
            # np.savez("checkpoints/{}/{}/Human_no_post_train_result_index{}.npz".format(str(args.exp_name), str(args.directory)
            #                                                                         index), points1=points1.cpu(),
            #          points2=points2.cpu(),
            #          colors1=colors1.cpu(),
            #          colors2=colors2.cpu(),
            #          mask=mask.squeeze().cpu(),
            #          relax_ratio1=relax_ratio1.cpu(),
            #          relax_ratio2=relax_ratio2.cpu(),
            #          pred_xyz=l_pred_xyz[0].cpu(),
            #          pred_mask1=torch.sigmoid(l_pred_mask1[0]).cpu(),
            #          pred_mask2=torch.sigmoid(l_pred_mask2[0]).cpu())
            mask1_acc_sum += result[0]
            mask2_acc_sum += result[1]
            if result[3] == 0:
                nan_sum += 1
            displace_gt_sum += result[2]
            displace_pred_sum += result[3]
            gaussian_displace_gt_sum += gaussian_result[0]
            gaussian_displace_pred_sum += gaussian_result[1]
            mask1 = pred_mask1.sum() / pred_mask1.shape[1] / pred_mask1.shape[0] * 100
            mask1_sum += mask1
            mask2 = pred_mask2.sum() / pred_mask2.shape[1] / pred_mask2.shape[0] * 100
            mask2_sum += mask2
            args.source_io.cprint(
                "Human Test,index {} , mask_sum1 is {}, mask_sum2 is {}, mask1_acc_sum is {}, mask2_acc_sum is {}, "
                "displace_gt is {}, displace_pred is {}".format(index, mask1, mask2, result[0],
                                                                result[1], result[2],
                                                                result[3]))
            args.source_io.cprint(
                "Human Test,index {} , gaussian_displace_gt is {}, gaussian_displace_pred is {}".format(
                    index, gaussian_result[0], gaussian_result[1]))
        np.save("checkpoints/{}/{}/Human_{}_result.npz".format(str(args.exp_name), str(args.directory), str(args.dataset_type)),
                np.array(result_list))
    return [mask1_sum / (index + 1), mask2_sum / (index + 1), mask1_acc_sum / (index + 1), mask2_acc_sum / (index + 1), \
            displace_gt_sum / (index + 1), displace_pred_sum / (index + 1 - nan_sum)], \
        [gaussian_displace_gt_sum / (index + 1), gaussian_displace_pred_sum / (index + 1 - nan_sum)]


if __name__ == "__main__":
    dataset_Path = r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/"
    # dataset_Path = r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmNpz/"
    model_test_type = "eval"  # "eval
    if dataset_Path == "/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/":
        if model_test_type == "Train":
            root_directory_name = "Human_mark_train"
        elif model_test_type == "eval":
            root_directory_name = "Human_mark"
    elif dataset_Path == r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmNpz/":
        if model_test_type == "Train":
            root_directory_name = "Human_mark_filter_train"
        elif model_test_type == "eval":
            root_directory_name = "Human_mark_filter"
    mask_ratio = 0.9
    dataset_type = "test"
    # for type in ["train", "test"]:
    args = init(dataset_type, root_directory_name)
    args.mask_ratio = mask_ratio
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.model_test_type = model_test_type
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.model_type == "Base":
        model = PTEnetBase(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow":
        model = PTFlow(c=6, args=args).to(args.device)
    else:
        raise SystemExit('Not impletion')
    modle_path = os.path.join('checkpoints/', args.exp_name, "saved_model/best_model.t7")
    try:
        model.load_state_dict(torch.load(modle_path))
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(modle_path))
    total = sum([param.nelement() for param in model.parameters()])
    args.source_io.cprint("Number of parameter: %.2fM" % (total / 1e6))
    Dastaset_human = HumanMarkDataSingle(type=dataset_type, root=dataset_Path)
    abs_test_loader = DataLoader(Dastaset_human, batch_size=1, shuffle=False, drop_last=False)
    test_acc, gaussian_test_acc = Normalize_Mask_Data_run(abs_test_loader, model, args)
    args.source_io.cprint(
        "Human Test all , mask_sum1 is {}, mask_sum2 is {}, mask1_acc_sum is {}, mask2_acc_sum is {}, "
        "displace_gt_sum is {}, displace_pred_sum is {}".format(test_acc[0], test_acc[1], test_acc[2],
                                                                test_acc[3], test_acc[4], test_acc[5]))
    args.source_io.cprint(
        "Human Test all , Gaussian displace_gt_sum is {}, Gaussian displace_pred_sum is {}".format(
            gaussian_test_acc[0],
            gaussian_test_acc[1]))
