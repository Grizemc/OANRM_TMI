#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/3 9:55
# @Author  : 沈子明
# @File    : PostTrainResultTanTai.py
# @Software: PyCharm
import glob
import multiprocessing
import os.path
import datetime

import numpy as np
from sklearn.neighbors import NearestNeighbors
"""
计算高斯滤波具体函数
"""
#   TDDP插值后处理  不同于Final中的其他后处理方式
#   hamlyn后处理 使用npz文件，即通过神经网络处理后的文件
# AcalGaussianPath函数包含了 GaussianResultEvalSingle 和 GaussianResultMultiNpz 两个函数


def GaussianResultEvalSingle(post_paths_knn_sigmas):
    file_result_list = []
    for post_paths_knn_sigma in post_paths_knn_sigmas:
        post_path, knn_num, sigma = post_paths_knn_sigma
        mask_trun = 0.9
        post_npz = np.load(post_path)
        # Gaussian
        point1 = post_npz["points1"].astype('float32').squeeze(0)
        point2 = post_npz["points2"].astype('float32').squeeze(0)
        mask_sum = post_npz["mask_sum"].squeeze(0)
        pred_pc = post_npz["pred_xyz"].astype('float32').squeeze(0)
        pred_mask1 = (post_npz["pred_mask1"].astype('float32') > mask_trun).squeeze()
        new_pred_pc = GaussianFilter(point1, pred_pc, knn_num, sigma)
        eval_new_pred_pc = new_pred_pc[-mask_sum:, :]
        eval_point2 = point2[-mask_sum:, :]
        gaussian_displace = np.sqrt(np.sum((eval_new_pred_pc - eval_point2) ** 2, axis=1))
        gaussian_displace_gt = gaussian_displace.mean()
        eval_pred_mask1 = pred_mask1[-mask_sum:]
        if eval_pred_mask1.sum() == 0:
            gaussian_displace_pred = np.nan
        else:
            gaussian_displace_pred = gaussian_displace[eval_pred_mask1].mean()

        # 　True Gaussian
        if pred_mask1.sum() >= 20:
            point1_in_mask1 = point1[pred_mask1]
            pred1_in_mask1 = pred_pc[pred_mask1]
            new_pred_pc_in_mask1 = GaussianFilter(point1_in_mask1, pred1_in_mask1, knn_num, sigma)
            new_pred_pc_out_mask1 = pred_pc.copy()
            new_pred_pc_out_mask1[pred_mask1] = new_pred_pc_in_mask1
            eval_new_pred_pc_out_mask1 = new_pred_pc_out_mask1[-mask_sum:, :]
            eval_point2 = point2[-mask_sum:, :]
            gaussian_true_displace = np.sqrt(np.sum((eval_new_pred_pc_out_mask1 - eval_point2) ** 2, axis=1))
            gaussian_true_displace_gt = gaussian_true_displace.mean()
            eval_pred_mask1 = pred_mask1[-mask_sum:]
            if eval_pred_mask1.sum() == 0:
                gaussian_true_displace_pred = np.nan
            else:
                gaussian_true_displace_pred = gaussian_true_displace[eval_pred_mask1].mean()
        else:
            gaussian_true_displace_gt = np.nan
            gaussian_true_displace_pred = np.nan
        file_result_list.append(
            [post_path, gaussian_displace_gt, gaussian_displace_pred, gaussian_true_displace_gt,
             gaussian_true_displace_pred])
    return file_result_list


def GaussianFilter(point1, pred_pc, knn_num, sigma):
    flow = pred_pc - point1
    nbrs = NearestNeighbors(n_neighbors=knn_num + 1, algorithm='auto').fit(point1)
    _, indices = nbrs.kneighbors(point1)
    flow_neigh = flow[indices]
    indices = indices[:, 1:]
    point1_neigh = point1[indices]
    point1_neigh_relative = point1_neigh - point1[:, np.newaxis, :]
    center_point = np.zeros_like(flow)
    point1_relative = np.concatenate((center_point[:, np.newaxis, :], point1_neigh_relative), axis=1)
    gaussian_weight_up = np.exp(-(np.sum(point1_relative ** 2, 2)) / (2 * sigma ** 2))
    gaussian_weight_down = (np.power(2 * np.pi, 1.5) * np.power(sigma, 3))
    gaussian_weight = gaussian_weight_up / gaussian_weight_down
    gaussian_weight = (gaussian_weight / gaussian_weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
    flow_result = (flow_neigh * gaussian_weight).sum(axis=1)
    new_pred_pc = flow_result + point1
    return new_pred_pc


def show_error12345(datas, if_print=True):
    sum_count = []
    sum_count_percentage = []
    errors = [1, 2, 3, 4, 5]
    for data in datas:
        count = []
        count_percentage = []
        sum = data.shape[0]
        for error in errors:
            count.append((data < error).sum())
            count_percentage.append((data < error).sum() / sum * 100)
        if if_print:
            print(
                "error 1 is {:.5f},2 is {:.5f}, 3 is {:.5f}, 4 is {:.5f}, 5 is {:.5f}".format(count[0], count[1],
                                                                                              count[2],
                                                                                              count[3], count[4]))
            print(" percentage, error 1 is {:.5f},2 is {:.5f}, 3 is {:.5f}, 4 is {:.5f}, 5 is {:.5f}".format(
                count_percentage[0],
                count_percentage[1],
                count_percentage[2],
                count_percentage[3],
                count_percentage[4]))
        sum_count.append(count)
        sum_count_percentage.append(count_percentage)
    return np.array(sum_count), np.array(sum_count_percentage)


def GaussianResultMultiNpz(root_path, knn_num, sigma, count):
    npz_result_path = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    npz_result_path.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    npz_result_path = [[path, knn_num, sigma] for path in npz_result_path]
    all_len = len(npz_result_path)
    temp_len = all_len // 20
    file_group_list = []
    last_number = 0
    for i in range(0, all_len, temp_len):
        if i != 0:
            temp_list = npz_result_path[last_number:i]
            file_group_list.append(temp_list)
        last_number = i
    if last_number != all_len:
        temp_list = npz_result_path[last_number:all_len]
        file_group_list.append(temp_list)
    if count == 0:
        multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    pool = multiprocessing.Pool(20)
    result = pool.map(GaussianResultEvalSingle, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出

    flattened_list = [value for sublist in result for value in sublist]
    float_values = [sublist[1:] for sublist in flattened_list]
    float_values = np.array(float_values)
    gaussian_result_gt_list = np.array(float_values[:, 0])
    gaussian_result_pred_list = np.array(float_values[:, 1])
    gaussian_true_displace_gt_list = np.array(float_values[:, 2])
    gaussian_true_displace_pred_list = np.array(float_values[:, 3])
    cleaned_gaussian_result_pred_list = gaussian_result_pred_list[~np.isnan(gaussian_result_pred_list)]
    cleaned_gaussian_true_displace_gt_list = gaussian_true_displace_gt_list[~np.isnan(gaussian_true_displace_gt_list)]
    cleaned_gaussian_displace_pred_list = gaussian_true_displace_pred_list[~np.isnan(gaussian_true_displace_pred_list)]
    return gaussian_result_gt_list, cleaned_gaussian_result_pred_list, cleaned_gaussian_true_displace_gt_list, cleaned_gaussian_displace_pred_list


def AMergeAllResult(root_path):
    post_result_path = glob.glob(os.path.join(root_path, "eval_result", "*[0-9].npy"))
    post_result_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0].split('_')[-1]))
    test_acc = []
    mask_acc = []
    precise_mask_acc = []
    for path in post_result_path:
        data = np.load(path)
        test_acc.append(data[0:6])
        mask_acc.append(data[6:11])
        precise_mask_acc.append(data[11:17])
    test_acc = np.array(test_acc)
    mask_acc = np.array(mask_acc)
    precise_mask_acc = np.array(precise_mask_acc)
    print("test_acc mean is {}".format(test_acc.mean(axis=0)))
    print("mask_acc mean is {}".format(mask_acc.mean(axis=0)))
    print("precise_mask_acc mean is {}".format(precise_mask_acc.mean(axis=0)))
    save_path = os.path.join(root_path, "post_result.npz")
    # displace
    np.savez(save_path, test_acc=test_acc, mask_acc=mask_acc, precise_mask_acc=precise_mask_acc)
    print("save post_result.npy to {}".format(save_path))

    post_npz_path = glob.glob(os.path.join(root_path,"npz_result", "*.npz"))
    post_npz_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-2]))
    mask_acc1 = []
    mask_acc2 = []
    pred_mask_ratio1 =[]
    pred_mask_ratio2 = []
    mask_trucation = 0.9
    for path in post_npz_path:
        data = np.load(path)
        mask_gt1 = data["mask_gt1"]
        mask_gt2 = data["mask_gt2"]
        pred_mask1 = data["pred_mask1"].squeeze(-1) > mask_trucation
        pred_mask2 = data["pred_mask2"] .squeeze(-1)> mask_trucation
        mask_acc1.append((mask_gt1 == pred_mask1).sum() / mask_gt1.size)
        mask_acc2.append((mask_gt2 == pred_mask2).sum() / mask_gt2.size)
        pred_mask_ratio1.append(pred_mask1.sum() / pred_mask1.size)
        pred_mask_ratio2.append(pred_mask2.sum() / pred_mask2.size)
    mask_acc1 = np.array(mask_acc1)
    mask_acc2 = np.array(mask_acc2)
    pred_mask_ratio1 = np.array(pred_mask_ratio1)
    pred_mask_ratio2 = np.array(pred_mask_ratio2)
    print("mask_acc1 mean is {}".format(mask_acc1.mean())) # 0.9373423144033628
    print("mask_acc2 mean is {}".format(mask_acc2.mean())) # 0.9370170843110249
    print("pred_mask_ratio1 mean is {}".format(pred_mask_ratio1.mean())) #  0.9635125436160781
    print("pred_mask_ratio2 mean is {}".format(pred_mask_ratio2.mean()))  #  0.9580344830067318

def AShowResult(path):
    path = os.path.join(path, "post_result.npz")
    print(path)
    mean_list =[]
    with open(path, 'rb') as fp:
        fp = np.load(fp)
        test_acc = fp['test_acc']
        mask_acc = fp['mask_acc']
        precise_mask_acc = fp['precise_mask_acc']
    print("test_acc mean is {}".format(test_acc.mean(axis=0)))
    print("mask_acc mean is {}".format(mask_acc.mean(axis=0)))
    print("precise_mask_acc mean is {}".format(precise_mask_acc.mean(axis=0)))



def AReadGaussianResultGoodRatio(root_path):
    print("root_path is {}".format(root_path))
    gaussian_result_paths = glob.glob(os.path.join(root_path, "gaussian*.npz"))
    mean_result_list = []
    gaussian_count_result_list = []
    gaussian_percentage_result_list = []
    for path in gaussian_result_paths:
        with np.load(path) as npz:
            temp_mean_result_list = []
            gaussian_result_gt_list = npz['gaussian_result_gt_list']
            gaussian_result_pred_list = npz['gaussian_result_pred_list']
            gaussian_true_displace_gt_list = npz['gaussian_true_displace_gt_list']
            gaussian_true_displace_pred_list = npz['gaussian_true_displace_pred_list']
            temp_mean_result_list.append(gaussian_result_gt_list.mean())
            temp_mean_result_list.append(gaussian_result_pred_list.mean())
            temp_mean_result_list.append(gaussian_true_displace_gt_list.mean())
            temp_mean_result_list.append(gaussian_true_displace_pred_list.mean())
            mean_result_list.append(temp_mean_result_list)
            count_result, percentage_result = show_error12345([gaussian_true_displace_gt_list,
                                                               gaussian_true_displace_pred_list,
                                                               gaussian_result_gt_list, gaussian_result_pred_list],
                                                              if_print=False)
            gaussian_count_result_list.append(count_result)
            gaussian_percentage_result_list.append(percentage_result)
    mean_result_list = np.array(mean_result_list)
    gaussian_count_result_list = np.array(gaussian_count_result_list)
    gaussian_percentage_result_list = np.array(gaussian_percentage_result_list)
    min_row_indices = np.argmin(mean_result_list, axis=0)
    print("min_row_indices:{}".format(min_row_indices))
    for i in range(5):
        gaussian_percentage_result = gaussian_percentage_result_list[:, :, i]
        min_row_indices = np.argmax(gaussian_percentage_result, axis=0)
        print("min_row_indices:{}".format(min_row_indices))



def AReadGaussianResult(root_path):
    gaussian_result_paths = glob.glob(os.path.join(root_path, "gaussian*.npz"))
    path = gaussian_result_paths[6]
    print("root_path is {}".format(path))
    with np.load(path) as npz:
        gaussian_result_gt_list = npz['gaussian_result_gt_list']
        gaussian_result_pred_list = npz['gaussian_result_pred_list']
        gaussian_true_displace_gt_list = npz['gaussian_true_displace_gt_list']
        gaussian_true_displace_pred_list = npz['gaussian_true_displace_pred_list']
        print("gaussian_result_gt_list mean is {}".format(gaussian_result_gt_list.mean()))
        print("gaussian_result_pred_list mean is {}".format(gaussian_result_pred_list.mean()))
        print("gaussian_true_displace_gt_list mean is {}".format(gaussian_true_displace_gt_list.mean()))
        print("gaussian_true_displace_pred_list mean is {}".format(gaussian_true_displace_pred_list.mean()))
        count_result, percentage_result = show_error12345([gaussian_true_displace_gt_list,
                                                           gaussian_true_displace_pred_list,
                                                           gaussian_result_gt_list, gaussian_result_pred_list],
                                                          if_print=True)
        mean_result_list = np.array(
            [gaussian_result_gt_list.mean(), gaussian_result_pred_list.mean(), gaussian_true_displace_gt_list.mean(),
             gaussian_true_displace_pred_list.mean()]).reshape(4, 1)
        all_result = np.concatenate((mean_result_list, count_result, percentage_result), axis=1)
        SaveToExcel(all_result,  r"csv_result/"+"gaussian_result_{}.xlsx".format(path.split('/')[-2]))
    print("==========" * 2)


def SaveToExcel(percentage_result, name="percentage_result.xlsx"):
    import pandas as pd
    df = pd.DataFrame(np.transpose(percentage_result))
    df.to_excel(name)

def AcalGaussianPath(root_paths):
    count = 0
    for root_path in root_paths:
        for knn_num in [3, 4, 5, 6, 7]:
            for sigma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4, 5, 5]:
                gaussian_result = GaussianResultMultiNpz(root_path=root_path, knn_num=knn_num, sigma=sigma,
                                                         count=count)
                save_path = os.path.join(root_path, "gaussian_knn_num{}_sigma{}_result.npz".format(knn_num, sigma))
                np.savez(save_path, gaussian_result_gt_list=gaussian_result[0],
                         gaussian_result_pred_list=gaussian_result[1],
                         gaussian_true_displace_gt_list=gaussian_result[2],
                         gaussian_true_displace_pred_list=gaussian_result[3]
                         )
                count += 1


if __name__ == "__main__":
    root_path1 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation"

    # AMergeAllResult(root_path4)
    # AShowResult(root_path5)
    # AMergeAllResult(root_path1)
    # AMergeAllResult(root_path4)
    # AMergeAllResult(root_path3)
    # AShowResult(root_path1)
    # AMergeAllResult(root_path2)
    # AShowResult(root_path2)
    # AcalGaussianPath([root_path1])
    # result_path = os.path.join(root_path, "post_result.npy")
