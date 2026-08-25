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
# 坍台后处理  使用神经网络处理后的npz文件，对tantai数据集进行后处理操作 结果存在其余文件夹下
# 同样的，不同于Final中的其他后处理方法
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
    npz_result_path = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    npz_result_path.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    mask1_sum_list = []
    mask2_sum_list = []
    mask1_acc_list = []
    mask2_acc_list = []
    nn_loss_sum_eval_list = []
    displace_gt_list = []
    displace_pred_list = []
    mask_trun = 0.9
    for path, npz_path in zip(post_result_path, npz_result_path):
        data = np.load(path)
        mask1_sum_list.append(data[0])
        mask2_sum_list.append(data[1])
        mask1_acc_list.append(data[2])
        mask2_acc_list.append(data[3])
        nn_loss_sum_eval_list.append(data[8])

        post_npz = np.load(npz_path)
        # Gaussian
        point2 = post_npz["points2"].astype('float32').squeeze(0)
        mask_sum = post_npz["mask_sum"].squeeze(0)
        pred_pc = post_npz["pred_xyz"].astype('float32').squeeze(0)
        pred_mask1 = (post_npz["pred_mask1"].astype('float32') > mask_trun).squeeze()
        pred_pc = pred_pc[-mask_sum:, :]
        eval_point2 = point2[-mask_sum:, :]
        displace = np.sqrt(np.sum((pred_pc - eval_point2) ** 2, axis=1))
        displace_gt = displace.mean()
        eval_pred_mask1 = pred_mask1[-mask_sum:]
        if eval_pred_mask1.sum() == 0:
            displace_pred = np.nan
        else:
            displace_pred = displace[eval_pred_mask1].mean()
        displace_gt_list.append(displace_gt)
        displace_pred_list.append(displace_pred)
    mask1_sum_list = np.array(mask1_sum_list)
    mask2_sum_list = np.array(mask2_sum_list)
    mask1_acc_list = np.array(mask1_acc_list)
    mask2_acc_list = np.array(mask2_acc_list)
    nn_loss_sum_eval_list = np.array(nn_loss_sum_eval_list)
    displace_gt_list = np.array(displace_gt_list)
    displace_pred_list = np.array(displace_pred_list)
    save_path = os.path.join(root_path, "post_result.npy")
    # displace
    np.save(save_path, [mask1_sum_list, mask2_sum_list, mask1_acc_list,
                        mask2_acc_list, displace_gt_list, displace_pred_list,
                        nn_loss_sum_eval_list])
    print("save post_result.npy to {}".format(save_path))


def AShowResult(path):
    path = os.path.join(path, "post_result.npy")
    print(path)
    mean_list = []
    with open(path, 'rb') as fp:
        data = np.load(fp)
        mask1_sum = data[0]
        mask2_sum = data[1]
        mask1_acc = data[2]
        mask2_acc = data[3]
        displace_gt = data[4]
        displace_pred = data[5]
        nn_loss_sum_eval = data[6]
        displace_pred = displace_pred[~np.isnan(displace_pred)]

    true1 = displace_gt < 3
    true2 = mask1_sum < 0.8
    trueall = true1 & true2
    mean_list.append(mask1_sum.mean())
    mean_list.append(mask2_sum.mean())
    mean_list.append(mask1_acc.mean())
    mean_list.append(mask2_acc.mean())
    mean_list.append(displace_gt.mean())
    mean_list.append(displace_pred.mean())
    mean_list.append(nn_loss_sum_eval.mean())
    mean_list = np.array(mean_list)
    print("mask1_sum:{:.5f}".format(mask1_sum.mean()))
    print("mask2_sum:{:.5f}".format(mask2_sum.mean()))
    print("mask1_acc:{:.5f}".format(mask1_acc.mean()))
    print("mask2_acc:{:.5f}".format(mask2_acc.mean()))
    print("displace_gt:{:.5f}".format(displace_gt.mean()))
    print("displace_pred:{:.5f}".format(displace_pred.mean()))
    print("nn_loss_sum_eval:{:.5f}".format(nn_loss_sum_eval.mean()))
    result1, result2 = show_error12345([displace_gt, displace_pred])
    result_list = np.concatenate([result1, result2], axis=1)
    if not os.path.exists("csv_result"):
        os.mkdir("csv_result")
    SaveToExcel(result_list, r"csv_result/" + "post_train_result_{}.xlsx".format(path.split('/')[-2]))


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
        SaveToExcel(all_result, r"csv_result/" + "gaussian_result_{}.xlsx".format(path.split('/')[-2]))
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
    # root_path1 = r"/home/szm/Paconv/checkpoints/Source_Flow_softmax_topkpoint_topmask_8192/fpfh_LianXu_fitness_laterhavePeduso"
    root_path2 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness"
    # root_path3 = r"/home/szm/Paconv/checkpoints/Source_Flow_softmax_topkpoint_topmask_8192/fpfh_LianXu_fitness_laterNoPeduso"
    # root_path4 = r"/home/szm/Paconv/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192/fpfh_LianXu_fitness_laterNoPeduso"
    # root_path3 = r"/home/szm/Paconv/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192/fpfh_LianXu_fitness_laterNoPeduso_smooth15"
    # for path in [root_path1, root_path2, root_path3, root_path4, root_path5]:
    # AReadGaussianResultGoodRatio(path)
    AReadGaussianResult(root_path2)
    # AMergeAllResult(root_path1)
    # AShowResult(root_path1)
    # AMergeAllResult(root_path2)
    # AShowResult(root_path1)
    # AShowResult(root_path2)
    # AShowResult(root_path3)
    # AShowResult(root_path4)

    # AcalGaussianPath([root_path2])
    # result_path = os.path.join(root_path, "post_result.npy")
