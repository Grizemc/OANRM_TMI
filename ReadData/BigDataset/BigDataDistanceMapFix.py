#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 16:21
# @Author  : 沈子明
# @File    : BigDataDistanceMap.py
# @Software: PyCharm
import argparse
import glob
import multiprocessing
import os
import numpy as np
import open3d


def cal_new_distance_weight1(mask_weight, mask_gt):
    true_mask_weight = mask_weight[mask_gt]
    false_mask_weight = mask_weight[~mask_gt]
    true_sort_index = np.argsort(true_mask_weight, axis=0)
    false_sort_index = np.argsort(false_mask_weight, axis=0)
    true_weight = np.arange(0.1, 1, 0.9 / true_mask_weight.shape[0]) + 1
    false_weight = np.arange(0.1, 1, 0.9 / false_mask_weight.shape[0]) + 1
    new_true_mask_weight = true_weight[true_sort_index]
    new_false_mask_weight = false_weight[false_sort_index]
    new_mask_weight = np.concatenate((new_true_mask_weight, new_false_mask_weight), axis=0)
    return new_mask_weight


def cal_new_distance_weight2(mask_weight, mask_gt):
    true_mask_weight = mask_weight[mask_gt]
    false_mask_weight = mask_weight[~mask_gt]
    true_sort_index = np.argsort(true_mask_weight, axis=0)
    false_sort_index = np.argsort(false_mask_weight, axis=0)
    true_weight = np.arange(0.1, 1, 0.9 / true_mask_weight.shape[0]) + 1
    false_weight = np.arange(0.1, 1, 0.9 / false_mask_weight.shape[0]) + 1
    new_true_mask_weight = true_weight[true_sort_index]
    new_false_mask_weight = false_weight[false_sort_index]
    new_mask_weight = np.concatenate((new_false_mask_weight, new_true_mask_weight), axis=0)
    return new_mask_weight


def cal_single_weight(file_list):
    for file in file_list:
        npz = np.load(file)
        point1 = npz["point1"]
        color1 = npz["color1"]
        ground_truth = npz["ground_truth"]
        point2 = npz["point2"]
        color2 = npz["color2"]
        mask_point1 = npz["mask_point1"]
        mask_color1 = npz["mask_color1"]
        mask_point2 = npz["mask_point2"]
        mask_color2 = npz["mask_color2"]
        mask_gt1 = npz["mask_gt1"]
        mask_gt2 = npz["mask_gt2"]
        mask_gt_pc = npz["mask_gt_pc"]
        mask_weight_1 = npz["mask_weight_1"]
        mask_weight_2 = npz["mask_weight_2"]

        mask_weight_1 = cal_new_distance_weight1(mask_weight_1, mask_gt1)
        mask_weight_2 = cal_new_distance_weight2(mask_weight_2, mask_gt2)
        np.savez_compressed(file, point1=point1, color1=color1, ground_truth=ground_truth,
                            point2=point2, color2=color2, mask_point1=mask_point1, mask_color1=mask_color1,
                            mask_point2=mask_point2, mask_color2=mask_color2, mask_gt1=mask_gt1, mask_gt2=mask_gt2,
                            mask_gt_pc=mask_gt_pc, mask_weight_1=mask_weight_1, mask_weight_2=mask_weight_2)
    print("One thread is ok !")


if __name__ == "__main__":
    path = r"/big_data/szm/H50000amlyn_mask_mutual/test"
    # path = r"/big_data/szm/M50000ICCAI_mask_mutual/train/temp_local"
    file_list = glob.glob(os.path.join(path, "*.npz"))
    all_len = len(file_list)
    temp_len = all_len // 80
    file_group_list = []
    last_number = 0
    for i in range(0, all_len, temp_len):
        if i != 0:
            temp_list = file_list[last_number:i]
            file_group_list.append(temp_list)
        last_number = i
    if last_number != all_len:
        temp_list = file_list[last_number:all_len]
        file_group_list.append(temp_list)
    # cal_single_weight(file_list)
    pool = multiprocessing.Pool(20)
    pool.map(cal_single_weight, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
