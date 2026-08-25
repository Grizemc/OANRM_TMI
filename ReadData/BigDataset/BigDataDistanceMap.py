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


def cal_distance_weight1(mask_point, mask_gt):
    true_point = mask_point[mask_gt]
    false_point = mask_point[~mask_gt]
    l1_true_distance = ((np.expand_dims(true_point, axis=1) - np.expand_dims(false_point, axis=0)) ** 2).sum(
        axis=-1).min(axis=1)
    l1_false_distance = ((np.expand_dims(false_point, axis=1) - np.expand_dims(true_point, axis=0)) ** 2).sum(
        axis=-1).min(axis=1)
    l1_true_quan = 1 / (l1_true_distance + 1e-5)
    l1_true_quan = l1_true_quan / l1_true_quan.sum(axis=0)
    l1_false_quan = 1 / (l1_false_distance + 1e-5)
    l1_false_quan = l1_false_quan / l1_false_quan.sum(axis=0)

    mask_point = np.concatenate((true_point, false_point), axis=0)
    weight = np.concatenate((np.expand_dims(l1_true_quan, axis=1), np.expand_dims(l1_false_quan, axis=1)), axis=0)
    return mask_point, weight


def cal_distance_weight2(mask_point, mask_gt):
    true_point = mask_point[mask_gt]
    false_point = mask_point[~mask_gt]
    l1_true_distance = ((np.expand_dims(true_point, axis=1) - np.expand_dims(false_point, axis=0)) ** 2).sum(
        axis=-1).min(axis=1)
    l1_false_distance = ((np.expand_dims(false_point, axis=1) - np.expand_dims(true_point, axis=0)) ** 2).sum(
        axis=-1).min(axis=1)
    l1_true_quan = 1 / (l1_true_distance + 1e-5)
    l1_true_quan = l1_true_quan / l1_true_quan.sum(axis=0)
    l1_false_quan = 1 / (l1_false_distance + 1e-5)
    l1_false_quan = l1_false_quan / l1_false_quan.sum(axis=0)

    mask_point = np.concatenate((false_point, true_point), axis=0)
    weight = np.concatenate((np.expand_dims(l1_false_quan, axis=1), np.expand_dims(l1_true_quan, axis=1)), axis=0)
    return mask_point, weight


def cal_single_weight(file_list):
    for file in file_list:
        npz = np.load(file)
        if len(npz.files) == 12:
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
            mask_point11, mask_weight_1 = cal_distance_weight1(mask_point1, mask_gt1)
            mask_point22, mask_weight_2 = cal_distance_weight2(mask_point2, mask_gt2)

            np.savez_compressed(file, point1=point1, color1=color1, ground_truth=ground_truth,
                                point2=point2, color2=color2, mask_point1=mask_point1, mask_color1=mask_color1,
                                mask_point2=mask_point2, mask_color2=mask_color2, mask_gt1=mask_gt1, mask_gt2=mask_gt2,
                                mask_gt_pc=mask_gt_pc, mask_weight_1=mask_weight_1, mask_weight_2=mask_weight_2)
            print("{} is success".format(file))
    print("One thread is ok !")


if __name__ == "__main__":
    root_path=  r"/big_data/szm/H50000amlyn_mask_mutual"
    path = r"/big_data/szm/H50000amlyn_mask_mutual/{}".format("test")
    all_file_list = glob.glob(os.path.join(path, "*.npz"))
    file_list = []
    for file in all_file_list:
        try:
            with np.load(file) as npz:
                if len(npz.files) == 12:
                    file_list.append(file)

        except:
            os.system("mv {} {}".format(file, os.path.join(root_path + "/no_npz",
                                                           file.split("/")[-1])))
    all_len = len(file_list)
    temp_len = all_len // 90
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
        #
    pool = multiprocessing.Pool(30)
    pool.map(cal_single_weight, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
