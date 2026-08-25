#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/3 21:44
# @Author  : 沈子明
# @File    : FPFHeval.py
# @Software: PyCharm
import glob
import multiprocessing
import os
import open3d as o3d
import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np


if __name__ == "__main__":
    files = glob.glob(os.path.join(r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/test", "*.npz"))
    files.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
    files.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
    files.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))

    fpfh_files = glob.glob(os.path.join(r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/test/fpfhfitness", "*.npz"))
    fpfh_files.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
    fpfh_files.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
    fpfh_files.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))
    mask_acc1_sum = []
    mask_acc2_sum = []
    displace_error_sum = []
    for file, fpfh_file in zip(files, fpfh_files):
        with np.load(file) as fp:
            points1 = fp["points1"].astype('float32')
            colors1 = fp["colors1"].astype('float32') / 255
            points2 = fp["points2"].astype('float32')
            colors2 = fp["colors2"].astype('float32') / 255
            label_xyz1 = fp["label_xyz1"].astype('float32')
            label_xyz2 = fp["label_xyz2"].astype('float32')
            label_color1 = fp["label_color1"].astype('float32') / 255
            label_color2 = fp["label_color2"].astype('float32') / 255
        with np.load(fpfh_file) as npz:
            distance_thresholds = npz['distance_thresholds']
            matches_list0 = npz['matches_list0']
            color_errors_list = npz['color_errors_list']
            inlier_rmse_list = npz['inlier_rmse_list']
            fitness_list = npz['fitness_list']
        label_num = label_xyz2.shape[0]
        mask_acc1 = 0
        mask_acc2 = 0
        displace_error = []
        for i in range(label_num):
            target_index1 = points1.shape[0] - label_num + i
            result_index1 = np.where(matches_list0[:, 0] == target_index1)[0]
            target_index2 = points2.shape[0] - label_num + i
            result_index2 = np.where(matches_list0[:, 1] == target_index2)[0]
            if result_index1.size != 0:
                mask_acc1 += 1
            if result_index2.size != 0:
                mask_acc2 += 1
            if result_index1.size != 0 and result_index2.size != 0:
                pred = points2[matches_list0[result_index1, 1]]
                pcd2 = points2[matches_list0[result_index2, 1]]
                displace_error.append(np.sqrt(np.sum((pred - pcd2) ** 2)))
        mask_acc1 = mask_acc1 / label_num * 100
        mask_acc2 = mask_acc2 / label_num * 100
        displace_error = np.array(displace_error)
        mask_acc1_sum.append(mask_acc1)
        mask_acc2_sum.append(mask_acc2)
        if len(displace_error) != 0:
            displace_error_sum.append(displace_error.mean())

    print("mask_acc1_sum is {}".format(np.array(mask_acc1_sum).mean()))
    print("mask_acc2_sum is {}".format(np.array(mask_acc2_sum).mean()))
    print("displace_error_sum is {}".format(np.array(displace_error_sum).mean()))
