#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 11:04
# @Author  : 沈子明
# @File    : CalTanTaiBiaoZhu.py
# @Software: PyCharm
import glob
import multiprocessing
import os.path
import datetime

import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from AVisShuJia.VisUtil import ShowTanTaiLabelPcd

if __name__ == "__main__":
    root_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Source_Flow_softmax_topkpoint_topmask_8192\fpfh_LianXu_fitness_laterhavePeduso"
    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    for file in npz_results:
        with np.load(file) as npz:
            point1 = npz["points1"][0, ::]
            point2 = npz["points2"][0, ::]
            color1 = npz["colors1"][0, ::]
            color2 = npz["colors2"][0, ::]
            pred_xyz = npz["pred_xyz"][0, ::]
            pred_mask1 = npz["pred_mask1"].squeeze()
            mask_sum = npz["mask_sum"][0]
        cal_error_num = np.arange(len(pred_mask1) - 1, -1, -1)
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

        # pcd2 =  o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(point2))
        # pcd2.colors = o3d.pybind.utility.Vector3dVector(color2)
        # ransac_pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(ratation_point1))
        # o3d.visualization.draw_geometries([pcd2, ransac_pred_pcd], mesh_show_wireframe=False)
        # 使用Flow剔除
        pred_mask1 = pred_mask1 > 0.99
        flow = pred_xyz - ratation_point1
        flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
        flow_true_mean = flow_true.mean(axis=0)
        dot_products = np.dot(flow_true, flow_true_mean)  # 计算向量的点积
        cosine_angles = dot_products / (np.linalg.norm(flow_true, axis=1) * np.linalg.norm(flow_true_mean))  # 计算余弦夹角
        angles_radians = np.degrees(np.arccos(cosine_angles))  # 计算角度（弧度）
        new_true = angles_radians < 90
        true_true_point = ratation_point1[pred_mask1][new_true]
        true_true_color = color1[pred_mask1][new_true]
        true_true_flow = flow[pred_mask1][new_true]
        false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]), axis=0)
        false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
        # 使用Flow插值
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
        distances, nearest_indices = nbrs.kneighbors(false_point)
        neigh_flow = true_true_flow[nearest_indices]
        result_flow = neigh_flow
        weight = 1 / distances
        result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
        result_point = false_point + result_flow.sum(axis=1)
        new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1][new_true]), axis=0)
        new_color = np.concatenate((false_color, true_true_color), axis=0)
        # new_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(new_pred_point))
        # new_pcd.colors = o3d.pybind.utility.Vector3dVector(new_color)
        # o3d.visualization.draw_geometries([pcd2, new_pcd], mesh_show_wireframe=False)
        false_cal_error_num = np.concatenate((cal_error_num[~pred_mask1], cal_error_num[pred_mask1][~new_true]), axis=0)
        new_cal_error_num = np.concatenate((false_cal_error_num, cal_error_num[pred_mask1][new_true]), axis=0)

        new_sort_index =np.argsort(new_cal_error_num)

        sort_new_color = new_color[new_sort_index]
        sort_pred_point = new_pred_point[new_sort_index]
        label_color1 = sort_new_color[:mask_sum, :][::-1, :]
        color11 = color1[-mask_sum:, :]

        label_xyz1 = new_pred_point[new_sort_index][:mask_sum, :][::-1, :]
        label_xyz2 = point2[-mask_sum:, :]
        ShowTanTaiLabelPcd(new_pred_point, new_color, label_xyz1, np.ones_like(label_xyz1))

        ShowTanTaiLabelPcd(pred_xyz, color1, pred_xyz[-mask_sum:, :], np.ones_like(pred_xyz[-mask_sum:, :]))
        ShowTanTaiLabelPcd(point2, color2, label_xyz2, np.ones_like(label_xyz2))
