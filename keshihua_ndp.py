#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 11:26
# @Author  : 沈子明
# @File    : Postprocessing.py
# @Software: PyCharm
import glob
import os
import time

import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np


def GaussianPostMostProcessHamlynResult(point1, color1, pred_xyz, point2, color2, pred_mask1, pred_mask2,
                                              mask_trun):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

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

    pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(point2))
    pcd2.colors = o3d.pybind.utility.Vector3dVector(color2)
    pred_xyz = GaussianFilter(point1, pred_xyz, 8, 5)
    # 使用ransac算法进行整体位移变换
    ransac_pred_mask1 = pred_mask1 > mask_trun
    ransac_source_cloud = o3d.geometry.PointCloud(
        o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
    ransac_pred_pcd = o3d.geometry.PointCloud(
        o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
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
    ratation_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(ratation_point1))
    o3d.visualization.draw_geometries([pcd2, ratation_pcd], mesh_show_wireframe=False)
    pred_mask1 = pred_mask1 > 0.9
    flow = pred_xyz - ratation_point1
    flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
    true_point = ratation_point1[pred_mask1]
    true_color = color1[pred_mask1]
    true_flow = flow[pred_mask1]
    false_point = ratation_point1[~pred_mask1]
    false_color = color1[~pred_mask1]

    # 使用Flow插值
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_point)
    distances, nearest_indices = nbrs.kneighbors(false_point)
    neigh_flow = true_flow[nearest_indices]
    result_flow = neigh_flow
    weight = 1 / distances
    result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
    result_point = false_point + result_flow.sum(axis=1)
    new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1]), axis=0)
    new_color = np.concatenate((false_color, true_color), axis=0)
    pcd_pred = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((new_pred_point)))
    pcd_pred.colors = o3d.pybind.utility.Vector3dVector(new_color)
    o3d.visualization.draw_geometries([pcd_pred], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([pcd2, pcd_pred], mesh_show_wireframe=False)
    post_train = True
    # np.savez(save_path,
    #          new_pred_point=new_pred_point,
    #          new_color=new_color,
    #          new_mask_gt_pc=new_mask_gt_pc,
    #          point2=point2,
    #          color2=color2,
    #          post_train=post_train
    #          )
    # o3d.visualization.draw_geometries([pcd2], mesh_show_wireframe=False)


if __name__ == "__main__":
    mask_trun = 0.9
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Source_Flow_softmax_topkpoint_topmask_fuse_8192\fpfh_HuaXiFpfh_305"
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_HuaXiFpfh_322"
    post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua\huaxi_keshihua_ndp"
    # post_train = r"D:\try\try\szmCode\paconv_\huaxi_keshihua_200_mix"
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_HuaXiFpfh_100"
    post_train_file_path = glob.glob(os.path.join(post_train, '*.npz'))
    # for file in post_train_file_path:
    file = post_train_file_path[-1]
    print("file:", file)
    npz = np.load(file)
    point1 = npz['points1'].squeeze()
    point2 = npz['points2'].squeeze()
    # pred_xyz = npz['pred_xyz'].squeeze()
    pred_xyz = npz['pred_point'].squeeze()

    # pred_mask1 = npz['pred_mask1'].squeeze()
    # pred_mask2 = npz['pred_mask2'].squeeze()
    color1 = npz['color1'].squeeze()
    color2 = npz['color2'].squeeze()
    temp_point2 = point2.copy()
    # temp_point2[:, 0] += 80
    pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(temp_point2))
    pcd2.colors = o3d.pybind.utility.Vector3dVector(color2)
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1)))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(color1)
    pre_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_xyz)))
    pre_pcd.colors = o3d.pybind.utility.Vector3dVector(color1)
    # o3d.visualization.draw_geometries([pcd2, pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([pcd2], mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([pre_pcd,pcd2], mesh_show_wireframe=False)


    # pred_mask1_bool= pred_mask1 > 0.7
    # color_pred_mask = np.copy(color1)
    # color_pred_mask[pred_mask1_bool, 0] = 0  # 绿色
    # color_pred_mask[pred_mask1_bool, 1] = 1
    # color_pred_mask[pred_mask1_bool, 2] = 0
    # color_pred_mask[~pred_mask1_bool, 0] = 1  # 深红色
    # color_pred_mask[~pred_mask1_bool, 1] = 0
    # color_pred_mask[~pred_mask1_bool, 2] = 0
    # pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    # o3d.visualization.draw_geometries([pcd1], mesh_show_wireframe=False)
    #
    # color_line = np.copy(color1)
    #
    # pred_point = pred_xyz[pred_mask1_bool]
    # pred_color = color1[pred_mask1_bool]
    # point1_overlap = point1[pred_mask1_bool]
    # pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1)))
    # pcd1.colors = o3d.pybind.utility.Vector3dVector(color1)
    #
    # pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    # pred_color[:, 0] = 0  # 绿色
    # pred_color[:, 1] = 1
    # pred_color[:, 2] = 0
    # pred_pcd.colors = o3d.pybind.utility.Vector3dVector(pred_color)
    # color_pred_mask = np.copy(color1)
    # # 创建LineSet对象以存储连线
    # lines = []
    # # 创建连线
    # for i in range(len(pred_point)):
    #     lines.append([i, i])  # 将同一索引的点连接起来
    #     lines.append([i, i + len(pred_point)])  # 将两个点云中同一索引的点连接起来
    # # 创建LineSet对象
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(np.vstack([pred_point, point1_overlap])),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )
    # o3d.visualization.draw_geometries([pcd1, pred_pcd, line_set], mesh_show_wireframe=False)
    #
    # GaussianPostMostProcessHamlynResult(point1, color1, pred_xyz, point2, color2, pred_mask1, pred_mask2,
    #                                           mask_trun)
