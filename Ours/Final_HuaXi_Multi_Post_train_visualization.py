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
    color_pred_mask2 = np.copy(color2)
    color_pred_mask2[pred_mask2_bool, 0] = 255 / 255  # 255/255  # 0  # 绿色
    color_pred_mask2[pred_mask2_bool, 1] = 140 / 255  # 192/255  # 0.9
    color_pred_mask2[pred_mask2_bool, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask2[~pred_mask2_bool, 0] = 171 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask2[~pred_mask2_bool, 1] = 27 / 255  # 0/255
    color_pred_mask2[~pred_mask2_bool, 2] = 4 / 255  # 212/255

    pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask2)
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
    # ratation_point1 = ratation_point1[~pred_mask1_bool]
    # color_pred_mask_rata = np.copy(color1)
    # color_pred_mask_rata = color_pred_mask_rata[~pred_mask1_bool]
    # color_pred_mask_rata[:, 0] = 0
    # color_pred_mask_rata[:, 1] = 0
    # color_pred_mask_rata[:, 2] = 1
    # ratation_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(ratation_point1))
    # ratation_pcd.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_rata)

    every_k_points = 8
    # ratation_pcd_downsampled = ratation_pcd.uniform_down_sample(every_k_points)
    pred_pcd_downsampled = pred_pcd.uniform_down_sample(every_k_points)
    pcd2_downsampled = pcd2.uniform_down_sample(every_k_points)

    # o3d.visualization.draw_geometries([ratation_pcd_downsampled , pred_pcd_downsampled,pcd2_downsampled], mesh_show_wireframe=False)


    pred_mask1 = pred_mask1 > 0.8
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
    false_color_num = false_color.shape[0]

    # new_color[0:false_color_num, :] = np.array([0, 0, 255 / 255])  # np.array([0,0,255])
    # new_color[false_color_num:, :] = np.array([0, 255 / 255, 0])  # np.array([255,255,255])

    pcd_pred.colors = o3d.pybind.utility.Vector3dVector(new_color)

    radius = 1.0
    every_k_points = 7
    pcd_pred_downsampled = pcd_pred.uniform_down_sample(every_k_points)
    pcd2_downsampled = pcd2.uniform_down_sample(every_k_points)

    o3d.visualization.draw_geometries([pcd_pred,pcd2], mesh_show_wireframe=False)
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
    mask_trun = 0.8
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Source_Flow_softmax_topkpoint_topmask_fuse_8192\fpfh_HuaXiFpfh_305"
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_HuaXiFpfh_322"
    post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua\fpfh_HuaXiFpfh_322_95"
    post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua\fpfh_95%HuaXiFpfh_200_7"
    post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua_stereo\fpfh_95%_Stereo_P2-5Fpfh1467_21"
    # post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua\huaxi_keshihua_200_mixdataset"
    # post_train = r"D:\try\try\szmCode\paconv_\duibi_keshihua\fpfh_HuaXiFpfh_322_mix"
    # post_train = r"D:\try\try\szmCode\paconv_\huaxi_keshihua_200_mix"
    # post_train = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_HuaXiFpfh_100"
    post_train_file_path = glob.glob(os.path.join(post_train, '*.npz'))
    # for file in post_train_file_path:
    file = post_train_file_path[-1]
    print("file:", file)
    npz = np.load(file)
    point1 = npz['points1'].squeeze()
    point2 = npz['points2'].squeeze()
    pred_xyz = npz['pred_xyz'].squeeze()
    pred_mask1 = npz['pred_mask1'].squeeze()
    pred_mask2 = npz['pred_mask2'].squeeze()
    color1 = npz['colors1'].squeeze()
    color2 = npz['colors2'].squeeze()
    # ground_truth = npz['mask_gt_pc'].squeeze()
    temp_point2 = point2.copy()
    # temp_point2[:, 0] += 80
    pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(temp_point2))
    pcd2.colors = o3d.pybind.utility.Vector3dVector(color2)
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1)))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(color1)
    # o3d.visualization.draw_geometries([pcd2, pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([pcd2], mesh_show_wireframe=False)
    pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(pred_xyz))
    pred_pcd.colors = o3d.pybind.utility.Vector3dVector(color1)
    o3d.visualization.draw_geometries([pred_pcd], mesh_show_wireframe=False)

    # o3d.visualization.draw_geometries([pred_pcd], mesh_show_wireframe=False)
                                # 173, 216, 230
    pred_mask1_bool= pred_mask1 > 0.3
    pred_mask2_bool = pred_mask2 > 0.3
    color_pred_mask = np.copy(color1)
    color_pred_mask[pred_mask1_bool, 0] = 0  # 绿色
    color_pred_mask[pred_mask1_bool, 1] = 1
    color_pred_mask[pred_mask1_bool, 2] = 0
    color_pred_mask[~pred_mask1_bool, 0] = 0  # 173/255 # 深红色
    color_pred_mask[~pred_mask1_bool, 1] = 0  # 100/255
    color_pred_mask[~pred_mask1_bool, 2] = 1  # 230/255

    color_pred_mask_rata = np.copy(color1)
    color_pred_mask_rata[pred_mask1_bool, 0] = 0  # 绿色
    color_pred_mask_rata[pred_mask1_bool, 1] = 1
    color_pred_mask_rata[pred_mask1_bool, 2] = 0
    color_pred_mask_rata[~pred_mask1_bool, 0] = 0  # 173/255 # 深红色
    color_pred_mask_rata[~pred_mask1_bool, 1] = 0  # 100/255
    color_pred_mask_rata[~pred_mask1_bool, 2] = 1  # 230/255

    # color_pred_mask = color_pred_mask[pred_mask1_bool]
    # point1 = point1[pred_mask1_bool]
    point1 = point1  # - np.array([-17,0,-10])
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1)))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    pcd1.colors = o3d.pybind.utility.Vector3dVector(color1)
    # o3d.visualization.draw_geometries([pcd1], mesh_show_wireframe=False)


    color_pred_mask2 = np.copy(color2)
    color_pred_mask2[pred_mask2_bool, 0] = 255 / 255  # 255/255  # 0  # 绿色
    color_pred_mask2[pred_mask2_bool, 1] = 140 / 255  # 192/255  # 0.9
    color_pred_mask2[pred_mask2_bool, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask2[~pred_mask2_bool, 0] = 171 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask2[~pred_mask2_bool, 1] = 27 / 255  # 0/255
    color_pred_mask2[~pred_mask2_bool, 2] = 4 / 255  # 212/255
    # color_pred_mask2 = color_pred_mask2[pred_mask2_bool]
    # point2 = point2[pred_mask2_bool]
    pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point2)))
    pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask2)
    pcd2.colors = o3d.pybind.utility.Vector3dVector(color2)

    radius = 1.0
    every_k_points =3
    pcd_downsampled_pcd1 = pcd1.uniform_down_sample(every_k_points)
    pcd_downsampled_pcd2 = pcd2.uniform_down_sample(every_k_points)

    # o3d.visualization.draw_geometries([pcd_downsampled_pcd2], mesh_show_wireframe=False)

    color_line = np.copy(color1)

    # pred_point = pred_xyz[pred_mask1_bool]
    pred_color = color1[pred_mask1_bool]
    point1_overlap = point1[pred_mask1_bool]
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((point1)))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)

    pred_points = pred_xyz # [pred_mask1_bool]
    # pred_points = pred_xyz[pred_mask1_bool]
    # pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_xyz)))
    pred_mask1_bool= pred_mask1 > 0.3
    pred_point = pred_xyz[pred_mask1_bool]
    pred_point = pred_point - np.array([-5,2,-10])
    # pred_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))

    color_pred_mask = np.copy(color1)
    pred_color = color_pred_mask[pred_mask1_bool]
    color_pred_mask[pred_mask1_bool, 0] = 0  # 绿色
    color_pred_mask[pred_mask1_bool, 1] = 1
    color_pred_mask[pred_mask1_bool, 2] = 0
    color_pred_mask[~pred_mask1_bool, 0] = 0  # 173/255 # 深红色
    color_pred_mask[~pred_mask1_bool, 1] = 0 # 100/255
    color_pred_mask[~pred_mask1_bool, 2] = 1  # 230/255

    pred_color[:, 0] = 0  # 绿色
    pred_color[:, 1] = 1
    pred_color[:, 2] = 0
    # pred_pcd.colors = o3d.pybind.utility.Vector3dVector(color1)
    # pred_pcd.colors = o3d.pybind.utility.Vector3dVector(pred_color)

    every_k_points = 7
    pred_pcd_downsampled = pred_pcd.uniform_down_sample(every_k_points)
    # pcd_downsampled_pcd1 = pcd1.uniform_down_sample(every_k_points)

    # 创建LineSet对象以存储连线
    lines = []
    # 创建连线
    # for i in range(len(pred_point)):
    #     lines.append([i, i])  # 将同一索引的点连接起来
    #     lines.append([i, i + len(pred_point)])  # 将两个点云中同一索引的点连接起来
    # # 创建LineSet对象
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(np.vstack([pred_point, point1_overlap])),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )

    # o3d.visualization.draw_geometries([pred_pcd], mesh_show_wireframe=False)

    GaussianPostMostProcessHamlynResult(point1, color1, pred_xyz, point2, color2, pred_mask1, pred_mask2,
                                          mask_trun)
