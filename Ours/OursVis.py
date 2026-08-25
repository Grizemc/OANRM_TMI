#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 16:56
# @Author  : 沈子明
# @File    : NDPVis.py
# @Software: PyCharm

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import sys
sys.path.append(r'D:\try\try\szmCode\paconv_\Ours\VisUtil.py')
from VisUtil import ShowTanTaiLabelPcd, ShowTanTaiLabelPcdTargetAndPred

def Xyz_restore(xyz_in_all, relax_proportion_all):
    xyz_in = xyz_in_all
    relax_proportion = relax_proportion_all
    len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
    new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
    new_z = (z + 0.5) * len_z + z_min
    temp = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return temp

def  OursShowHamlyn_rata(sample_data_number):
    """
    可视化最终的配准结果
    :param sample_data_number:
    :return:
    """
    ratation_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours_ratation"
    npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours\npz"
    ratation_files = glob.glob(os.path.join(ratation_path, "*.npz"))
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    ratation_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    ratation_files = ratation_files[sample_data_number]
    npz_file = npz_files[sample_data_number]
    result = np.load(ratation_files)
    npz = np.load(npz_file)
    ratation_point = result["ratation_point"]
    ratation_color = result["ratation_color"]
    # new_mask_gt_pc = result["new_mask_gt_pc"]
    post_train = result["post_train"]
    color1 = npz["colors1"][0, ::]
    color2 = npz["colors2"][0, ::]
    points1 = npz["points1"][0, ::]
    points2 = npz["points2"][0, ::]
    pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    pred_mask2 = npz["pred_mask2"].squeeze() > 0.9
    pred_point = npz["pred_xyz"][0, ::]

    # print("overlap percentage is {}%.".format(mask_gt2.sum()/mask_gt2.shape[0] * 100))
    # ground_truth_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_mask_gt_pc))
    # ground_truth_pcd.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)
    pred_mask1_bool = pred_mask1
    pred_mask2_bool = pred_mask2
    color_pred_mask = np.copy(color1)
    color_pred_mask1 = np.copy(color1)

    # 带预测的非重叠结果
    pred_point = pred_point # [pred_mask1_bool]
    ratation_point = ratation_point[~pred_mask1_bool]
    pred_color = color1[~pred_mask1_bool]
    ratation_color = ratation_color[~pred_mask1_bool]

    # pred_point = pred_point[pred_mask1_bool]
    # ratation_point = ratation_point[~pred_mask1_bool]
    # pred_color = color1[pred_mask1_bool]
    # ratation_color = ratation_color[~pred_mask1_bool]

    # pred_color[:, 0] = 0  # 绿色
    # pred_color[:, 1] = 1
    # pred_color[:, 2] = 0
    # ratation_color[:, 0] = 0  # 绿色
    # ratation_color[:, 1] = 0
    # ratation_color[:, 2] = 1

    # color_pred_mask = pred_color
    color_pred_mask[pred_mask1_bool, 0] = 0 / 255  # 255/255  # 0  # 绿色
    color_pred_mask[pred_mask1_bool, 1] = 255 / 255  # 192/255  # 0.9
    color_pred_mask[pred_mask1_bool, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask[~pred_mask1_bool, 0] = 0 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask[~pred_mask1_bool, 1] = 0 / 255  # 0/255
    color_pred_mask[~pred_mask1_bool, 2] = 255 / 255  # 212/255
    #
    pred_color = color_pred_mask[~pred_mask1_bool]

    color_pred_mask1[pred_mask2_bool, 0] = 255 / 255  # 255/255  # 0  # 绿色
    color_pred_mask1[pred_mask2_bool, 1] = 140 / 255  # 192/255  # 0.9
    color_pred_mask1[pred_mask2_bool, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask1[~pred_mask2_bool, 0] = 171 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask1[~pred_mask2_bool, 1] = 27 / 255  # 0/255
    color_pred_mask1[~pred_mask2_bool, 2] = 4 / 255  # 212/255

    source_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
    source_pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask1)
    # source_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_point1))
    # source_pcd1.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)

    # 没有后处理过的
    result_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    # pred_point = pred_point[pred_mask1_bool]
    result_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))

    result_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    # color_pred_mask_ = color_pred_mask[pred_mask1_bool]
    # result_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_)

    # result_pcd1.colors = o3d.pybind.utility.Vector3dVector(pred_color)

    ratation_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((ratation_point)))
    ratation_pcd.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    ratation_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((ratation_point)))
    # ratation_pcd.colors = o3d.pybind.utility.Vector3dVector(ratation_color)
    ratation_pcd.colors = o3d.pybind.utility.Vector3dVector(pred_color)

    every_k_points = 1
    ratation_downsampled_pcd = ratation_pcd.uniform_down_sample(every_k_points)
    every_k_points1 = 6
    result1_downsampled_pcd = result_pcd1.uniform_down_sample(every_k_points1)
    every_k_points1 = 8
    source_pcd2_downsampled_pcd = source_pcd2.uniform_down_sample(every_k_points1)
    # pred_mask1_bool = pred_mask1 > 0.6
    # pred_mask2_bool = pred_mask2 > 0.8
    # color_pred_mask = np.copy(npz_result_color1)
    # color_pred_mask[pred_mask1_bool, 0] = 1  # 绿色
    # color_pred_mask[pred_mask1_bool, 1] = 1
    # color_pred_mask[pred_mask1_bool, 2] = 1
    # color_pred_mask[~pred_mask1_bool, 0] = 0  # 深红色
    # color_pred_mask[~pred_mask1_bool, 1] = 0
    # color_pred_mask[~pred_mask1_bool, 2] = 1
    # source_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    # o3d.visualization.draw_geometries([ratation_pcd,result_pcd1], mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([ratation_downsampled_pcd,result1_downsampled_pcd], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([ratation_downsampled_pcd,result1_downsampled_pcd], mesh_show_wireframe=False)


def OursShowHamlyn(sample_data_number):
    """
    可视化最终的配准结果
    :param sample_data_number:
    :return:
    """
    result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85\ours_85"
    # npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85noise\ours_85\npz"

    npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85\ours_85\npz"
    # npz_path_yuan = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85"
    # npz_files_yuan = glob.glob(os.path.join(npz_path_yuan, "*.npz"))
    # npz_files_yuan.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    # npz_files_yuan = npz_files_yuan[sample_data_number]
    # result_yuan = np.load(npz_files_yuan)
    # npz_result_point1_yuan = result_yuan["points1"][0, ::]

    # result_path = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Post_Train_Hamlyn_no_rotation_95_datiao330/GaussianInter"
    # npz_path = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Post_Train_Hamlyn_no_rotation_95_datiao330/npz_result"
    # npz_path = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation/npz_result"
    # result_path = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation/GaussianInter"
    # result_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\GaussianInter"
    # npz_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\npz_result"
    result_files = glob.glob(os.path.join(result_path, "*.npz"))
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    result_files.sort(key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0].split('_')[-2]))
    npz_files.sort(key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0].split('_')[-2]))
    result_file = result_files[sample_data_number]
    npz_file = npz_files[sample_data_number]

    print("result_file:", result_file)
    result = np.load(result_file)
    npz = np.load(npz_file)
    pred_point = result["new_pred_point"]
    new_color = result["new_color"]

    print("new_color.shape:",new_color.shape)

    new_mask_gt_pc = result["new_mask_gt_pc"]
    points2 = result["point2"]
    color2 = result["color2"]
    points1 = result["point2"]
    color1 = result["color2"]
    post_train = result["post_train"]
    print(result_file)
    print(npz_file)
    npz_result_point1 = npz["points1"][0, ::]
    npz_result_point2 = npz["points2"][0, ::]
    npz_result_color1 = npz["colors1"][0, ::]
    npz_result_color2 = npz["colors2"][0, ::]
    npz_result_pred_xyz = npz["pred_xyz"][0, ::]
    mask_gt1 = npz["mask_gt1"]
    mask_gt2 = npz["mask_gt2"]

    print("npz_result_point1:",npz_result_point1.shape)

    npz_result_mask_gt_pc = npz["mask_gt_pc"][0]
    pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    pred_mask2 = npz["pred_mask2"].squeeze() > 0.9
    mask_gt1 = npz["mask_gt1"][0]
    mask_gt2 = npz["mask_gt2"][0]
    print("overlap percentage is {}%.".format(mask_gt2.sum()/mask_gt2.shape[0] * 100))
    ground_truth_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_mask_gt_pc))
    ground_truth_pcd.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)

    pred_mask1_bool = pred_mask1
    pred_mask2_bool = pred_mask2
    color_pred_mask___ = np.copy(color2)
    color_pred_mask___[pred_mask2_bool, 0] = 255/255        # 255/255  # 0  # 绿色
    color_pred_mask___[pred_mask2_bool, 1] = 140/255        # 192/255  # 0.9
    color_pred_mask___[pred_mask2_bool, 2] = 0/255        # 203/255 # 0.8
    color_pred_mask___[~pred_mask2_bool, 0] = 171/255          # 192/255  # 深红色  128, 0, 128
    color_pred_mask___[~pred_mask2_bool, 1] = 27/255          # 0/255
    color_pred_mask___[~pred_mask2_bool, 2] = 4/255         # 212/255

    color_pred_mask_gt1 = np.copy(color1)
    color_pred_mask_gt1[mask_gt1, 0] = 0/ 255  # 255/255  # 0  # 绿色
    color_pred_mask_gt1[mask_gt1, 1] = 255 / 255  # 192/255  # 0.9
    color_pred_mask_gt1[mask_gt1, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask_gt1[~mask_gt1, 0] = 255 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask_gt1[~mask_gt1, 1] = 0 / 255  # 0/255
    color_pred_mask_gt1[~mask_gt1, 2] = 0 / 255  # 212/255

    color_pred_mask_gt2 = np.copy(color2)
    color_pred_mask_gt2[mask_gt2, 0] = 0/ 255  # 255/255  # 0  # 绿色
    color_pred_mask_gt2[mask_gt2, 1] = 255 / 255  # 192/255  # 0.9
    color_pred_mask_gt2[mask_gt2, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask_gt2[~mask_gt2, 0] = 255 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask_gt2[~mask_gt2, 1] = 0 / 255  # 0/255
    color_pred_mask_gt2[~mask_gt2, 2] = 0 / 255  # 212/255

    source_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
    source_pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask___)
    source_pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_gt2)
    source_pcd2.colors = o3d.pybind.utility.Vector3dVector(npz_result_color2)

    source_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_point1))
    source_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask___)
    source_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_gt2)
    source_pcd1.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)

    pred_mask1_bool = pred_mask1
    pred_mask2_bool = pred_mask2
    color_pred_mask = np.copy(npz_result_color1)

    color_pred_mask[pred_mask1_bool, 0] = 0 / 255  # 255/255  # 0  # 绿色
    color_pred_mask[pred_mask1_bool, 1] = 255 / 255  # 192/255  # 0.9
    color_pred_mask[pred_mask1_bool, 2] = 0 / 255  # 203/255 # 0.8
    color_pred_mask[~pred_mask1_bool, 0] = 255 / 255  # 192/255  # 深红色  128, 0, 128
    color_pred_mask[~pred_mask1_bool, 1] = 255/ 255  # 0/255
    color_pred_mask[~pred_mask1_bool, 2] = 255 / 255  # 212/255

    color_pred_mask_ = np.copy(npz_result_color1)
    false_color = color_pred_mask_[~pred_mask1_bool]
    false_color_num = false_color.shape[0]
    true_color = color_pred_mask_[pred_mask1_bool]
    new_color_true_and_false = np.concatenate((false_color, true_color), axis=0)
    # new_color[0:false_color_num,:] = np.array([0,0,255 / 255]) # np.array([0,0,255])
    # new_color[false_color_num:,:] = np.array([0,255 / 255,0])  # np.array([255,255,255])


    mask_gt_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_mask_gt_pc))
    mask_gt_pcd.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)

    color_pred_mask_[pred_mask1_bool, 0] = 0  # 绿色
    color_pred_mask_[pred_mask1_bool, 1] = 1
    color_pred_mask_[pred_mask1_bool, 2] = 0
    color_pred_mask_[~pred_mask1_bool, 0] = 0  # 深红色
    color_pred_mask_[~pred_mask1_bool, 1] = 0
    color_pred_mask_[~pred_mask1_bool, 2] = 1

    color_pred_mask__ = np.copy(npz_result_color1)
    color_pred_mask__[pred_mask1_bool, 0] = 1  # 绿色
    color_pred_mask__[pred_mask1_bool, 1] = 1
    color_pred_mask__[pred_mask1_bool, 2] = 1
    color_pred_mask__[~pred_mask1_bool, 0] = 0  # 深红色
    color_pred_mask__[~pred_mask1_bool, 1] = 0
    color_pred_mask__[~pred_mask1_bool, 2] = 1

    npz_result_point1 = npz_result_point1 # - np.array([-1,-1.8,0]) # np.array([-10,-5,-10]) # - np.array([-0.7,0.5,0]) # error 5 - np.array([-1,-1,0]) # - np.array([-0.7,0.5,0])  # - np.array([-10,-2,-15])  # 0 - np.array([-8,-15,-15])      # -np.array([-15,15,-10])   # 4 np.array([-10,-5,-15])   # - np.array([5,0,0])  #  - np.array([-20,-15,-10])
    npz_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_point1))
    npz_pcd.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask__)
    npz_pcd.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_gt1)
    npz_pcd.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)

    # 没有后处理过的
    result_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    result_pcd1.colors = o3d.pybind.utility.Vector3dVector(npz_result_color1)
    result_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask_)

    # 后处理过的
    # 此时的点云位置和颜色  应该要对应起来，即 pred_point（new_points) 与 color_pred_mask（new_color)_对应
    result_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    result_pcd.colors = o3d.pybind.utility.Vector3dVector(new_color)

    pred_mask1_bool = pred_mask1 > 0.6
    pred_mask2_bool = pred_mask2 > 0.8
    color_pred_mask = np.copy(npz_result_color1)
    # color_pred_mask[pred_mask1_bool, 0] = 1  # 绿色
    # color_pred_mask[pred_mask1_bool, 1] = 1
    # color_pred_mask[pred_mask1_bool, 2] = 1
    # color_pred_mask[~pred_mask1_bool, 0] = 0  # 深红色
    # color_pred_mask[~pred_mask1_bool, 1] = 0
    # color_pred_mask[~pred_mask1_bool, 2] = 1
    # source_pcd1.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)

    # o3d.visualization.draw_geometries([source_pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([source_pcd2], mesh_show_wireframe=False)

    color_pred_mask = np.copy(npz_result_color2)
    color_pred_mask[pred_mask2_bool, 0] = 0  # 绿色
    color_pred_mask[pred_mask2_bool, 1] = 1
    color_pred_mask[pred_mask2_bool, 2] = 0
    color_pred_mask[~pred_mask2_bool, 0] = 0  # 深红色
    color_pred_mask[~pred_mask2_bool, 1] = 0
    color_pred_mask[~pred_mask2_bool, 2] = 1

    lines = []
    # 创建连线
    for i in range(len(npz_result_point1)):
        lines.append([i, i])  # 将同一索引的点连接起来
        lines.append([i, i + len(npz_result_point1)])  # 将两个点云中同一索引的点连接起来
    # 创建LineSet对象
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([npz_result_point1, npz_result_point2])),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # o3d.visualization.draw_geometries([result_pcd,source_pcd2,line_set], mesh_show_wireframe=False)
    # source_pcd2.colors = o3d.pybind.utility.Vector3dVector(color_pred_mask)
    # o3d.visualization.draw_geometries([ground_truth_pcd,source_pcd2], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([result_pcd], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([source_pcd2], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([source_pcd1,result_pcd1], mesh_show_wireframe=False)
    # o3d.visualization.draw_geometries([source_pcd1,source_pcd2], mesh_show_wireframe=False)

    radius = 1.0
    every_k_points = 8
    result_downsampled_pcd = result_pcd.uniform_down_sample(every_k_points)
    result1_downsampled_pcd = result_pcd1.uniform_down_sample(every_k_points)
    # source_downsampled_pcd1 = source_pcd1.uniform_down_sample(every_k_points)
    # source_downsampled_pcd2 = source_pcd2.uniform_down_sample(every_k_points)
    # downsampled_pcd, _ = source_pcd1.remove_radius_outlier(nb_points=4, radius=radius)

    # bg_color = np.array([1.0, 1.0, 1.0, 0], dtype=np.float32)  # [R, G, B, A]

    # o3d.visualization.draw(downsampled_pcd, point_size=9,bg_color=bg_color)
    # o3d.visualization.draw_geometries([result1_downsampled_pcd], window_name="Radius Downsampled Point Cloud")
    # o3d.visualization.draw_geometries([result_pcd,source_pcd2], window_name="Radius Downsampled Point Cloud")
    # point1_overlap = npz_result_point1[pred_mask1_bool]
    # pred_point = pred_point[pred_mask1_bool]

    # pred_color = npz_result_color1[pred_mask1_bool]
    # pred_color[:, 0] = 0  # 绿色
    # pred_color[:, 1] = 1
    # pred_color[:, 2] = 0
    # result_pcd.colors = o3d.pybind.utility.Vector3dVector(pred_color)

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

    # o3d.visualization.draw_geometries([source_pcd1, result_pcd], mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([source_pcd1])

    o3d.visualization.draw_geometries([source_pcd2])
    o3d.visualization.draw_geometries([source_pcd2,ground_truth_pcd])
    # o3d.visualization.draw_geometries([ground_truth_pcd, source_pcd2])
    # o3d.visualization.draw_geometries([result_pcd,source_pcd2])
    # o3d.visualization.draw_geometries([result_pcd1,source_pcd2])
    # o3d.visualization.draw_geometries([result_pcd,source_pcd2])
    # o3d.visualization.draw_geometries([source_pcd2])
    # o3d.visualization.draw_geometries([result_pcd])
    # o3d.visualization.draw_geometries([result_pcd,source_pcd2])
    # o3d.visualization.draw_geometries([source_pcd2, result_pcd])

def OursShowHamlynError(sample_data_number):
    """
    可视化误差伪彩图
    :param sample_data_number:
    :return: # duibikeshihua_hamlyn_85RT
    """
    result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85\ours_85"
    # result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours"
    npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85\ours_85\npz"
    # npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours\npz"
    # result_path = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation/GaussianInter"
    result_files = glob.glob(os.path.join(result_path, "*.npz"))
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))

    result_files.sort(key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0].split('_')[-2]))
    result_file = result_files[sample_data_number]
    result = np.load(result_file)

    npz_files.sort(key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0].split('_')[-2]))
    npz_file = npz_files[sample_data_number]
    npz = np.load(npz_file)

    pred_point = result["new_pred_point"]
    new_mask_gt_pc = result["new_mask_gt_pc"]
    npz_point = npz["pred_xyz"].squeeze()
    npz_mask_gt_pc = npz["mask_gt_pc"].squeeze()


    print("new_mask_gt_pc:",new_mask_gt_pc.shape)
    print("npz_point:",npz_point.shape)
    print("pred_point:",pred_point.shape)

    pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    pred_mask2 = npz["pred_mask2"].squeeze() > 0.9

    pred_mask1_bool = pred_mask1
    pred_mask2_bool = pred_mask2

    # npz_non_overlap = npz_point[~pred_mask1_bool]
    # npz_result_point1 = npz["points1"][0, ::]  # - np.array([-1,-1.8,0]) # 5 - np.array([-1,-1,0])  # 4 - np.array([-0.7,0.5,0])    # 0 np.array([-6,-15,-15])
    # npz_non_overlap = npz_result_point1[~pred_mask1_bool]
    # non_TDDP = npz_point
    # non_TDDP[~pred_mask1_bool] = npz_non_overlap

    # 有TDDP
    error = np.linalg.norm((new_mask_gt_pc - pred_point), axis=1)
    # 无TDDP
    error = np.linalg.norm((npz_mask_gt_pc - npz_point), axis=1)
    # error = np.linalg.norm((npz_mask_gt_pc - npz_point), axis=1)
    error = error / 5
    error[error > 1] = 1
    # 计算每个点的颜色，根据误差值的绝对值映射到颜色映射范围
    colors = plt.cm.RdYlBu_r(error)[:, :3]  # 使用RdYlBu颜色映射，红色表示较大误差，蓝色表示较小误差
    result_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    result_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    # 计算每个点的颜色，根据误差值的绝对值映射到颜色映射范围
    npz_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_point))
    npz_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    # non_TDDP_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(non_TDDP))
    # non_TDDP_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([npz_pcd])
    # o3d.visualization.draw_geometries([npz_pcd])

def OursShowHamlynError_rata(sample_data_number):
    """
    可视化误差伪彩图
    :param sample_data_number:
    :return:
    """
    result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours_ratation"
    result_path1 = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours"

    npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn\ours\npz"
    # result_path = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation/GaussianInter"
    result_files = glob.glob(os.path.join(result_path, "*.npz"))
    result_files1 = glob.glob(os.path.join(result_path1, "*.npz"))
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))

    result_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    result_file = result_files[sample_data_number]
    result = np.load(result_file)

    result_files1.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    result_file1 = result_files1[sample_data_number]
    result1 = np.load(result_file1)

    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_file = npz_files[sample_data_number]
    npz = np.load(npz_file)

    ratation_point = result["ratation_point"]
    new_mask_gt_pc = result1["new_mask_gt_pc"]
    npz_point = npz["pred_xyz"].squeeze()
    npz_mask_gt_pc = npz["mask_gt_pc"].squeeze()


    pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    pred_mask2 = npz["pred_mask2"].squeeze() > 0.9

    pred_mask1_bool = pred_mask1
    pred_mask2_bool = pred_mask2

    # npz_non_overlap = npz_point[~pred_mask1_bool]
    # npz_result_point1 = npz["points1"][0, ::]  # - np.array([-1,-1.8,0]) # 5 - np.array([-1,-1,0])  # 4 - np.array([-0.7,0.5,0])    # 0 np.array([-6,-15,-15])
    npz_non_overlap = ratation_point[~pred_mask1_bool]  # - np.array([0,0.5,0])
    non_TDDP = npz_point
    non_TDDP[~pred_mask1_bool] = npz_non_overlap

    # 有TDDP
    # error = np.linalg.norm((new_mask_gt_pc - pred_point), axis=1)
    # 无TDDP
    error = np.linalg.norm((npz_mask_gt_pc - non_TDDP), axis=1)
    # error = np.linalg.norm((npz_mask_gt_pc - npz_point), axis=1)
    error = error / 5
    error[error > 1] = 1
    # 计算每个点的颜色，根据误差值的绝对值映射到颜色映射范围
    colors = plt.cm.RdYlBu_r(error)[:, :3]  # 使用RdYlBu颜色映射，红色表示较大误差，蓝色表示较小误差
    # result_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((pred_point)))
    # result_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    # 计算每个点的颜色，根据误差值的绝对值映射到颜色映射范围
    npz_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_point))
    npz_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    non_TDDP_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(non_TDDP))
    non_TDDP_pcd.colors = o3d.pybind.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([result_pcd])
    o3d.visualization.draw_geometries([non_TDDP_pcd])

def OursShowTanTai(sample_data_number):
    # result_path = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix_bian_2/fpfh_Tantai_fitness_new1/GaussianInter"
    result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_biaozhu\ours_zhao\result"
    result_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_biaozhu\ours\result"

    result_files = glob.glob(os.path.join(result_path, "*.npz"))
    result_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    result_file = result_files[sample_data_number]
    result = np.load(result_file)
    label_xyz2 = result["label_xyz2"]
    label_xyz1 = result["label_xyz1"]
    sort_pred_point = result["sort_pred_point"]
    sort_new_color = result["sort_new_color"]
    post_train = result["post_train"]
    print("label_xyz num is {}".format(len(label_xyz2)))
    npz_path = r"D:\try\try\szmCode\paconv_\duibikeshihua_biaozhu\ours\npz"
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_file = npz_files[sample_data_number]
    print(npz_file)
    print(result_file)
    npz = np.load(npz_file)
    point1 = npz["points1"][0, ::]
    point2 = npz["points2"][0, ::]
    color1 = npz["colors1"][0, ::]
    color2 = npz["colors2"][0, ::]
    pred_xyz = npz["pred_xyz"][0, ::]
    pred_mask1 = npz["pred_mask1"].squeeze()
    eval_num = npz["mask_sum"][0]
    #
    # result_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((sort_pred_point)))
    # result_pcd.colors = o3d.pybind.utility.Vector3dVector(sort_new_color)
    # o3d.visualization.draw_geometries([result_pcd])

    # ShowTanTaiLabelPcd(point1[:-eval_num], color1[:-eval_num], point1[-eval_num:], color1[-eval_num:])
    # ShowTanTaiLabelPcd(point2[:-eval_num], color2[:-eval_num], point2[-eval_num:], color2[-eval_num:])
    ShowTanTaiLabelPcdTargetAndPred(point2[eval_num:], color2[eval_num:], point2[-eval_num:], color2[-eval_num:],
                                     sort_pred_point[:eval_num], sort_new_color[:eval_num])
    # ShowTanTaiLabelPcd(sort_pred_point[eval_num:], sort_new_color[eval_num:], sort_pred_point[:eval_num],
              #          sort_new_color[:eval_num])


def OursShowSimpleHamlyn(sample_data_number):
    """
    可视化没有经过后处理的点云非刚性配准的结果
    :param sample_data_number:
    :return:
    """
    npz_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\npz_result"
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_file = npz_files[sample_data_number]
    npz = np.load(npz_file)
    npz_result_point1 = npz["points1"][0, ::]
    npz_result_point2 = npz["points2"][0, ::]
    mask_color1 = npz["colors1"][0, ::]
    mask_color2 = npz["colors2"][0, ::]
    npz_result_pred_xyz = npz["pred_xyz"][0, ::]
    npz_result_mask_gt_pc = npz["mask_gt_pc"][0]
    npz_result_pred_mask1 = npz["pred_mask1"].squeeze()
    mask_gt1 = npz["mask_gt1"][0]
    mask_gt2 = npz["mask_gt2"][0]
    pred_mask1 = npz["pred_mask1"][0]
    pred_mask2 = npz["pred_mask2"][0]
    mask_color2[mask_gt2, 0] = 0.5  # 深绿色
    mask_color2[mask_gt2, 1] = 0.5
    mask_color2[mask_gt2, 2] = 0
    mask_color2[~mask_gt2, 0] = 0.5  # 浅绿色
    mask_color2[~mask_gt2, 1] = 1
    mask_color2[~mask_gt2, 2] = 0

    mask_color1[mask_gt1, 0] = 0.5  # 深橙色
    mask_color1[mask_gt1, 1] = 0.25
    mask_color1[mask_gt1, 2] = 0
    mask_color1[~mask_gt1, 0] = 1  # 浅橙色
    mask_color1[~mask_gt1, 1] = 0.5
    mask_color1[~mask_gt1, 2] = 0

    source_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_point2))
    source_pcd2.colors = o3d.pybind.utility.Vector3dVector(mask_color2)
    source_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(npz_result_mask_gt_pc))
    source_pcd1.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    result_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((npz_result_pred_xyz)))
    result_pcd.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    o3d.visualization.draw_geometries([result_pcd])
    # o3d.visualization.draw_geometries([source_pcd1])
    o3d.visualization.draw_geometries([source_pcd2, source_pcd1])
    o3d.visualization.draw_geometries([source_pcd2, result_pcd])


def ShowMaskResults(sample_data_number):
    """
    可视化重叠区域的真值和预测结果
    :param sample_data_number:
    :return:
    """

    npz_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\npz_result"
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_file = npz_files[sample_data_number]
    npz = np.load(npz_file)
    mask_point1 = npz["points1"][0, ::]
    mask_point2 = npz["points2"][0, ::]
    mask_color1 = npz["colors1"][0, ::]
    mask_color2 = npz["colors2"][0, ::]
    pred_xyz = npz["pred_xyz"][0, ::]
    mask_gt_pc = npz["mask_gt_pc"][0]
    mask_gt1 = npz["mask_gt1"][0]
    mask_gt2 = npz["mask_gt2"][0]
    pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    pred_mask2 = npz["pred_mask2"].squeeze() > 0.9
    print("maks1的预测准确率为：{}", (pred_mask1 == mask_gt1).sum() / mask_gt1.shape[0] * 100)
    print("maks2的预测准确率为：{}", (pred_mask2 == mask_gt2).sum() / mask_gt2.shape[0] * 100)
    # show pcd1 & pcd2
    mask_color1[mask_gt1, 0] = 0  # 绿色
    mask_color1[mask_gt1, 1] = 1
    mask_color1[mask_gt1, 2] = 0
    mask_color1[~mask_gt1, 0] = 1  # 红色
    mask_color1[~mask_gt1, 1] = 0
    mask_color1[~mask_gt1, 2] = 0
    mask_color2[mask_gt2, 0] = 0  # 绿色
    mask_color2[mask_gt2, 1] = 1
    mask_color2[mask_gt2, 2] = 0
    mask_color2[~mask_gt2, 0] = 1  # 浅绿色
    mask_color2[~mask_gt2, 1] = 0
    mask_color2[~mask_gt2, 2] = 0
    print("显示预测mask的真值")
    mask_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point1))
    mask_pcd1.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    mask_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point2))
    mask_pcd2.colors = o3d.pybind.utility.Vector3dVector(mask_color2)
    o3d.visualization.draw_geometries([mask_pcd1,], mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([mask_pcd2], mesh_show_wireframe=False)

    # show pcd1 & pcd2
    mask_color1[mask_gt1, 0] = 0.5  # 深橙色
    mask_color1[mask_gt1, 1] = 0.25
    mask_color1[mask_gt1, 2] = 0
    mask_color1[~mask_gt1, 0] = 1  # 浅橙色
    mask_color1[~mask_gt1, 1] = 0.5
    mask_color1[~mask_gt1, 2] = 0
    mask_color2[mask_gt2, 0] = 0.5  # 深绿色
    mask_color2[mask_gt2, 1] = 0.5
    mask_color2[mask_gt2, 2] = 0
    mask_color2[~mask_gt2, 0] = 0.5  # 浅绿色
    mask_color2[~mask_gt2, 1] = 1
    mask_color2[~mask_gt2, 2] = 0
    print("显示预测mask的真值")
    mask_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point1))
    mask_pcd1.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    mask_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point2))
    mask_pcd2.colors = o3d.pybind.utility.Vector3dVector(mask_color2)
    o3d.visualization.draw_geometries([mask_pcd1,], mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([mask_pcd2], mesh_show_wireframe=False)
    # source_pcd_gt = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_gt_pc))
    # source_pcd_gt.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    # o3d.visualization.draw_geometries([source_pcd_gt, mask_pcd2], mesh_show_wireframe=False)
    print("显示预测mask的预测结果")
    mask_color2[pred_mask2, 0] = 0.5  # 深绿色
    mask_color2[pred_mask2, 1] = 0.5
    mask_color2[pred_mask2, 2] = 0
    mask_color2[~pred_mask2, 0] = 0.5  # 浅绿色
    mask_color2[~pred_mask2, 1] = 1
    mask_color2[~pred_mask2, 2] = 0
    mask_pcd2.colors = o3d.pybind.utility.Vector3dVector(mask_color2)
    mask_color1[pred_mask1, 0] = 0.5  # 深橙色
    mask_color1[pred_mask1, 1] = 0.25
    mask_color1[pred_mask1, 2] = 0
    mask_color1[~pred_mask1, 0] = 1  # 浅橙色
    mask_color1[~pred_mask1, 1] = 0.5
    mask_color1[~pred_mask1, 2] = 0
    mask_pcd1.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
    o3d.visualization.draw_geometries([mask_pcd1, mask_pcd2], mesh_show_wireframe=False)


def printLowOverlap():
    """
    读取hamlyn数据集对用的索引

    """
    # result_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\GaussianInter"
    # npz_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\npz_result"
    # result_files = glob.glob(os.path.join(result_path, "*.npz"))
    # npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    # result_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    # npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    # overlap_list = []
    # for npz_file in npz_files:
    #     npz = np.load(npz_file)
    #     npz_result_point1 = npz["points1"][0, ::]
    #     npz_result_point2 = npz["points2"][0, ::]
    #     npz_result_color1 = npz["colors1"][0, ::]
    #     npz_result_color2 = npz["colors2"][0, ::]
    #     npz_result_pred_xyz = npz["pred_xyz"][0, ::]
    #     npz_result_mask_gt_pc = npz["mask_gt_pc"][0]
    #     pred_mask1 = npz["pred_mask1"].squeeze() > 0.9
    #     pred_mask2 = npz["pred_mask2"].squeeze() > 0.9
    #     mask_gt1 = npz["mask_gt1"][0]
    #     mask_gt2 = npz["mask_gt2"][0]
    #     overlap_list.append(mask_gt2.sum() / mask_gt2.shape[0])
    # overlap_array = np.array(overlap_list)
    # np.save("overlap_array_Hamlyn.npy", overlap_array)
    overlap_array = np.load("overlap_array_Hamlyn.npy")

    index = overlap_array.argsort()
    print(index[10:80])


if __name__ == "__main__":
    # 1405 1295 1231 1806 1223 1636 1789  999 1852 2752 1576 2212  266   53
    #  1334 2583  719 1159 1812 2819 1834 2488 1291 2130 2144  164 2006 1717  971  119 1851 1907  253 1214 1574 1838 1844 1627 1090  433 1055  636
    #  1928  883 1443  869 1620 2489 1671 1841  450 2600 1388 2512 1481 2288  423 1483    6 2645  876 1485 1010 2059 2305  734  879 1424 2668 2185
    #  2248   15 2460 1793 2211 1079 2162   97  736  982  970 1818  498 1354   24 1054 1578 2734 2031 2439 2068  804  244  559  213 1324 2178 2301
    #   900 1786 2390 1720 2367 2545 1930 2148  163 1675  856 2730 1242  934 2554  101 1042  714 2120 2408  330 2396 1579  685 2515 2496 1651 1577
    #  1956  845  600 2447   45 1437 2265 1896  624 1062 1795  766 1156  257
    #   646 1989 1891 1682 2256 2632 2507  118 1632 1974 2429  697 1144  729
    #  1039 1240 1621 2712 1165 1542 1663 1321  276 1085 1872 1137  716 1308
    #  2724 1686  177 1364 2358  890 1687 1967  530 2099  447 2137  141 2402
    #   898 1451  964   92  285  620  656  603 2268 2788 2169 1435  121 2565
    #  2570  915 2044 2121 2405 1361 2017 2790 2075 2188  400 2097 2252  334
    #  1648 2015  843  863 1100 2117 1943 2340 2203  674 1227 1735 1461 2407
    #  2281  780 1477 1753 1617 1626 1526 1204  240 2592 2100 1146  501 2809
    #  1933 1748 1171 1099 2387 1343 2164 1482  393  678  426  909  454 1693
    #  1667 2275  186
    # sample_data_number = 2516
    sample_data_number = 11 #
    sample_data_number = 11
    sample_data_number = 11
    sample_data_number = 11
    sample_data_number = 0  # 9  # 5 # 0 和 4
    # sample_data_number = 4 # 1520不要
    # sample_data_number = 155
    # 500 显示mask 的预测结果`
    # for sample_data_number in range(1685, 2821, 5):
    #     print("sample_data_number is {}".format(sample_data_number)
    #     )
    # ShowMaskResults(sample_data_number)
    # sample_data_number = 50
    # OursShowTanTai(sample_data_number)
    #

    # OursShowHamlynError(sample_data_number)
    OursShowHamlyn(sample_data_number)

    # OursShowSimpleHamlyn(sample_data_number)
    # printLowOverlap()  # 寻找重叠区域比较小的地方
    # OursShowHamlyn_rata(sample_data_number)
    # OursShowHamlynError_rata(sample_data_number)