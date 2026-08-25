#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 15:27
# @Author  : 沈子明
# @File    : VisScreen.py
# @Software: PyCharm
import glob
import os

import numpy as np
import open3d as o3d
from AVisShuJia.VisUtil import ShowTanTaiLabelPcd


if __name__ =="__main__":
    result_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\GaussianInter"
    npz_path = r"D:\ProgramofSZM\remote\PAConv\checkpoints\Zall\fpfh_Post_Train_Hamlyn_no_rotation\npz_result"
    result_files = glob.glob(os.path.join(result_path, "*.npz"))
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    result_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    npz_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))

    # 创建一个窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    len_npz = len(npz_files)
    for index in range(0, len_npz, 5):
        npz_file = npz_files[index]
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
        mask_pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point1))
        mask_pcd1.colors = o3d.pybind.utility.Vector3dVector(mask_color1)
        mask_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point2))
        mask_pcd2.colors = o3d.pybind.utility.Vector3dVector(mask_color2)

        # 将点云添加到可视化窗口
        vis.add_geometry(mask_pcd1)
        vis.add_geometry(mask_pcd2)

        vis.update_geometry(mask_pcd1)
        vis.update_geometry(mask_pcd2)
        vis.poll_events()
        vis.update_renderer()

        # 截图
        vis.capture_screen_image("mask_pictures/ npz_file_{}.png".format(npz_file.split("\\")[-1].split(".")[0]))
        # 移除当前点云，准备加载下一个
        vis.clear_geometries()
    # 关闭可视化窗口
    vis.destroy_window()
