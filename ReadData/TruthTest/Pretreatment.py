#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/9 9:49
# @Author  : 沈子明
# @File    : ReadTiffCoordinate.py
# @Software: PyCharm
"""
插值出人工标记点的颜色，澹台师姐的数据集是不带颜色的
"""
# 插值出人工标记点的颜色
import glob
import os.path
import numpy as np
import open3d
import torch
from lib.pointops.functions import pointops

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# -----------------------------------------------------------------------------
# print the str on the screen and store them in the log file
# -----------------------------------------------------------------------------
class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        print("=================================")
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def get_subdirectories(folder_path):
    subdirectories = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories


def InterpolatedColors(point, color, label):
    # Interpolated colors of label points
    py_labels = torch.from_numpy(label).unsqueeze(0).cuda()
    py_point = torch.from_numpy(point).unsqueeze(0).cuda()
    py_color = torch.from_numpy(color).unsqueeze(0).transpose(1, 2).contiguous().cuda()
    dist, idx = pointops.nearestneighbor(py_labels, py_point)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_colors = pointops.interpolation(py_color, idx, weight)
    interpolated_colors = np.array(interpolated_colors.cpu().transpose(1, 2).squeeze())
    return interpolated_colors


def filter_pcd(files):
    # 　remove the noise points
    target_path = r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/test"
    for file in files:
        fp = np.load(file)
        points1 = fp["points1"].astype('float32')
        points2 = fp["points2"].astype('float32')
        colors2 = fp["colors2"].astype('float32')
        colors1 = fp["colors1"].astype('float32')
        label_xyz1 = fp["label_xyz1"].astype('float32')
        label_xyz2 = fp["label_xyz2"].astype('float32')
        label_color1 = fp["label_color1"].astype('float32')
        label_color2 = fp["label_color2"].astype('float32')
        point_cloud1 = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(points1))
        point_cloud1.colors = open3d.pybind.utility.Vector3dVector(colors1)
        point_cloud2 = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(points2))
        point_cloud2.colors = open3d.pybind.utility.Vector3dVector(colors2)
        pcd_remove1 = open3d.geometry.PointCloud.remove_statistical_outlier(point_cloud1, nb_neighbors=20, std_ratio=2)[
            0]
        pcd_remove2 = open3d.geometry.PointCloud.remove_statistical_outlier(point_cloud2, nb_neighbors=20, std_ratio=2)[
            0]
        points1 = np.array(pcd_remove1.points)
        colors1 = np.array(pcd_remove1.colors)
        points2 = np.array(pcd_remove2.points)
        colors2 = np.array(pcd_remove2.colors)
        store_path = os.path.join(target_path, file.split("/")[-1])
        np.savez(store_path, points1=points1,
                 points2=points2,
                 colors1=colors1,
                 colors2=colors2,
                 label_xyz1=label_xyz1,
                 label_xyz2=label_xyz2,
                 label_color1=label_color1,
                 label_color2=label_color2)


if __name__ == "__main__":
    log = IOStream(r"/big_data/szm/TanTaiBiaoZhu/szmNpzlog.txt")
    target_path = r"/big_data/szm/TanTaiBiaoZhu/szmNpz"
    source_path = r"/big_data/szm/TanTaiBiaoZhu/TanTaiNpz"
    for dataset_path in ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7']:
        for keyframe_path in get_subdirectories(os.path.join(source_path, dataset_path)):
            keyframe_name = keyframe_path.split('/')[-1]
            for source_npz_path in glob.glob(os.path.join(keyframe_path, "*.npz")):
                single_name = source_npz_path.split('.')[0].split('/')[-1]
                source_npz = np.load(source_npz_path)
                points1 = source_npz['points1'].astype('float32')
                points2 = source_npz['points2'].astype('float32')
                colors1 = source_npz['colors1'].astype('float32')
                colors2 = source_npz['colors2'].astype('float32')
                label1 = source_npz['label1'].astype('float32')
                label2 = source_npz['label2'].astype('float32')
                label_color1 = InterpolatedColors(points1, colors1, label1)
                label_color2 = InterpolatedColors(points2, colors2, label2)
                save_path = os.path.join(target_path,
                                         dataset_path + "_" + keyframe_name + "_label" + single_name + ".npz")
                np.savez(save_path, points1=points1,
                         points2=points2,
                         colors1=colors1,
                         colors2=colors2,
                         label_xyz1=label1,
                         label_xyz2=label2,
                         label_color1=label_color1,
                         label_color2=label_color2)
        print("{}_ {} is  finished".format(dataset_path, keyframe_name))
