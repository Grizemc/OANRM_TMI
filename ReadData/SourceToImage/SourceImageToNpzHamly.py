#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 16:42
# @Author  : 沈子明
# @File    : HamlySourceHamlyToImage.py
# @Software: PyCharm
import glob
import multiprocessing
import os.path

import numpy as np
import open3d
from PIL import Image


def remove_000(pcd):
    """
    :param pcd: 待处理的点云
    :return: 删除零000后的点云
    """
    pcd_points = np.array(pcd.points)  # 读取点云中的点
    # 生成需要删除的点的索引
    mask_source = [3]  # 初始化列表，方便处理
    temp_judge = (pcd_points != [0, 0, 0])
    for i in range(temp_judge.shape[0]):
        mask_source.append(temp_judge[i, :].all())
    mask_source.pop(0)  # 删除列表中第一个元素
    mask_source = np.array(mask_source)
    mask = np.where(mask_source == True)
    pcd_del = pcd.select_by_index(mask[0])
    return pcd_del


def imageToPC(image_path, depth_path, fx, fy, cx, cy):
    # 读取rgb图片
    image = np.array(Image.open(image_path))
    # plt.imshow(image, cmap='gray')
    # plt.show()

    # 读取深度
    Zc = np.array(Image.open(depth_path)).astype(np.float32)

    # 去除某些离群点
    invalid_mask = Zc > 180
    Zc[invalid_mask] = 0

    p1_v = np.reshape(np.linspace(0, Zc.shape[0] - 1, Zc.shape[0]), (-1, 1)).repeat(axis=1, repeats=Zc.shape[1])
    p1_u = np.reshape(np.linspace(0, Zc.shape[1] - 1, Zc.shape[1]), (1, -1)).repeat(axis=0, repeats=Zc.shape[0])
    xc = Zc * (p1_u - cx) / fx
    yc = Zc * (p1_v - cy) / fy

    xc = np.expand_dims(xc, axis=2)
    yc = np.expand_dims(yc, axis=2)
    Zc = np.expand_dims(Zc, axis=2)
    xyz = np.concatenate((xc, yc, Zc), axis=2)
    xyz = np.reshape(xyz, (-1, 3)).astype(np.float32)
    rgb = np.reshape(image, (-1, 3)).astype(np.float32)
    rgb = rgb / 255
    return xyz, rgb


def SourceToImageSingle(file_paths):
    # target_path = "/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSourceNpz_91864"
    target_path = "/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSourceNpz_91864"
    # target_path = r"/big_data/szm/cwz_stereomis_P2-5_new/"
        
    for image_path in file_paths:
        file_num = str(image_path.split(".")[0].split("0")[-1]).rjust(10, '0')
        depth_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), "depth01", file_num + ".png")
        store_path = os.path.join(target_path, image_path.split("/")[-3] + "_" + str(
            int(image_path.split(".")[0].split("/")[-1])) + ".npz")
        if os.path.exists(image_path) and os.path.exists(depth_path) and not os.path.exists(store_path) :
            intrinsics_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), "intrinsics.txt")
            # 内参读取
            intrinsics = np.loadtxt(intrinsics_path)
            left_intrinsics = intrinsics[:, :3].astype(np.float32)
            fx = left_intrinsics[0, 0]  # 内参
            fy = left_intrinsics[1, 1]
            cx = left_intrinsics[0, 2]
            cy = left_intrinsics[1, 2]
            xyz, rgb = imageToPC(image_path, depth_path, fx, fy, cx, cy)

            np.savez_compressed(store_path, xyz=xyz, rgb=rgb)
    print("One thread success")


if __name__ == "__main__":
    all_split_num = 60
    pool_num = 30
    # root_path = r"/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSource"
    root_path = r"/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSource"

    image01_paths = []
    for folder_path in glob.glob(os.path.join(root_path, "rectified*")):
        image01_paths.extend(glob.glob(os.path.join(root_path, folder_path, "image01/", "*.jpg")))
    all_len = len(image01_paths)
    temp_len = all_len // all_split_num
    file_group_list = []
    last_number = 0
    for i in range(0, all_len, temp_len):
        if i != 0:
            temp_list = image01_paths[last_number:i]
            file_group_list.append(temp_list)
        last_number = i
    if last_number != all_len:
        temp_list = image01_paths[last_number:all_len]
        file_group_list.append(temp_list)
    # SourceToImageSingle(temp_list)

    multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    pool = multiprocessing.Pool(pool_num)
    pool.map(SourceToImageSingle, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
