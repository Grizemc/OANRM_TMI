#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 16:43
# @Author  : 沈子明
# @File    : MICCAISourceImageToNpz.py
# @Software: PyCharm
"""
将MICCAI视频和深度图像，转换成点云，并存储为Npz格式的文件。
"""
import os
import glob
import logging
import numpy as np
import cv2 as cv
import tifffile


if __name__ == "__main__":

    root_source_path = r"/big_data/szm/MICCAISource/"  # source traget, rgb.mp4 offer rgb,  scene_points folder offer xyz
    root_target_path = r"/big_data/szm/MICCAISourceNpz"  # target store folder
    for dataste_file in os.listdir(root_source_path):
        for keyframe_name in os.listdir(root_source_path + dataste_file):
            vedio_path = os.path.join(root_source_path, dataste_file, keyframe_name, "data/rgb.mp4")
            picture_path = os.path.join(root_source_path, dataste_file, keyframe_name, "data/scene_points/")
            count_file = 0
            for i, xyz_file in enumerate(glob.glob(picture_path + "*.tiff")):
                # 读取tiff文件中的点云坐标
                xyz = tifffile.imread(

                )
                xyz = xyz[0:1024, :, :].reshape(-1, 3)  # num_pints, xyz

                # 读取视频中的rgb颜色
                video = cv.VideoCapture(vedio_path)  # 读取视频，参数为路径
                ret, frame = video.read()  # frame为一帧图像，当frame为空时，ret返回false，否则为true
                if ret:  # 判断是否是最后一帧图像
                    if i == 0:
                        logging.info('视频读取成功，正在逐帧截取，颜色为RGB格式，范围为open3d的[0,1]..')
                    frame = frame[0:1024, :, :].astype('float32')
                    rgb = (frame.reshape(-1, 3)[:, [2, 1, 0]] / 255.0)

                    save_path = os.path.join(root_target_path,
                                             "{}_{}_{}.npz".format(dataste_file, keyframe_name, i))  # 创建目标文件夹
                    np.savez_compressed(save_path, xyz=xyz, rgb=rgb)
            print("已保存{}_{}".format(dataste_file, keyframe_name))


