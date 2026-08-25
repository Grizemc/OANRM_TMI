#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 16:34
# @Author  : 沈子明
# @File    : ReadSourceHamly.py
# @Software: PyCharm

"""
专门读取数据集的npz文件。
"""
import glob
import numpy as np
import open3d


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

def xyz_restore(xyz_in, relax_proportion):
    len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
    new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
    new_z = (z + 0.5) * len_z + z_min
    result = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return result

if __name__ == "__main__":
    # path = r"/big_data/szm/Hamlyn/test/"
    path = r"/big_data/szm/PostTrain/"
    # path = r"/big_data/szm/HamlyAll/"
    # path = r"/big_data/szm/temp60000Miccai/test"
    file1 = r"/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSourceNpz_91864/rectified14_911.npz"
    file2 = r"/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual/train/dataset2_keyframe_4_916.npz"
    npz1 = np.load(file1)
    npz2 = np.load(file2)

    for file in glob.glob(path + "/*.npz"):
        with np.load(file) as npz:
            point1 = npz["point1"]
            color1 = npz["color1"]
            ground_truth = npz["ground_truth"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            Nor_point1 = npz["Nor_point1"]
            Nor_point2 = npz["Nor_point2"]
            Nor_ground_truth = npz["Nor_ground_truth"]
            point1_ratio = npz["point1_ratio"]
            point2_ratio = npz["point2_ratio"]
            point_gt_ratio = npz["point_gt_ratio"]
            retsore_point1 = xyz_restore(Nor_point1, point1_ratio)
            retsore_point2 = xyz_restore(Nor_point2, point2_ratio)
            retsore_pointgt = xyz_restore(Nor_ground_truth, point_gt_ratio)


            # mask_point1 = npz["mask_point1"]
            # mask_color1 = npz["mask_color1"]
            # mask_point2 = npz["mask_point2"]
            # mask_color2 = npz["mask_color2"]
            # mask = npz["mask"]
            # mask_gt = npz["mask_gt"]
            # diff_mask = mask_gt - mask_point1
            # diff = ground_truth - point1
            # print(mask_point2.shape[0])
        print("Finished")
