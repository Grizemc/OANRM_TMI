#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 11:35
# @Author  : 沈子明
# @File    : CalPercentage.py
# @Software: PyCharm
import glob
import multiprocessing
import os.path
import datetime
import numpy as np
from sklearn.neighbors import NearestNeighbors


def cal_percentage(path):
    files = glob.glob(os.path.join(path, "*.npz"))
    percentage_list = []
    for file in files:
        npz = np.load(file)
        mask_gt1 = npz["mask_gt1"]
        percentage_list.append(mask_gt1.sum() / mask_gt1.shape[0])
    percentage_np = np.array(percentage_list)
    hist, bin_edges = np.histogram(percentage_np, bins='auto')
    import matplotlib.pyplot as plt
    plt.hist(percentage_np, bins='auto')
    plt.title('Histogram of the NumPy Array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    print("path is {}, percentage_np.mean() is {}".format(path, percentage_np.mean()))


if __name__ == "__main__":
    # paths = [r"/big_data/szm/szm_MICCAI_Hamlyn/MICCAI_8192_Train_center/test",
    #          r"/big_data/szm/szm_MICCAI_Hamlyn/MICCAI_8192_Train_center/train",
    #          r"/big_data/szm/szm_MICCAI_Hamlyn/Hamlyn_8192_test_center/test"]

    # paths = [r"/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual/test",
    #          r"/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual/train",
    #          r"/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual/test"]
    paths = [r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/test"]
    #paths = [r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/test"]
    for path in paths:
        cal_percentage(path)
