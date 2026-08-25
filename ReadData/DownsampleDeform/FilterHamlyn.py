#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 20:29
# @Author  : 沈子明
# @File    : FilterHamlyn.py
# @Software: PyCharm

import glob
import multiprocessing
import os
import numpy as np
# 筛选hamlyn数据集

if __name__ == "__main__":
    # config 某些设置文件写在 GenerateDataset 函数的里面
    root_path = r"/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSourceNpz_91864"
    file_list = glob.glob(os.path.join(root_path + "/*.npz"))
    # split dataset
    file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    file_list.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
    for i in range(0, len(file_list), 50):
        print(i)
        file = file_list[i]
        os.system("cp {} {} ".format(file, os.path.join(root_path, "additional", file.split('/')[-1])))
