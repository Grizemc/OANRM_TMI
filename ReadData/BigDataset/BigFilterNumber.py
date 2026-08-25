#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 21:50
# @Author  : 沈子明
# @File    : BigFilter.py
# @Software: PyCharm
import glob
import os
import numpy as np

if __name__ == "__main__":
    path = r"/big_data/szm/M50000ICCAI_mask_mutual/train"
    file_list = glob.glob(os.path.join(path + "/*.npz"))
    fail_file = []
    num = 0
    success = 0
    fail = 0
    yicahng = 0
    for file in file_list:
        npz = np.load(file)
        if len(npz.files) == 12:
            fail += 1
        elif len(npz.files) == 14:
            success += 1
        else:
            yicahng += 1
            print(file)
        num += 1
    print("sum is {}, success is {}, fail is {}, yichang is {}".format(num, success, fail, yicahng))
