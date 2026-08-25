#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 16:54
# @Author  : 沈子明
# @File    : Sample.py
# @Software: PyCharm
"""
取部分Hamlyn的数据作为数据集
"""

import glob
import os


def sample_npz(path):
    path = r"/big_data/szm/TanTaiBiaoZhu/szmNpz/test"
    target_path = r"/big_data/szm/TanTaiBiaoZhu/szmNpz/test/little"
    file_list = glob.glob(os.path.join(path, "*.npz"))
    # Hamlyn
    file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    file_list.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
    # MICCAI
    file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-2]))
    # for i in range(0, len(file_list), 50):
    #     print(i)
    #     os.system('cp {} {}'s.format(file_list[i], os.path.join(target_path, file_lit[i].split('/')[-1])))


def sample_image01():
    root_path = r"/big_data/szm/Cache_MICCAI_Hamlyn/HamlynSource"
    target_path = r"/big_data/szm/szm_MICCAI_Hamlyn/Hamlyn_8192_1973/image01/"
    image01_paths = []
    for folder_path in glob.glob(os.path.join(root_path, "rectified*")):
        image01_paths.extend(glob.glob(os.path.join(root_path, folder_path, "image01/", "*.jpg")))
    image01_paths.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    image01_paths.sort(key=lambda x: int(x.split('rectified')[-1].split('/')[0]))
    for i in range(0, len(image01_paths), 50):
        print(i)
        save_path = os.path.join(target_path, image01_paths[i].split('/')[-3]) + "_" + str(
            int(image01_paths[i].split('/')[-1].split(".")[0])) + ".jpg"
        os.system('cp {} {}'.format(image01_paths[i], save_path))


if __name__ == "__main__":
    path1 = r"/big_data/szm/szm_MICCAI_Hamlyn/Hamlyn_8192_1973/image01/"
    path2 = r"/big_data/szm/szm_MICCAI_Hamlyn/Hamlyn_8192_1973/test/"
    files1 = glob.glob(os.path.join(path1, "*.jpg"))
    files2 = glob.glob(os.path.join(path2, "*.npz"))
    for file in files1:
        file_name = file.split("/")[-1].split(".")[0]
        # find file_name not in files1
        if not os.path.exists(os.path.join(path2, file_name + ".npz")):
            os.system("rm -rf  {} ".format(file))
            # del file in windwos
            # os.system("del  {} ".format(file))
