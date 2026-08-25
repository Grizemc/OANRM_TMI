#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 16:34
# @Author  : 沈子明
# @File    : ReadSourceHamly.py
# @Software: PyCharm

"""
专门读取数据集的npz文件。
"""
# 测试数据集重叠比率的文件

import glob
import os.path

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
    # file2 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_80_bigdeform/test"
    # file1 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85_bigdeform/fpft_file/fpft_file"
    # file3 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85_bigdeform/test"

    file2 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_80_bigdeform/test"
    file1 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/test"
    file3 = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85_bigdeform/test"

# H8amlyn_8192_Mask_3332_new_mutual_85_bigdeform
    # file /big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual/test/dataset7_keyframe_4_1913.npz all percentage is 94.78116218974483
    for file in [file1,file2,file3]:
        # if file == file3:
           #  import pdb; pdb.set_trace()
        files = glob.glob(os.path.join(file, "*.npz"))
        num = 0
        for file in files:
            try:
                with np.load(file) as fp:
                    # 重叠比例计算
                    mask_gt1 = fp["mask_gt1"]
                # per += mask_gt2.astype(int).sum()/mask_gt2.shape[0]
                    num += mask_gt1.sum()/mask_gt1.shape[0]
            except:
                print(file)
                print("111111111111111")
        num = num/len(files) * 100
        print("file:", len(files))
        print("file {} all percentage is {} ".format(file, num))
    # path = files[0]

    # data = np.load(path)










    # # 将数据保存到字典中
    # data_dict = {key: data[key] for key in data.files}
    # print("dasd {} ")