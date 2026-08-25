#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 17:43
# @Author  : 沈子明
# @File    : TorchCal.py
# @Software: PyCharm
"""
用于验证点云归一化算法，和恢复算法的准确性。
"""
import torch
import numpy as np


def xyz_restore(xyz_in_all, relax_proportion_all):
    B = xyz_in_all.shape[0]
    result =[]
    for i in range(B):
        xyz_in = xyz_in_all[i, :]
        relax_proportion = relax_proportion_all[i]
        len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
        x = xyz_in[:, 0].reshape(-1, 1)
        y = xyz_in[:, 1].reshape(-1, 1)
        z = xyz_in[:, 2].reshape(-1, 1)
        new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
        new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
        new_z = (z + 0.5) * len_z + z_min
        temp = torch.cat((new_x, new_y, new_z), axis=1).reshape(-1, 3)
        result.append(temp)
    result = torch.stack(result)
    return result


def single_xyz_restore(xyz_in, relax_proportion):
    len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
    x = xyz_in[0, :, 0].reshape(-1, 1)
    y = xyz_in[0, :, 1].reshape(-1, 1)
    z = xyz_in[0, :, 2].reshape(-1, 1)
    new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
    new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
    new_z = (z + 0.5) * len_z + z_min
    result = torch.cat((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return result
def single_xyz_to_normalization(xyz_in):
    x = xyz_in[0, :, 0].reshape(-1, 1)
    y = xyz_in[0, :, 1].reshape(-1, 1)
    z = xyz_in[0, :, 2].reshape(-1, 1)
    len_x = x.max() - x.min()
    len_y = y.max() - y.min()
    len_z = z.max() - z.min()
    new_x = (((x - x.min()) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y.min()) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z.min()) / len_z - 0.5
    result = torch.cat((new_x, new_y, new_z), dim=1).reshape(-1, 3)
    return result, ([len_x, len_y, len_z, x.min(), y.min(), z.min()])


if __name__ == "__main__":
    source = torch.rand(20, 4096, 3)
    nor = []
    nor_ratio = []
    for i in range(20):
        sample_data = source[i, :, :].reshape(1, -1, 3)
        nor.append(single_xyz_to_normalization(sample_data)[0])
        nor_ratio.append(single_xyz_to_normalization(sample_data)[1])
    nor_data = torch.stack(nor)

    restroe_single= []
    for i in range(20):
        sample_data = nor_data[i, :, :].reshape(1, -1, 3)
        sample_ratio = nor_ratio[i]
        restroe_single.append(single_xyz_restore(sample_data, sample_ratio))
    restroe_single = torch.stack(restroe_single)
    restore_all = xyz_restore(nor_data, torch.tensor(nor_ratio))
