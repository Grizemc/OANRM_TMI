#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 10:50
# @Author  : 沈子明
# @File    : GenerateMaskUtil.py
# @Software: PyCharm
import argparse
import glob
import multiprocessing
import os
import numpy as np
import open3d
from scipy.spatial.transform import Rotation


def Mask(point1_in, color1_in, ground_truth_in, index_percentage):
    # Random rotation
    rot_random = Rotation.random().as_matrix()
    point1_in = np.matmul(rot_random, point1_in.T).T.astype('float32')
    ground_truth_in = np.matmul(rot_random, ground_truth_in.T).T.astype('float32')
    xyz1_rgb = np.concatenate((point1_in, color1_in), axis=1)
    xyz2_rgb = np.concatenate((ground_truth_in, color1_in), axis=1)

    # sort point cloud coordinate by x coordinate
    xyz1_rgb_sort_index = np.argsort(xyz1_rgb[:, 0], axis=0)
    xyz1_rgb_sort = xyz1_rgb[xyz1_rgb_sort_index, :]
    xyz2_rgb_sort = xyz2_rgb[xyz1_rgb_sort_index, :]

    # overlap
    len_point = color1_in.shape[0]
    mask = np.array([True] * len_point)
    mask1 = mask.copy()
    mask2 = mask.copy()
    index1 = int(index_percentage * len_point)
    index2 = int(len_point - index_percentage * len_point)
    mask1[: index1] = False
    mask2[index2:] = False
    mask_dataset = np.array([True] * mask1.sum())
    mask_dataset[-index1:] = False

    xyz1_rgb_result = xyz1_rgb_sort[mask1]
    xyz2_rgb_result = xyz2_rgb_sort[mask2]
    result_point1 = xyz1_rgb_result[:, 0:3]
    result_color1 = xyz1_rgb_result[:, 3:6]
    result_point2 = xyz2_rgb_result[:, 0:3]
    result_color2 = xyz2_rgb_result[:, 3:6]
    return result_point1, result_color1, result_point2, result_color2, mask_dataset


def Zenike(p, theta):
    """
    :param p:
    :param theta:
    :return: List
    """
    k_coefficient = np.random.rand(20)
    k_coefficient = k_coefficient / k_coefficient.sum()
    zenike_list = []
    result = 0
    zenike_list.append(np.ones(theta.shape))
    zenike_list.append(p * np.cos(theta))
    zenike_list.append(p * np.sin(theta))
    zenike_list.append(-1 + 2 * np.power(p, 2))
    zenike_list.append(np.power(p, 2) * np.cos(2 * theta))
    zenike_list.append(np.power(p, 2) * np.sin(2 * theta))
    zenike_list.append(p * (-2 + 3 * np.power(p, 2)) * np.cos(theta))
    zenike_list.append(p * (-2 + 3 * np.power(p, 2)) * np.sin(theta))
    zenike_list.append(1 - 6 * np.power(p, 2) + 6 * np.power(p, 4))
    zenike_list.append(np.power(p, 3) * np.cos(3 * theta))
    zenike_list.append(np.power(p, 3) * np.sin(3 * theta))
    zenike_list.append(np.power(p, 2) * (-3 + 4 * np.power(p, 2)) * np.cos(2 * theta))
    zenike_list.append(np.power(p, 2) * (-3 + 4 * np.power(p, 2)) * np.sin(2 * theta))
    zenike_list.append(p * (3 - 12 * np.power(p, 2) + 10 * np.power(p, 4)) * np.cos(theta))
    zenike_list.append(p * (3 - 12 * np.power(p, 2) + 10 * np.power(p, 4)) * np.sin(theta))
    zenike_list.append(-1 + 12 * np.power(p, 2) - 30 * np.power(p, 4) + 20 * np.power(p, 6))
    zenike_list.append(np.power(p, 4) * np.cos(4 * theta))
    zenike_list.append(np.power(p, 4) * np.sin(4 * theta))
    zenike_list.append(np.power(p, 3) * (-4 + 5 * np.power(p, 2)) * np.cos(3 * theta))
    zenike_list.append(np.power(p, 3) * (-4 + 5 * np.power(p, 2)) * np.sin(3 * theta))
    for i in range(20):
        result += k_coefficient[i] * zenike_list[i]
    return result


def Read_xyz_len(xyz_in):
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    len_x = np.max(x) - np.min(x)
    len_y = np.max(y) - np.min(y)
    len_z = np.max(z) - np.min(z)
    # 返回x，y，z坐标集合以及对应的每个坐标轴的最大最小值差异
    return x, y, z, len_x, len_y, len_z





def apply_rigid_transform(points):
    def random_rotation_matrix():
        """
        生成一个随机的旋转矩阵（绕 X、Y、Z 轴的旋转）。
        返回 3x3 旋转矩阵。
        """
        # 生成随机角度（-10 到 10 之间）
        angle_x = np.random.uniform(-np.pi / 18, np.pi / 18)
        angle_y = np.random.uniform(-np.pi / 18, np.pi / 18)
        angle_z = np.random.uniform(-np.pi / 18, np.pi / 18)

        # 绕 X 轴的旋转矩阵
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])

        # 绕 Y 轴的旋转矩阵
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])

        # 绕 Z 轴的旋转矩阵
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])

        # 组合所有旋转矩阵
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def random_translation():
        """
        生成一个随机平移向量，用于模拟腹腔镜器官的位移。
        返回 3D 平移向量，X、Y、Z方向的平移值。
        """
        # 生成平移向量，限制范围为 -0.1 到 0.1 或 -0.15 的小范围
        tx = np.random.uniform(-2, 2)  # X 方向平移
        ty = np.random.uniform(-2, 2)  # Y 方向平移
        tz = np.random.uniform(-1, 1)  # Z 方向平移，适当增加 Z 轴范围
        return np.array([tx, ty, tz])
    """
    对输入的点云应用随机刚性变换（旋转 + 平移）。
    :param points: 点云数据 (N, 3)
    :return: 变换后的点云数据
    """
    # 获取随机旋转矩阵和平移向量
    R = random_rotation_matrix()
    t = random_translation()

    # 对点云应用旋转和位移
    transformed_points = np.dot(points, R.T) + t
    return transformed_points

def GendeformSourceSize(xyz_in):
    """
    生成形变所需要的数据
    :param xyz_in: No normalized coordinates
    """
    x, y, z, len_x, len_y, len_z = Read_xyz_len(xyz_in)
    new_x = (((x - np.min(x)) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - np.min(y)) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - np.min(z)) / len_z - 0.5
    point = np.concatenate((new_x, new_y, new_z), axis=1)  # 在最右侧添加一列
    # =================随机选取中心点, 距离, 角度==========================
    sample_center = True
    while sample_center:
        pnum = point.shape[0]
        center_index = np.random.randint(0, pnum)
        center_point = point[center_index, :]
        center_point_x = np.abs(center_point[0])
        center_point_y = np.abs(center_point[1])
        center_point_z = np.abs(center_point[2])
        # 采样中心点直到sample_center = False
        if center_point_x < 0.15 and center_point_y < 0.15 and center_point_z < 0.25:
            sample_center = False
    center_point = point[center_index, :].reshape(1, 3).repeat(pnum, axis=0)
    # 距离
    pdifference = np.sqrt(np.sum(np.square(point[:, :2] - center_point[:, :2]), axis=1))
    pmax, pmin = pdifference.max(), pdifference.min()
    pdifference = (pdifference - pmin) / (pmax - pmin)
    # 角度，利用反正切
    tanangle = (point[:, 1] - center_point[:, 1]) / (point[:, 0] - center_point[:, 0] + 1e-10)
    tanangle = np.where(tanangle > 0, tanangle, -tanangle)
    theta = np.arctan(tanangle)  # 角度
    # =================生成形变==========================
    deforml = Zenike(pdifference, theta)
    while deforml.max() > 1.0 or deforml.max() < 0.1:
        deforml = Zenike(pdifference, theta)  # 保证形变的程度不太大
    ground_truth_in = xyz_in.copy()
    ground_truth_in[:, 2] += deforml * len_z
    # Step 5: 对形变后的点云应用随机刚性变换
    ground_truth_in = apply_rigid_transform(ground_truth_in)

    return xyz_in, ground_truth_in


# 变的是点云坐标，而不是点云颜色
def New_Mutual_Mask(point1_in, color1_in, ground_truth_in, index_percentage):
    # Random rotation 随机旋转
    rot_random = Rotation.random().as_matrix()
    temp_point1_in = np.matmul(rot_random, point1_in.T).T.astype('float64')

    # 第一片点云的xyz和rgb
    xyz1_rgb = np.concatenate((point1_in, color1_in), axis=1)
    # 第二片点云的xyz和rgb
    xyz2_rgb = np.concatenate((ground_truth_in, color1_in), axis=1)

    # sort point cloud coordinate by x coordinate，点云排序
    # np.argsort返回排序的点云在原始点云集下的索引，然后排序点云
    xyz1_rgb_sort_index = np.argsort(temp_point1_in[:, 0], axis=0)
    xyz1_rgb_sort = xyz1_rgb[xyz1_rgb_sort_index, :]
    # ！！！！！！！！！！！！！！！！！！！！！
    # 第二片点云也按照索引排序，以实现第一片和第二片的对应
    xyz2_rgb_sort = xyz2_rgb[xyz1_rgb_sort_index, :]

    # overlap，重叠
    len_point = color1_in.shape[0]
    # 长度为N
    mask = np.array([True] * len_point)
    mask1 = mask.copy()
    mask2 = mask.copy()
    index = int(index_percentage * len_point)
    # 定点数
    index1 = index
    index2 = len_point - index
    mask1[: index1] = False  # 仅用于取点云坐标，前index个
    mask2[index2:] = False  # 仅用于取点云坐标，后index个
    # 用于掩码的生成，mask_gt1生成len_point - index 个[True]
    mask_gt1 = np.array([True] * mask1.sum())
    # 用于掩码的生成，mask_gt2生成len_point - index 个[True]
    mask_gt2 = np.array([True] * mask2.sum())

    # 真正的重叠区域索引在mask_gt处，前后都减去了
    mask_gt1[-index1:] = False  # 将 mask_gt1 数组的最后 index1 个元素设置为 False
    mask_gt2[:index1] = False  # 将 mask_gt2 数组的前 index1 个元素设置为 False

    # 根据掩码中筛选出有效的点云数据和颜色信息。返回的是xyz和rgb的筛选结果
    # 分别筛选两篇点云
    xyz1_rgb_result = xyz1_rgb_sort[mask1]
    xyz2_rgb_result = xyz2_rgb_sort[mask2]

    # 第二片点云去除掉 第一片能看到，第二片看不到的点。符合实际实验需要
    # mask1 数组中 True 所在位置的行被选择并返回。
    mask_pc_gt = xyz2_rgb_sort[mask1][:, 0:3]

    result_point1 = xyz1_rgb_result[:, 0:3]
    result_color1 = xyz1_rgb_result[:, 3:6]
    result_point2 = xyz2_rgb_result[:, 0:3]
    result_color2 = xyz2_rgb_result[:, 3:6]
    # 所以最后重叠区域的索引肯定是result_point1和mask_gt1共同决定。
    # xyz1_rgb_sort和xyz2_rgb_sort 也return一下
    return result_point1, result_color1, result_point2, result_color2, mask_gt1, mask_gt2, mask_pc_gt

# 点云下采样
def VoxelDownSample(pcd, temp_size, index_percentage, source_pcd_num):
    xyz_in = np.array(pcd.points)
    x, y, z, len_x, len_y, len_z = Read_xyz_len(xyz_in)
    # Control the number of iterations
    number_iteration = 0
    return_type = True
    # sample point
    if source_pcd_num == 8192:
        # 目标点数目 source_pcd_num=8192
        target_num = (source_pcd_num / (1 - index_percentage))
        large_num = ((source_pcd_num + 500) / (1 - index_percentage))
    else:
        # 目标点数目
        target_num = (source_pcd_num / (1 - index_percentage))
        large_num = ((source_pcd_num + 2000) / (1 - index_percentage))
        # 若点的数量在目标点数目和最大数目之间，则符合要求
    if target_num <= np.array(pcd.points).shape[0] <= large_num:
        return pcd, return_type, temp_size
    # sample 体素均匀下采样
    # voxel_size = 400
    voxel_size_in = (len_x + len_y + len_z) / temp_size
    pcd_voxel_down = pcd.voxel_down_sample(voxel_size=voxel_size_in)  # 体素均匀下采样
    num_last = np.array(pcd_voxel_down.points).shape[0]
    factor = 1
    # 调整体素大小，直到点数在目标范围内
    while num_last < target_num or num_last >= large_num:
        if num_last < target_num:
            temp_size += factor
        elif num_last >= target_num:
            temp_size -= factor
        else:
            break
        number_iteration += 1

        # voxel_size = 400
        # voxel_size_in随着temp_size改变，temp随着factor改变而改变
        # 各个坐标轴上的范围和长度，以此取个具体的平均值，以此平均值作为体素的长宽高
        voxel_size_in = (len_x + len_y + len_z) / temp_size
        # 体素变小，采样点数会变多
        pcd_voxel_down = pcd.voxel_down_sample(voxel_size=voxel_size_in)
        num = np.array(pcd_voxel_down.points).shape[0]
        # 调整体素大小，target_num为8192还多
        # num为下采样点数，num_last也为下采样点数
        if num_last > target_num * 2 and abs(num - num_last) < 200:
            factor = factor * 3
        elif num_last < target_num * 1.2 and abs(num - num_last) < 50:
            factor = factor * 2
        elif num_last < target_num * 0.8 and abs(num - num_last) < 200:
            factor = factor / 3
        elif num_last < target_num * 0.8 and abs(num - num_last) > 60:
            factor = factor / 2
        num_last = num # 重新在whlie循环中判断，看看数量点的变换，然后
        if number_iteration > 500:
            return_type = False
            # 退出循环
            break
    return pcd_voxel_down, return_type, temp_size


def quickRemove000(pcd):
    """
    :param pcd: point cloud open3d
    :return: point cloud open3d without (0,0,0)
    """
    xyz_in = np.array(pcd.points)  # 读取点云中的点坐标
    rgb = np.array(pcd.colors)  # 读取点云中的点颜色
    # 生成需要删除的点的索引
    temp_mask = (xyz_in == [0, 0, 0])
    # 布尔值相乘，得到一个N*1的数组
    mask = temp_mask[:, 0] * temp_mask[:, 1] * temp_mask[:, 2]
    # 将为0的置为False
    mask = ~ mask
    # 取除了0点以外的其他坐标和颜色信息
    xyz_remove_0 = xyz_in[mask, :]
    rgb_remove_0 = rgb[mask, :]
    point_cloud = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(xyz_remove_0))
    point_cloud.colors = open3d.pybind.utility.Vector3dVector(rgb_remove_0)
    return point_cloud


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
