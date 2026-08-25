#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 15:04
# @Author  : 沈子明
# @File    : FPFHMultiHamlyn.py
# @Software: PyCharm
import glob
import multiprocessing
import os
import open3d as o3d
import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation

# 生成hamlyn对应不同重叠比例数据集的伪真值
def teturn_color(result, source_cloud, target_cloud):
    matches = np.asarray(result.correspondence_set)
    if len(matches) == 0:
        color_error = 2238
    else:
        idx_source = matches[:, 0]
        idx_target = matches[:, 1]
        color_source = np.array(source_cloud.colors)[idx_source]
        color_target = np.array(target_cloud.colors)[idx_target]
        color_error_in = np.mean(np.linalg.norm((color_target - color_source), axis=1))
    return color_error_in


def single_fpfh(source_cloud, target_cloud, source_fpfh, target_fpfh, distance_threshold):
    need_ransac = True
    count = 0
    while need_ransac:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            # o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
             ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        if result.fitness != 0:
            # if np.max(np.abs(R.from_matrix(result.transformation[:3, :3].copy()).as_euler("xyz", degrees=True))) < 45:
            need_ransac = False
            color_error = teturn_color(result, source_cloud, target_cloud)
            count += 1
    return color_error, result

def augmentation_onlyRT_test_data(pos1, pos2, ground_truth):
    # random rotation
    np.random.seed(2000)
    rot_target = Rotation.random().as_matrix()
    # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
    pos2 = np.einsum('ij,kj->ki', rot_target, pos2).astype('float32')
    ground_truth = np.einsum('ij,kj->ki', rot_target, ground_truth).astype('float32')
    # random offset
    np.random.seed(2000)
    offset2 = np.random.rand(1, 3)
    pos2 += offset2
    ground_truth += offset2
    return pos1, pos2, ground_truth
def CalFPFh(paths):
    distance_thresholds = [1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]
    target_path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/fpft_file"
    for path in paths:
        with np.load(path) as fp:
            points1 = fp["mask_point1"].astype('float32')
            colors1 = fp["mask_color1"].astype('float32')
            points2 = fp["mask_point2"].astype('float32')
            colors2 = fp["mask_color2"].astype('float32')
        _, points2, _ = augmentation_onlyRT_test_data(points2, points2, points2)
        source_cloud = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((points1)))
        source_cloud.colors = o3d.pybind.utility.Vector3dVector(colors1)
        target_cloud = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector((points2)))
        target_cloud.colors = o3d.pybind.utility.Vector3dVector(colors2)
        # 为两个点云计算FPFH特征
        radius_normal = 2  # 法线估计半径
        radius_feature = 2.5  # FPFH计算半径

        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        matches_list = []
        transformations_list = []
        color_errors_list = []
        fitness_list = []
        inlier_rmse_list = []
        for distance_threshold in distance_thresholds:
            color_error1, result1 = single_fpfh(source_cloud, target_cloud, source_fpfh, target_fpfh,
                                                distance_threshold)
            color_error2, result2 = single_fpfh(source_cloud, target_cloud, source_fpfh, target_fpfh,
                                                distance_threshold)
            color_error3, result3 = single_fpfh(source_cloud, target_cloud, source_fpfh, target_fpfh,
                                                distance_threshold)
            if color_error1 <= color_error2:
                result = result1
                color_error = color_error1
            else:
                result = result2
                color_error = color_error2
            if color_error3 <= color_error:
                result = result3
                color_error = color_error3
            if result.fitness == 0:
                pass
            matches = np.asarray(result.correspondence_set)
            transformations_list.append(np.asarray(result.transformation))
            color_errors_list.append(color_error)
            matches_list.append(matches)
            fitness_list.append(result.fitness)
            inlier_rmse_list.append(result.inlier_rmse)

        color_errors_list = np.array(color_errors_list)
        inlier_rmse_list = np.array(inlier_rmse_list)
        fitness_list = np.array(fitness_list)
        distance_thresholds = np.array(distance_thresholds)
        transformations_list = np.array(transformations_list)

        # 选择第一个列表中最大的5个元素的索引
        max_indices = sorted(range(len(inlier_rmse_list)), key=lambda k: inlier_rmse_list[k], reverse=True)
        # 根据索引取出第二个列表中的值，并将这些值与对应的索引建立映射关系
        value_mapping = [(color_errors_list[idx], idx) for idx in max_indices]
        # 根据映射关系中的值从大到小排序，并获取排序后的索引
        sorted_indices = [idx for _, idx in sorted(value_mapping, key=lambda x: x[0], reverse=False)]
        # 获取不在最大5个元素索引内的索引
        remaining_indices = [idx for idx in range(len(fitness_list)) if idx not in max_indices]
        index = sorted_indices + remaining_indices
        matches_list = [matches_list[i] for i in index]
        distance_thresholds = distance_thresholds[index]
        transformations_list = transformations_list[index]
        color_errors_list = color_errors_list[index]
        inlier_rmse_list = inlier_rmse_list[index]
        fitness_list = fitness_list[index]
        save_path = os.path.join(target_path, path.split('/')[-1])

        pad_list(matches_list, 7, matches_list[-1])
        np.savez(save_path, distance_thresholds=distance_thresholds,
                 matches_list0=matches_list[0],
                 matches_list1=matches_list[1],
                 matches_list2=matches_list[2],
                 matches_list3=matches_list[3],
                 matches_list4=matches_list[4],
                 matches_list5=matches_list[5],
                 matches_list6=matches_list[6],
                 color_errors_list=color_errors_list,
                 transformations_list=transformations_list,
                 inlier_rmse_list=inlier_rmse_list, fitness_list=fitness_list)
        print("save_path:", save_path)


def pad_list(lst, target_length, fill_value):
    if len(lst) >= target_length:
        return lst
    else:
        padding = [fill_value] * (target_length - len(lst))
        return lst + padding


if __name__ == "__main__":
    all_split_num = 81
    pool_num = 3
    # files中全是hamlyn测试集，用来生成其对应的伪真值
    files = glob.glob(os.path.join(r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/test", "*.npz"))
    files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    files.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
    files = files[1200:]
    all_len = len(files)
    temp_len = all_len // all_split_num
    file_group_list = []
    last_number = 0
    for i in range(0, all_len, temp_len):
        if i != 0:
            temp_list = files[last_number:i]
            file_group_list.append(temp_list)
        last_number = i
    if last_number != all_len:
        temp_list = files[last_number:all_len]
        file_group_list.append(temp_list)
    # CalFPFh(files)

    multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    pool = multiprocessing.Pool(pool_num)
    pool.map(CalFPFh, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
