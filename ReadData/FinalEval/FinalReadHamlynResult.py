#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 20:06
# @Author  : 沈子明
# @File    : FinalReadHamlynResult.py
# @Software: PyCharm
import glob
import os.path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

# 首先加载无监督微调权重
# 之后对95%数据集进行后处理操作
# 有五种不同的后处理方式
# PostMostProcessHamlynResult是进行后处理的操作 保存在Inter中
# GaussianPostMostProcessHamlynResult保存在GaussianInter

# CalMostGaussinaPostProcessHamlynResult 为评测精度的文件 # 读取后处理的精度文件：
# DirectCalReadHanlynResult   读取有监督+无监督结果的精度文件
# 精度文件存在GaussianInter.npy中 Inter.npy中 FilterInter.npy中

# DirectCalReadHanlynResult函数读取 无监督微调的文件npz_result 得到EvalHamlyn.npz
# EvalHamlyn.npz为读取无监督微调npz_result的精度结果

# 形变场方差评估
class FinalReadHamlynResultIOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        print("=================================")
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def Cal_Final_Hamlyn_result_np(pcd1, pcd2):
    eval_num = pcd1.shape[0]
    pcd1 = pcd1[:, :]
    pcd2 = pcd2[:, :]
    displace = np.linalg.norm(pcd1 - pcd2, axis=1)
    relax_error = [1, 2, 3, 4, 5]
    acc_list = []
    for error in relax_error:
        acc_list.append((displace < error).sum() / eval_num)
    acc_list.append(displace.mean())
    acc_list = np.array(acc_list)
    return acc_list

# 形变场方差评估
def displacement_variance(pcd1, pcd2):
    eval_num = pcd1.shape[0]
    pcd1 = pcd1[:, :]
    pcd2 = pcd2[:, :]
    # 计算每个向量（每行）的 L2 范数（欧几里得距离）
    displace = np.linalg.norm(pcd1 - pcd2, axis=1)
    mean = displace.mean()
    # variance = np.linalg.norm(displace-mean,axis=1)/eval_num

    variance  = np.sum((displace - mean) ** 2)/eval_num
    relax_error = [1, 2, 3, 4, 5]
    variance_acc_list = []
    for error in relax_error:
        variance_acc_list.append((variance < error).sum() / eval_num)
    variance_acc_list = np.array(variance_acc_list)
    return variance_acc_list


def Cal_Mask_evaluate_result(pred, gt):
    pred = pred.squeeze()
    N = gt.shape[0]
    truncation_nums = [0.7, 0.75, 0.8, 0.85, 0.9]
    success_ratio = []
    for truncation_num in truncation_nums:
        pred_trun = pred > truncation_num
        success_ratio.append((pred_trun == gt).astype(np.float64).sum() / N)
    success_ratio = np.array(success_ratio)
    return success_ratio


def DirectCalReadHanlynResult(root_path):
    """
    """
    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    final_acc = 0.
    mask_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            point1 = npz["points1"][0, ::]
            point2 = npz["points2"][0, ::]
            color1 = npz["colors1"][0, ::]
            color2 = npz["colors2"][0, ::]
            pred_xyz = npz["pred_xyz"][0, ::]
            mask_gt_pc = npz["mask_gt_pc"][0]
            pred_mask1 = npz["pred_mask1"][0, ::]
            mask_gt1 = npz["mask_gt1"][0]
            mask_acc += Cal_Mask_evaluate_result(pred_mask1, mask_gt1)
            final_acc += Cal_Final_Hamlyn_result_np(pred_xyz, mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    mask_acc = mask_acc * 100
    mask_acc = mask_acc / len(npz_results)
    print("final_acc is {}, mask acc is {}".format(final_acc, mask_acc))
    save_path = root_path + "/EvalHamlyn.npz"
    np.savez(save_path, final_acc=final_acc, mask_acc=mask_acc)


# TDDP操作，针对无重叠区域进行的
def PostProcessHamlynResult(root_path):
    """
    ①首先使用重叠区域的点估计出一个RANSAC，
    ②使用一个整体的位移向量进行滤波，率除掉一些异常点
    ③滤波后的点记为 高置信点，使用高置信点进行插值
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "FilterInter")
    io = FinalReadHamlynResultIOStream(root_path + '/FilterInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    for file in npz_results:
        save_path = os.path.join(results_save_path, file.split("/")[-1])
        if os.path.exists((save_path)):
            pass
        else:
            try:
                with np.load(file) as npz:
                    point1 = npz["points1"][0, ::]
                    point2 = npz["points2"][0, ::]
                    color1 = npz["colors1"][0, ::]
                    color2 = npz["colors2"][0, ::]
                    pred_xyz = npz["pred_xyz"][0, ::]
                    mask_gt_pc = npz["mask_gt_pc"][0]
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_gt1 = npz["mask_gt1"][0]

                # 使用ransac算法进行整体位移变换
                ransac_pred_mask1 = pred_mask1 > 0.9
                ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
                ransac_pred_pcd = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
                ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
                ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
                distance_threshold = 2
                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
                )
                ratation_point1 = np.dot(ransac_result.transformation,
                                         np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]
                # 使用Flow剔除
                pred_mask1 = pred_mask1 > 0.9
                flow = pred_xyz - ratation_point1
                flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
                flow_true_mean = flow_true.mean(axis=0)
                dot_products = np.dot(flow_true, flow_true_mean)  # 计算向量的点积
                cosine_angles = dot_products / (
                        np.linalg.norm(flow_true, axis=1) * np.linalg.norm(flow_true_mean))  # 计算余弦夹角
                angles_radians = np.degrees(np.arccos(cosine_angles))  # 计算角度（弧度）
                new_true = angles_radians < 90
                true_true_point = ratation_point1[pred_mask1][new_true]
                true_true_color = color1[pred_mask1][new_true]
                true_true_flow = flow[pred_mask1][new_true]
                true_mask_gt_pc = mask_gt_pc[pred_mask1][new_true]
                false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]),
                                             axis=0)
                false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
                false_mask_gt_pc = np.concatenate((mask_gt_pc[~pred_mask1], mask_gt_pc[pred_mask1][~new_true]), axis=0)
                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1][new_true]), axis=0)
                new_color = np.concatenate((false_color, true_true_color), axis=0)
                new_mask_gt_pc = np.concatenate((false_mask_gt_pc, true_mask_gt_pc), axis=0)

                post_train = True
                np.savez(save_path,
                         new_pred_point=new_pred_point,
                         new_color=new_color,
                         new_mask_gt_pc=new_mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
            except:
                post_train = False
                np.savez(save_path,
                         new_pred_point=pred_xyz,
                         new_color=color1,
                         new_mask_gt_pc=mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def PostMostProcessHamlynResult(root_path):
    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "Inter")
    io = FinalReadHamlynResultIOStream(root_path + '/Inter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    for file in npz_results:
        save_path = os.path.join(results_save_path, file.split("/")[-1])
        if os.path.exists((save_path)):
            pass
        else:
            try:
                with np.load(file) as npz:
                    point1 = npz["points1"][0, ::]
                    point2 = npz["points2"][0, ::]
                    color1 = npz["colors1"][0, ::]
                    color2 = npz["colors2"][0, ::]
                    pred_xyz = npz["pred_xyz"][0, ::]
                    mask_gt_pc = npz["mask_gt_pc"][0]
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_gt1 = npz["mask_gt1"][0]

                # 使用ransac算法进行整体位移变换
                ransac_pred_mask1 = pred_mask1 > 0.9
                ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
                ransac_pred_pcd = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
                ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
                ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
                distance_threshold = 2
                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
                )
                ratation_point1 = np.dot(ransac_result.transformation,
                                         np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]
                pred_mask1 = pred_mask1 > 0.9
                flow = pred_xyz - ratation_point1
                flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
                true_point = ratation_point1[pred_mask1]
                true_color = color1[pred_mask1]
                true_flow = flow[pred_mask1]
                true_mask_gt_pc = mask_gt_pc[pred_mask1]
                false_point = ratation_point1[~pred_mask1]
                false_color = color1[~pred_mask1]
                false_mask_gt_pc = mask_gt_pc[~pred_mask1]
                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1]), axis=0)
                new_color = np.concatenate((false_color, true_color), axis=0)
                new_mask_gt_pc = np.concatenate((false_mask_gt_pc, true_mask_gt_pc), axis=0)

                post_train = True
                np.savez(save_path,
                         new_pred_point=new_pred_point,
                         new_color=new_color,
                         new_mask_gt_pc=new_mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
            except:
                post_train = False
                np.savez(save_path,
                         new_pred_point=pred_xyz,
                         new_color=color1,
                         new_mask_gt_pc=mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def GaussianPostMostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    def GaussianFilter(point1, pred_pc, knn_num, sigma):
        flow = pred_pc - point1
        nbrs = NearestNeighbors(n_neighbors=knn_num + 1, algorithm='auto').fit(point1)
        _, indices = nbrs.kneighbors(point1)
        flow_neigh = flow[indices]
        indices = indices[:, 1:]
        point1_neigh = point1[indices]
        point1_neigh_relative = point1_neigh - point1[:, np.newaxis, :]
        center_point = np.zeros_like(flow)
        point1_relative = np.concatenate((center_point[:, np.newaxis, :], point1_neigh_relative), axis=1)
        gaussian_weight_up = np.exp(-(np.sum(point1_relative ** 2, 2)) / (2 * sigma ** 2))
        gaussian_weight_down = (np.power(2 * np.pi, 1.5) * np.power(sigma, 3))
        gaussian_weight = gaussian_weight_up / gaussian_weight_down
        gaussian_weight = (gaussian_weight / gaussian_weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
        flow_result = (flow_neigh * gaussian_weight).sum(axis=1)
        new_pred_pc = flow_result + point1
        return new_pred_pc

    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "GaussianInter")
    io = FinalReadHamlynResultIOStream(root_path + '/GaussianInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    for file in npz_results:
        save_path = os.path.join(results_save_path, file.split("/")[-1])
        if os.path.exists((save_path)):
            pass
        else:
            try:
                with np.load(file) as npz:
                    point1 = npz["points1"][0, ::]
                    point2 = npz["points2"][0, ::]
                    color1 = npz["colors1"][0, ::]
                    color2 = npz["colors2"][0, ::]
                    pred_xyz = npz["pred_xyz"][0, ::]
                    mask_gt_pc = npz["mask_gt_pc"][0]
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_gt1 = npz["mask_gt1"][0]
                pred_xyz = GaussianFilter(point1, pred_xyz, 8, 5)
                # 使用ransac算法进行整体位移变换
                ransac_pred_mask1 = pred_mask1 > 0.9
                ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
                ransac_pred_pcd = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
                ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
                ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
                distance_threshold = 2
                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
                )
                ratation_point1 = np.dot(ransac_result.transformation,
                                         np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]

                pred_mask1 = pred_mask1 > 0.9
                flow = pred_xyz - ratation_point1
                flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
                true_point = ratation_point1[pred_mask1]
                true_color = color1[pred_mask1]
                true_flow = flow[pred_mask1]
                true_mask_gt_pc = mask_gt_pc[pred_mask1]
                false_point = ratation_point1[~pred_mask1]
                false_color = color1[~pred_mask1]
                false_mask_gt_pc = mask_gt_pc[~pred_mask1]
                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1]), axis=0)
                new_color = np.concatenate((false_color, true_color), axis=0)
                new_mask_gt_pc = np.concatenate((false_mask_gt_pc, true_mask_gt_pc), axis=0)

                post_train = True
                np.savez(save_path,
                         new_pred_point=new_pred_point,
                         new_color=new_color,
                         new_mask_gt_pc=new_mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
            except:
                post_train = False
                np.savez(save_path,
                         new_pred_point=pred_xyz,
                         new_color=color1,
                         new_mask_gt_pc=mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def GaussianPostProcessHamlynResult(root_path):
    def GaussianFilter(point1, pred_pc, knn_num, sigma):
        flow = pred_pc - point1
        nbrs = NearestNeighbors(n_neighbors=knn_num + 1, algorithm='auto').fit(point1)
        _, indices = nbrs.kneighbors(point1)
        flow_neigh = flow[indices]
        indices = indices[:, 1:]
        point1_neigh = point1[indices]
        point1_neigh_relative = point1_neigh - point1[:, np.newaxis, :]
        center_point = np.zeros_like(flow)
        point1_relative = np.concatenate((center_point[:, np.newaxis, :], point1_neigh_relative), axis=1)
        gaussian_weight_up = np.exp(-(np.sum(point1_relative ** 2, 2)) / (2 * sigma ** 2))
        gaussian_weight_down = (np.power(2 * np.pi, 1.5) * np.power(sigma, 3))
        gaussian_weight = gaussian_weight_up / gaussian_weight_down
        gaussian_weight = (gaussian_weight / gaussian_weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
        flow_result = (flow_neigh * gaussian_weight).sum(axis=1)
        new_pred_pc = flow_result + point1
        return new_pred_pc

    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "GaussianFilterInter")
    io = FinalReadHamlynResultIOStream(root_path + '/GaussianFilterInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    for file in npz_results:
        save_path = os.path.join(results_save_path, file.split("/")[-1])
        if os.path.exists((save_path)):
            pass
        else:
            try:
                with np.load(file) as npz:
                    point1 = npz["points1"][0, ::]
                    point2 = npz["points2"][0, ::]
                    color1 = npz["colors1"][0, ::]
                    color2 = npz["colors2"][0, ::]
                    pred_xyz = npz["pred_xyz"][0, ::]
                    mask_gt_pc = npz["mask_gt_pc"][0]
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_gt1 = npz["mask_gt1"][0]
                pred_xyz = GaussianFilter(point1, pred_xyz, 8, 5)
                # 使用ransac算法进行整体位移变换
                ransac_pred_mask1 = pred_mask1 > 0.9
                ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
                ransac_pred_pcd = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
                ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
                ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
                distance_threshold = 2
                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
                )
                ratation_point1 = np.dot(ransac_result.transformation,
                                         np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]
                # 使用Flow剔除
                pred_mask1 = pred_mask1 > 0.9
                flow = pred_xyz - ratation_point1
                flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
                flow_true_mean = flow_true.mean(axis=0)
                dot_products = np.dot(flow_true, flow_true_mean)  # 计算向量的点积
                cosine_angles = dot_products / (
                        np.linalg.norm(flow_true, axis=1) * np.linalg.norm(flow_true_mean))  # 计算余弦夹角
                angles_radians = np.degrees(np.arccos(cosine_angles))  # 计算角度（弧度）
                new_true = angles_radians < 90
                true_true_point = ratation_point1[pred_mask1][new_true]
                true_true_color = color1[pred_mask1][new_true]
                true_true_flow = flow[pred_mask1][new_true]
                true_mask_gt_pc = mask_gt_pc[pred_mask1][new_true]
                false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]),
                                             axis=0)
                false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
                false_mask_gt_pc = np.concatenate((mask_gt_pc[~pred_mask1], mask_gt_pc[pred_mask1][~new_true]), axis=0)
                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1][new_true]), axis=0)
                new_color = np.concatenate((false_color, true_true_color), axis=0)
                new_mask_gt_pc = np.concatenate((false_mask_gt_pc, true_mask_gt_pc), axis=0)

                post_train = True
                np.savez(save_path,
                         new_pred_point=new_pred_point,
                         new_color=new_color,
                         new_mask_gt_pc=new_mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
            except:
                post_train = False
                np.savez(save_path,
                         new_pred_point=pred_xyz,
                         new_color=color1,
                         new_mask_gt_pc=mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def TrueGaussianPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    def TrueFlowGaussianFilter(point1, flow, knn_num, sigma):
        nbrs = NearestNeighbors(n_neighbors=knn_num + 1, algorithm='auto').fit(point1)
        _, indices = nbrs.kneighbors(point1)
        flow_neigh = flow[indices]
        indices = indices[:, 1:]
        point1_neigh = point1[indices]
        point1_neigh_relative = point1_neigh - point1[:, np.newaxis, :]
        center_point = np.zeros_like(flow)
        point1_relative = np.concatenate((center_point[:, np.newaxis, :], point1_neigh_relative), axis=1)
        gaussian_weight_up = np.exp(-(np.sum(point1_relative ** 2, 2)) / (2 * sigma ** 2))
        gaussian_weight_down = (np.power(2 * np.pi, 1.5) * np.power(sigma, 3))
        gaussian_weight = gaussian_weight_up / gaussian_weight_down
        gaussian_weight = (gaussian_weight / gaussian_weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
        flow_result = (flow_neigh * gaussian_weight).sum(axis=1)
        return flow_result

    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "TrueGaussianFilterInter")
    io = FinalReadHamlynResultIOStream(root_path + '/TrueGaussianFilterInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    for file in npz_results:
        save_path = os.path.join(results_save_path, file.split("/")[-1])
        if os.path.exists((save_path)):
            pass
        else:
            try:
                with np.load(file) as npz:
                    point1 = npz["points1"][0, ::]
                    point2 = npz["points2"][0, ::]
                    color1 = npz["colors1"][0, ::]
                    color2 = npz["colors2"][0, ::]
                    pred_xyz = npz["pred_xyz"][0, ::]
                    mask_gt_pc = npz["mask_gt_pc"][0]
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_gt1 = npz["mask_gt1"][0]

                # 使用ransac算法进行整体位移变换
                ransac_pred_mask1 = pred_mask1 > 0.9
                ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((point1[ransac_pred_mask1])))
                ransac_pred_pcd = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector((pred_xyz[ransac_pred_mask1])))
                ransac_index = np.arange(0, ransac_pred_mask1.sum(), 1)
                ransac_corres = o3d.utility.Vector2iVector(np.stack((ransac_index, ransac_index), axis=1))
                distance_threshold = 2
                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    ransac_source_cloud, ransac_pred_pcd, ransac_corres, distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.9)
                )
                ratation_point1 = np.dot(ransac_result.transformation,
                                         np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1).T).T[:, :3]
                # 使用Flow剔除
                pred_mask1 = pred_mask1 > 0.9
                flow = pred_xyz - ratation_point1
                flow_true = pred_xyz[pred_mask1] - ratation_point1[pred_mask1]
                flow_true_mean = flow_true.mean(axis=0)
                dot_products = np.dot(flow_true, flow_true_mean)  # 计算向量的点积
                cosine_angles = dot_products / (
                        np.linalg.norm(flow_true, axis=1) * np.linalg.norm(flow_true_mean))  # 计算余弦夹角
                angles_radians = np.degrees(np.arccos(cosine_angles))  # 计算角度（弧度）
                new_true = angles_radians < 90
                true_true_point = ratation_point1[pred_mask1][new_true]
                true_true_flow = flow[pred_mask1][new_true]
                true_true_flow = TrueFlowGaussianFilter(true_true_point, true_true_flow, 8, 5)
                true_true_color = color1[pred_mask1][new_true]

                true_mask_gt_pc = mask_gt_pc[pred_mask1][new_true]
                false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]),
                                             axis=0)
                false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
                false_mask_gt_pc = np.concatenate((mask_gt_pc[~pred_mask1], mask_gt_pc[pred_mask1][~new_true]), axis=0)
                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1][new_true]), axis=0)
                new_color = np.concatenate((false_color, true_true_color), axis=0)
                new_mask_gt_pc = np.concatenate((false_mask_gt_pc, true_mask_gt_pc), axis=0)

                post_train = True
                np.savez(save_path,
                         new_pred_point=new_pred_point,
                         new_color=new_color,
                         new_mask_gt_pc=new_mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
            except:
                post_train = False
                np.savez(save_path,
                         new_pred_point=pred_xyz,
                         new_color=color1,
                         new_mask_gt_pc=mask_gt_pc,
                         point2=point2,
                         color2=color2,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))

# 读取后处理的精度文件：
def CalTrueGaussianPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    npz_results = glob.glob(os.path.join(root_path, 'TrueGaussianFilterInter', "*.npz"))
    # npz_results = glob.glob(os.path.join(root_path, 'gaussian_post_process', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            new_pred_point = npz["new_pred_point"]
            new_color = npz["new_color"]
            new_mask_gt_pc = npz["new_mask_gt_pc"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            post_train = npz["post_train"]
            final_acc += Cal_Final_Hamlyn_result_np(new_pred_point, new_mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/TrueGaussianFilterInter.npy"
    np.save(save_path, final_acc)


def CalGaussianPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    npz_results = glob.glob(os.path.join(root_path, 'GaussianFilterInter', "*.npz"))
    # npz_results = glob.glob(os.path.join(root_path, 'gaussian_post_process', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            new_pred_point = npz["new_pred_point"]
            new_color = npz["new_color"]
            new_mask_gt_pc = npz["new_mask_gt_pc"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            post_train = npz["post_train"]
            final_acc += Cal_Final_Hamlyn_result_np(new_pred_point, new_mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/GaussianFilterInter.npy"
    np.save(save_path, final_acc)


def CalMostPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来，去除了对重叠区域的二次判定
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    npz_results = glob.glob(os.path.join(root_path, 'Inter', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            new_pred_point = npz["new_pred_point"]
            new_color = npz["new_color"]
            new_mask_gt_pc = npz["new_mask_gt_pc"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            post_train = npz["post_train"]
            final_acc += Cal_Final_Hamlyn_result_np(new_pred_point, new_mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/Inter.npy"
    np.save(save_path, final_acc)


def CalMostGaussinaPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来，去除了对重叠区域的二次判定
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    npz_results = glob.glob(os.path.join(root_path, 'GaussianInter', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            new_pred_point = npz["new_pred_point"]
            new_color = npz["new_color"]
            new_mask_gt_pc = npz["new_mask_gt_pc"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            post_train = npz["post_train"]
            final_acc += Cal_Final_Hamlyn_result_np(new_pred_point, new_mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/GaussianInter.npy"
    np.save(save_path, final_acc)


def CalPostProcessHamlynResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    npz_results = glob.glob(os.path.join(root_path, 'FilterInter', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            new_pred_point = npz["new_pred_point"]
            new_color = npz["new_color"]
            new_mask_gt_pc = npz["new_mask_gt_pc"]
            point2 = npz["point2"]
            color2 = npz["color2"]
            post_train = npz["post_train"]
            final_acc += Cal_Final_Hamlyn_result_np(new_pred_point, new_mask_gt_pc)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/FilterInter.npy"
    np.save(save_path, final_acc)


def DirectReadHanlynResultRead(root_path,round_num=2):
    """
    直接读取澹台标注数据集的精度评价
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    target_npz_path = root_path + "/EvalHamlyn.npz"
    print_data = np.load(target_npz_path)
    final_acc = print_data['final_acc']
    final_acc = np.around(final_acc,round_num)
    mask_acc = print_data['mask_acc']
    mask_acc = np.around(mask_acc,round_num)
    print("root_path is {}, mask acc is {}".format(root_path.split('/')[-2:], mask_acc))
    print("root_path is {}, EvalHamlyn final_acc is {}".format(root_path.split('/')[-2:], final_acc))

    # try:
    #     npy_name = "FilterInter.npy"
    #     target_npz_path = os.path.join(root_path, npy_name)
    #     print_data = np.load(target_npz_path)
    #     print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    # except:
    #     print("FilterInter is No implementation")
    # try:
    #     npy_name = "GaussianFilterInter.npy"
    #     target_npz_path = os.path.join(root_path, npy_name)
    #     print_data = np.load(target_npz_path)
    #     print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    # except:
    #     print("GaussianFilterInter is No implementation")

    try:
        npy_name = "GaussianInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.around(np.load(target_npz_path), round_num)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianInter is No implementation")
    # try:0
    #     npy_name = "Inter.npy"
    #     target_npz_path = os.path.join(root_path, npy_name)
    #     print_data = np.load(target_npz_path)
    #     print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    # except:
    #     print("Inter is No implementation")
    # try:
    #     npy_name = "TrueGaussianFilterInter.npy"
    #     target_npz_path = os.path.join(root_path, npy_name)
    #     print_data = np.load(target_npz_path)
    #     print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    # except:
    #     print("TrueGaussianFilterInter is No implementation")

def ShowFinalBestResult(root_path):
    """
    直接读取澹台标注数据集的精度评价
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    target_npz_path = root_path + "/EvalHamlyn.npz"
    print_data = np.load(target_npz_path)
    final_acc = print_data['final_acc']
    mask_acc = print_data['mask_acc']
    print("root_path is {}, mask acc is {}".format(root_path.split('/')[-2:], mask_acc))
    print("root_path is {}, EvalHamlyn final_acc is {}".format(root_path.split('/')[-2:], final_acc))
    try:
        npy_name = "GaussianInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianInter is No implementation")

if __name__ == "__main__":
    # 全都是权重文件
    root_path1 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path2 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path3 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path4 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation"
    # 低重叠率数据集
    root_path5 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path6 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path7 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path8 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_low_overlap"

    """
    EvalHamlyn.npz 直接无监督微调之后的结果 
    """
    # DirectCalReadHanlynResult(root_path1)
    # DirectCalReadHanlynResult(root_path2)
    # DirectCalReadHanlynResult(root_path3)
    # DirectCalReadHanlynResult(root_path4)
    # DirectCalReadHanlynResult(root_path5)
    # DirectCalReadHanlynResult(root_path6)
    # DirectCalReadHanlynResult(root_path7)
    # DirectCalReadHanlynResult(root_path8)

    """
    post_process 文件夹  --> FilterInter 文件夹  FilterInter.npy
    ①使用一个整体的位移向量进行滤波，率除掉一些异常点 
    ②滤波后的点记为 高置信点，使用高置信点进行插值
    """

    # 后处理操作
    # PostProcessHamlynResult(root_path1) # 后处理点云
    # PostProcessHamlynResult(root_path2) # 后处理点云
    # PostProcessHamlynResult(root_path3) # 后处理点云
    # PostProcessHamlynResult(root_path4) # 后处理点云
    # PostProcessHamlynResult(root_path5) # 后处理点云
    # PostProcessHamlynResult(root_path6) # 后处理点云
    # PostProcessHamlynResult(root_path7) # 后处理点云
    # PostProcessHamlynResult(root_path8) # 后处理点云
    # CalPostProcessHamlynResult(root_path1)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path2)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path3)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path4)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path5)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path6)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path7)  # 评价后处理的点云
    # CalPostProcessHamlynResult(root_path8)  # 评价后处理的点云

    """
    gaussian_post_process文件夹 --> GaussianFilterInter 文件夹 GaussianFilterInter.npy
    ①首先使用高斯滤波函数进行滤波
    ②使用一个整体的位移向量进行滤波，率除掉一些异常点 
    ③滤波后的点记为 高置信点，使用高置信点进行插值
    """
    # GaussianPostProcessHamlynResult(root_path1)
    # CalGaussianPostProcessHamlynResult(root_path1)

    """
    true_gaussian_post_process 文件夹， TrueGaussianFilterInter.npy
    ①首先使用高斯滤波函数对于重叠区域进行滤波
    ②使用一个整体的位移向量进行滤波，率除掉一些异常点 
    ③滤波后的点记为 高置信点，使用高置信点进行插值
    """
    # TrueGaussianPostProcessHamlynResult(root_path1)
    # CalTrueGaussianPostProcessHamlynResult(root_path1)

    """
    most_post_process 文件夹 ---> Inter文件夹  Inter.npy
    ①使用pred mask估计出处于重叠区域的点，然后进行插值
    """
    # PostMostProcessHamlynResult(root_path1)
    # PostMostProcessHamlynResult(root_path2)
    # PostMostProcessHamlynResult(root_path3)
    # PostMostProcessHamlynResult(root_path4)
    # PostMostProcessHamlynResult(root_path5)
    # PostMostProcessHamlynResult(root_path6)
    # PostMostProcessHamlynResult(root_path7)
    # PostMostProcessHamlynResult(root_path8)
    # CalMostPostProcessHamlynResult(root_path1)
    # CalMostPostProcessHamlynResult(root_path2)
    # CalMostPostProcessHamlynResult(root_path3)
    # CalMostPostProcessHamlynResult(root_path4)
    # CalMostPostProcessHamlynResult(root_path5)
    # CalMostPostProcessHamlynResult(root_path6)
    # CalMostPostProcessHamlynResult(root_path7)
    # CalMostPostProcessHamlynResult(root_path8)
    """
    most_gaussian_post_process 文件夹  --> GaussianInter 文件夹 GaussianInter.npy
    ①首先使用高斯滤波函数进行滤波
    ②使用pred mask估计出处于重叠区域的点，然后进行插值
    """
    GaussianPostMostProcessHamlynResult(root_path1)
    # GaussianPostMostProcessHamlynResult(root_path2)
    # GaussianPostMostProcessHamlynResult(root_path3)
    # GaussianPostMostProcessHamlynResult(root_path4)
    # GaussianPostMostProcessHamlynResult(root_path5)
    # GaussianPostMostProcessHamlynResult(root_path6)
    # GaussianPostMostProcessHamlynResult(root_path7)
    # GaussianPostMostProcessHamlynResult(root_path8)
    CalMostGaussinaPostProcessHamlynResult(root_path1)
    # CalMostGaussinaPostProcessHamlynResult(root_path2)
    # CalMostGaussinaPostProcessHamlynResult(root_path3)
    # CalMostGaussinaPostProcessHamlynResult(root_path4)
    # CalMostGaussinaPostProcessHamlynResult(root_path5)
    # CalMostGaussinaPostProcessHamlynResult(root_path6)
    # CalMostGaussinaPostProcessHamlynResult(root_path7)
    # CalMostGaussinaPostProcessHamlynResult(root_path8)

    #
    # paths = [root_path1, root_path2, root_path3, root_path4]
    # for path in paths:
    #     DirectReadHanlynResultRead(path)  # 读取所有结果
    #     print("===============================")
    paths = [root_path5,  root_path6, root_path7, root_path8]
    for path in paths:
        # 最终读取所有结果
        DirectReadHanlynResultRead(path)  # 读取所有结果
        print("===============================")
