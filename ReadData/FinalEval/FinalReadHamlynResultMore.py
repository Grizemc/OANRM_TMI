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

"""
FinalReadHamlynResult的后续文件，增加了新的消融实验，以增加结果图
"""

def Gaussian_ceshidange(root_path):
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
    results_save_path = os.path.join(root_path, "GaussianInter_ceshi")
    io = FinalReadHamlynResultIOStream(root_path + '/GaussianInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]))

    file = npz_results[361]
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
    mask_acc = 0.
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
    mask_acc = mask_acc * 100
    mask_acc = mask_acc / len(npz_results)
    print("final_acc is {}, mask acc is {}".format(final_acc, mask_acc))
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


def DirectReadHanlynResultRead(root_path, round_num=6):
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
    final_acc = np.around(final_acc, round_num)
    mask_acc = print_data['mask_acc']
    mask_acc = np.around(mask_acc, round_num)
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
    # try:
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
    # target_npz_path = root_path + "/EvalHamlyn_last.npz"
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


def ShowFinalBestResult_gaussian(root_path):
    """
    直接读取澹台标注数据集的精度评价
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    # target_npz_path = root_path + "/GaussianInter.npy"
    # print_data = np.load(target_npz_path)
    # final_acc = print_data['final_acc']
    # mask_acc = print_data['mask_acc']
    # print("root_path is {}, mask acc is {}".format(root_path.split('/')[-2:], mask_acc))
    # print("root_path is {}, EvalHamlyn final_acc is {}".format(root_path.split('/')[-2:], final_acc))
    try:
        npy_name = "GaussianInterlast.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianInter is No implementation")


if __name__ == "__main__":
    # 80重叠率的数据集
    root_path1 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path2 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path3 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path4 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_80"
    # 85 重叠率数据集
    root_path5 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path6 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path7 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path8 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_85"
    # 　90 重叠率的数据集
    root_path9 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path10 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path11 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path12 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_90"
    root_path13 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path14 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192/Hamlyn_DircetTest_no_rotation_95"
    root_path15 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_low_overlap"

    # 95
    # root_path16 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_90_4/fpfh_Post_Train_Hamlyn_no_rotation_95"

    root_path17 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_80/fpfh_Post_Train_Hamlyn_no_rotation_75_datiao_13"
    # root_path17 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation"
    # root_path17 = r"/home/szm/Paconv_730/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_75_shixiong"
    # root_path18 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path18 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"

    # 80数据集训练
    root_path19 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_80/Hamlyn_DircetTest_no_rotation_95"

    # 65数据集训练
    root_path20 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_65/Hamlyn_DircetTest_no_rotation_95"

    # 55数据集训练
    root_path21 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_55/Hamlyn_DircetTest_no_rotation_75"

    # 35数据集训练
    root_path22 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_35/Hamlyn_DircetTest_no_rotation_75"

    # no_fuse
    root_path23 = r"/home/szm/Paconv_730/checkpoints/Zall_no_fuse/Hamlyn_DircetTest_no_rotation_75"

    # focal_loss1
    root_path24 = r"/home/szm/Paconv_730/checkpoints/Zall_focal_loss1/Hamlyn_DircetTest_no_rotation_85"

    root_path25 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Post_Train_Hamlyn_no_rotation_95_datiao_qiuqiu"

    # ShowFinalBestResult(root_path16)
    # ShowFinalBestResult_gaussian(root_path16)
    # DirectCalReadHanlynResult(root_path17)
    # DirectCalReadHanlynResult(root_path2)
    # DirectCalReadHanlynResult(root_path3)
    # DirectCalReadHanlynResult(root_path17)
    # DirectCalReadHanlynResult(root_path15)
    # DirectCalReadHanlynResult(root_path6)
    # DirectCalReadHanlynResult(root_path7)
    # DirectCalReadHanlynResult(root_path8)
    # DirectCalReadHanlynResult(root_path9)
    # DirectCalReadHanlynResult(root_path10)
    # DirectCalReadHanlynResult(root_path11)
    # DirectCalReadHanlynResult(root_path12)
    #
    # """
    # most_gaussian_post_process 文件夹  --> GaussianInter 文件夹 GaussianInter.npy
    # ①首先使用高斯滤波函数进行滤波
    # ②使用pred mask估计出处于重叠区域的点，然后进行插值
    # """
    # GaussianPostMostProcessHamlynResult(root_path16)
    # Gaussian_ceshidange(root_path17)

    GaussianPostMostProcessHamlynResult(root_path25)

# GaussianPostMostProcessHamlynResult(root_path3)
# GaussianPostMostProcessHamlynResult(root_path4)
# GaussianPostMostProcessHamlynResult(root_path5)
# GaussianPostMostProcessHamlynResult(root_path6)
# GaussianPostMostProcessHamlynResult(root_path7)
# GaussianPostMostProcessHamlynResult(root_path8)
# GaussianPostMostProcessHamlynResult(root_path9)
# GaussianPostMostProcessHamlynResult(root_path10)
# GaussianPostMostProcessHamlynResult(root_path11)
# GaussianPostMostProcessHamlynResult(root_path12)
# CalMostGaussinaPostProcessHamlynResult(root_path17)
# CalMostGaussinaPostProcessHamlynResult(root_path2)
# CalMostGaussinaPostProcessHamlynResult(root_path3)
# CalMostGaussinaPostProcessHamlynResult(root_path4)
# CalMostGaussinaPostProcessHamlynResult(root_path5)
# CalMostGaussinaPostProcessHamlynResult(root_path6)
# CalMostGaussinaPostProcessHamlynResult(root_path7)
# CalMostGaussinaPostProcessHamlynResult(root_path17)
# CalMostGaussinaPostProcessHamlynResult(root_path9)
# CalMostGaussinaPostProcessHamlynResult(root_path10)
# CalMostGaussinaPostProcessHamlynResult(root_path11)
# CalMostGaussinaPostProcessHamlynResult(root_path12)

# paths = [root_path1, root_path2, root_path3, root_path4]
# for path in paths:
#     DirectReadHanlynResultRead(path)  # 读取所有结果
#     print("===============================")
# paths = [root_path5,  root_path6, root_path7, root_path8]
# for path in paths:
#     DirectReadHanlynResultRead(path)  # 读取所有结果
#     print("===============================")
# paths = [root_path9,  root_path10, root_path11, root_path12]
# for path in paths:
#     DirectReadHanlynResultRead(path)  # 读取所有结果
#     print("===============================")

