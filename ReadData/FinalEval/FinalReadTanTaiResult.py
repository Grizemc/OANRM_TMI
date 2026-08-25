#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 20:06
# @Author  : 沈子明
# @File    : FinalReadTanTaiResult.py
# @Software: PyCharm
import glob
import os.path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import sys

# 评价澹台数据集的代码 tantai后处理的文件

class FinalReadTanTaiResultIOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        print("=================================")
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def DirectCalTanTaiResult(root_path):
    """


    """

    def Final_TanTai_result_np(pcd1, pcd2, in_eval_num):
        pcd1 = pcd1[-in_eval_num:, :]
        pcd2 = pcd2[-in_eval_num:, :]
        displace = np.linalg.norm(pcd1 - pcd2, axis=1)
        relax_error = [1, 2, 3, 4, 5]
        acc_list = []
        for error in relax_error:
            acc_list.append((displace < error).sum() / in_eval_num)
        acc_list.append(displace.mean())
        acc_list = np.array(acc_list)
        return acc_list

    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            point1 = npz["points1"][0, ::]
            point2 = npz["points2"][0, ::]
            color1 = npz["colors1"][0, ::]
            color2 = npz["colors2"][0, ::]
            pred_xyz = npz["pred_xyz"][0, ::]
            pred_mask1 = npz["pred_mask1"].squeeze()
            mask_sum = npz["mask_sum"][0]
            final_acc += Final_TanTai_result_np(pred_xyz, point2, mask_sum)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    # 坍台数据集精度
    save_path = root_path + "/EvalTanTai.npy"
    np.save(save_path, final_acc)


def PostProcessTanTaiResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    npz_results = glob.glob(os.path.join(root_path, 'npz_result', "*.npz"))
    results_save_path = os.path.join(root_path, "FilterInter")
    io = FinalReadTanTaiResultIOStream(root_path + '/FilterInter.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
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
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_sum = npz["mask_sum"][0]
                cal_error_num = np.arange(len(pred_mask1) - 1, -1, -1)
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
                false_point = np.concatenate((ratation_point1[~pred_mask1], ratation_point1[pred_mask1][~new_true]),
                                             axis=0)
                false_color = np.concatenate((color1[~pred_mask1], color1[pred_mask1][~new_true]), axis=0)
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
                false_cal_error_num = np.concatenate((cal_error_num[~pred_mask1], cal_error_num[pred_mask1][~new_true]),
                                                     axis=0)
                new_cal_error_num = np.concatenate((false_cal_error_num, cal_error_num[pred_mask1][new_true]), axis=0)
                new_sort_index = np.argsort(new_cal_error_num)
                sort_new_color = new_color[new_sort_index]
                sort_pred_point = new_pred_point[new_sort_index]
                label_color1 = sort_new_color[:mask_sum, :][::-1, :]
                color11 = color1[-mask_sum:, :]
                label_xyz1 = new_pred_point[new_sort_index][:mask_sum, :][::-1, :]
                label_xyz2 = point2[-mask_sum:, :]
                post_train = True
                np.savez(save_path,
                         label_xyz2=label_xyz2,
                         label_xyz1=label_xyz1,
                         sort_pred_point=sort_pred_point,
                         sort_new_color=sort_new_color,
                         post_train=post_train
                         )
            except:
                post_train = False
                label_xyz1 = pred_xyz[-5:, :]
                label_xyz2 = point2[-5:, :]
                sort_pred_point = pred_xyz
                sort_new_color = color1
                np.savez(save_path,
                         label_xyz2=label_xyz2,
                         label_xyz1=label_xyz1,
                         sort_pred_point=sort_pred_point,
                         sort_new_color=sort_new_color,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def MostPostProcessTanTaiResult(root_path, Gaussian=True):
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
    if Gaussian == True:
        results_save_path = os.path.join(root_path, "GaussianInter")
        io = FinalReadTanTaiResultIOStream(root_path + '/GaussianMostPostProcessTanTaiResult.log')
    else:
        results_save_path = os.path.join(root_path, "Inter")
        io = FinalReadTanTaiResultIOStream(root_path + '/MostPostProcessTanTaiResult.log')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
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
                    pred_mask1 = npz["pred_mask1"].squeeze()
                    mask_sum = npz["mask_sum"][0]
                if Gaussian == True:
                    pred_xyz = GaussianFilter(point1, pred_xyz, 8, 5)
                else:
                    pass
                cal_error_num = np.arange(len(pred_mask1) - 1, -1, -1)
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

                true_true_point = ratation_point1[pred_mask1]
                true_true_color = color1[pred_mask1]
                true_true_flow = flow[pred_mask1]
                false_point = ratation_point1[~pred_mask1]
                false_color = color1[~pred_mask1]

                # 使用Flow插值
                nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(true_true_point)
                distances, nearest_indices = nbrs.kneighbors(false_point)
                neigh_flow = true_true_flow[nearest_indices]
                result_flow = neigh_flow
                weight = 1 / distances
                result_flow = result_flow * (weight / weight.sum(axis=1, keepdims=True))[:, :, np.newaxis]
                result_point = false_point + result_flow.sum(axis=1)
                new_pred_point = np.concatenate((result_point, pred_xyz[pred_mask1]), axis=0)
                new_color = np.concatenate((false_color, true_true_color), axis=0)
                # 先是低置信度点，后是高置信度点
                # 我的猜想是，10，9，8，7，6，5，4，3，2，1
                # 然后再这些从大到小的点中按照预测点的索引进行排序，先低置信度，再高置信度，同时，两部分的
                # 两部分的点的数值，从大到小排序，最后由np.argsort(new_cal_error_num)取从小到大的索引时
                # 就是先从两部分的最后侧（或者说最右侧）开始取对应的索引位置的点坐标数值，
                # 最后组成的数组再由[::-1, :]翻转一下
                new_cal_error_num = np.concatenate((cal_error_num[~pred_mask1], cal_error_num[pred_mask1]), axis=0)
                new_sort_index = np.argsort(new_cal_error_num)
                sort_new_color = new_color[new_sort_index]
                sort_pred_point = new_pred_point[new_sort_index]
                label_color1 = sort_new_color[:mask_sum, :][::-1, :]
                color11 = color1[-mask_sum:, :]
                label_xyz1 = new_pred_point[new_sort_index][:mask_sum, :][::-1, :]
                label_xyz2 = point2[-mask_sum:, :]
                post_train = True
                np.savez(save_path,
                         label_xyz2=label_xyz2,
                         label_xyz1=label_xyz1,
                         sort_pred_point=sort_pred_point,
                         sort_new_color=sort_new_color,
                         post_train=post_train
                         )
            except:
                post_train = False
                label_xyz1 = pred_xyz[-mask_sum:, :]
                label_xyz2 = point2[-mask_sum:, :]
                sort_pred_point = pred_xyz
                sort_new_color = color1
                np.savez(save_path,
                         label_xyz2=label_xyz2,
                         label_xyz1=label_xyz1,
                         sort_pred_point=sort_pred_point,
                         sort_new_color=sort_new_color,
                         post_train=post_train
                         )
                io.cprint(" file {} is error.".format(file))


def CalPostProcessTanTaiResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    def Final_TanTai_result_np(pcd1, pcd2):

        displace = np.linalg.norm(pcd1 - pcd2, axis=1)
        relax_error = [1, 2, 3, 4, 5]
        acc_list = []
        for error in relax_error:
            acc_list.append((displace < error).sum() / pcd1.shape[0])
        acc_list.append(displace.mean())
        acc_list = np.array(acc_list)
        return acc_list

    npz_results = glob.glob(os.path.join(root_path, 'post_process', "*.npz"))
    # os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            label_xyz2 = npz["label_xyz2"]
            label_xyz1 = npz["label_xyz1"]
            # sort_pred_point = npz["sort_pred_point"]
            # sort_new_color = npz["sort_new_color"]
            final_acc += Final_TanTai_result_np(label_xyz2, label_xyz1)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/PostProcessEvalTanTai.npy"
    np.save(save_path, final_acc)


def CalMostPostProcessTanTaiResult(root_path, Gaussian):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    def Final_TanTai_result_np(pcd1, pcd2):

        displace = np.linalg.norm(pcd1 - pcd2, axis=1)
        relax_error = [1, 2, 3, 4, 5]
        acc_list = []
        for error in relax_error:
            acc_list.append((displace < error).sum() / pcd1.shape[0])
        acc_list.append(displace.mean())
        acc_list = np.array(acc_list)
        return acc_list

    if Gaussian == True:
        npz_results = os.path.join(root_path, "GaussianInter")
    else:
        npz_results = os.path.join(root_path, "Inter")
    npz_results = glob.glob(os.path.join(npz_results, "*.npz"))
    # os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            label_xyz2 = npz["label_xyz2"] # 标注点
            label_xyz1 = npz["label_xyz1"]
            # sort_pred_point = npz["sort_pred_point"]
            # sort_new_color = npz["sort_new_color"]
            final_acc += Final_TanTai_result_np(label_xyz2, label_xyz1)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    if Gaussian:
        save_path = root_path + "/GaussianInter.npy"
    else:
        save_path = root_path + "/Inter.npy"
    np.save(save_path, final_acc)


def CalPostProcessTanTaiResult(root_path):
    """
    后处理需要的文件，并存储起来
    Parameters
    ----------
    root_path

    Returns
    -------

    """

    def Final_TanTai_result_np(pcd1, pcd2):

        displace = np.linalg.norm(pcd1 - pcd2, axis=1)
        relax_error = [1, 2, 3, 4, 5]
        acc_list = []
        for error in relax_error:
            acc_list.append((displace < error).sum() / pcd1.shape[0])
        acc_list.append(displace.mean())
        acc_list = np.array(acc_list)
        return acc_list

    npz_results = glob.glob(os.path.join(root_path, "FilterInter", '*.npz'))

    # os.makedirs(results_save_path)
    npz_results.sort(key=lambda x: int(x.split('/')[-1].split('_')[-2].split('.')[0].split('_')[-1]))
    final_acc = 0.
    for file in npz_results:
        with np.load(file) as npz:
            label_xyz2 = npz["label_xyz2"]
            label_xyz1 = npz["label_xyz1"]
            # sort_pred_point = npz["sort_pred_point"]
            # sort_new_color = npz["sort_new_color"]
            final_acc += Final_TanTai_result_np(label_xyz2, label_xyz1)
    final_acc = final_acc * 100
    final_acc[-1] = final_acc[-1] / 100
    final_acc = final_acc / len(npz_results)
    print("final_acc is {}".format(final_acc))
    save_path = root_path + "/FilterInter.npy"
    np.save(save_path, final_acc)


def ShowFinalBestResult(root_path):
    """
    直接读取澹台标注数据集的精度评价
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    target_npz_path = root_path + "/EvalTanTai.npy"
    final_acc = np.load(target_npz_path)
    final_acc = np.around(final_acc,2)
    print("root_path is {}, EvalHamlyn final_acc is {}".format(root_path.split('/')[-2:], final_acc))

    try:
        npy_name = "GaussianInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print_data = np.around(print_data, 2)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianInter is No implementation")


def DirectReadResult(root_path):
    """
    直接读取澹台标注数据集的精度评价
    Parameters
    ----------
    root_path

    Returns
    -------

    """
    target_npz_path = root_path + "/EvalTanTai.npy"
    final_acc = np.load(target_npz_path)
    print("root_path is {}, EvalTanTai final_acc is {}".format(root_path.split('/')[-2:], final_acc))

    try:
        npy_name = "FilterInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("FilterInter is No implementation")
    try:
        npy_name = "GaussianFilterInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianFilterInter is No implementation")

    try:
        npy_name = "GaussianInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("GaussianInter is No implementation")
    try:
        npy_name = "Inter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("Inter is No implementation")
    try:
        npy_name = "TrueGaussianFilterInter.npy"
        target_npz_path = os.path.join(root_path, npy_name)
        print_data = np.load(target_npz_path)
        print("root_path is {}, {} final_acc is {}".format(root_path.split('/')[-2:], npy_name, print_data))
    except:
        print("TrueGaussianFilterInter is No implementation")


if __name__ == "__main__":
    # 无监督训练好的路径
    # root_path1 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness"
    root_path2 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Tantai_fitness"
    root_path3 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Tantai_fitness"
    root_path4 = r"/home/szm/Paconv/checkpoints/Zall/BiaoZhu_DircetTest"

    # root_path5 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Tantai_fitness_new1"
    root_path5 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new1"
    # root_path2 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new2"
    # root_path3 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new3"
    # root_path4 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new6"
    # root_path5 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new7"
    # root_path6 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Tantai_fitness_new3"
    # root_path5 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Tantai_fitness_new1"
       # 下面这个不太行
    root_path5 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Tantai_fitness_new13"
    # root_path5 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Tantai_fitness_new2"
    # root_path5 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/fpfh_Tantai_fitness_new1"
    # root_path5 = r"/home/szm/Paconv_730/checkpoints/Zall/fpfh_Tantai_fitness_new1"
    # root_path5 = r"/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_80/fpfh_Tantai_fitness_new3"

    """
     直接计算无监督后的精度 EvalTanTai.npy 直接无监督微调之后的结果 
     
    """
    # # DirectCalTanTaiResult(root_path1)
    DirectCalTanTaiResult(root_path5)
    # DirectCalTanTaiResult(root_path3)
    # DirectCalTanTaiResult(root_path4)
    # DirectCalTanTaiResult(root_path5)
    # DirectCalTanTaiResult(root_path6)

    """
    post_process 文件夹  --> FilterInter 文件夹  FilterInter.npy
    ①使用一个整体的位移向量进行滤波，率除掉一些异常点 
    ②滤波后的点记为 高置信点，使用高置信点进行插值
    """
    # PostProcessTanTaiResult(root_path5) # 后处理点云
    # PostProcessTanTaiResult(root_path2) # 后处理点云
    # PostProcessTanTaiResult(root_path5) # 后处理点云
    # PostProcessTanTaiResult(root_path6) # 后处理点云
    # CalPostProcessTanTaiResult(root_path5)  # 评价后处理的点云
    # CalPostProcessTanTaiResult(root_path6)  # 评价后处理的点云
    # CalPostProcessTanTaiResult(root_path1)  # 评价后处理的点云
    # CalPostProcessTanTaiResult(root_path2)  # 评价后处理的点云
    # CalPostProcessTanTaiResult(root_path3)  # 评价后处理的点云
    # CalPostProcessTanTaiResult(root_path4)  # 评价后处理的点云

    # GaussianPostProcessTanTaiResult(root_path1)

    """
    most_post_process 文件夹 ---> Inter 文件夹  Inter.npy
    ①使用pred mask估计出处于重叠区域的点，然后进行插值

    most_gaussian_post_process 文件夹  --> GaussianInter 文件夹 GaussianInter.npy
    ①首先使用高斯滤波函数进行滤波
    ②使用pred mask估计出处于重叠区域的点，然后进行插值
    """
    # MostPostProcessTanTaiResult(root_path5, Gaussian=True)
    # MostPostProcessTanTaiResult(root_path2,Gaussian=True )
    # MostPostProcessTanTaiResult(root_path3,Gaussian=True )
    # MostPostProcessTanTaiResult(root_path4,Gaussian=True )
    # MostPostProcessTanTaiResult(root_path1,Gaussian=False )
    # MostPostProcessTanTaiResult(root_path2,Gaussian=False )
    # MostPostProcessTanTaiResult(root_path3,Gaussian=False )
    # MostPostProcessTanTaiResult(root_path4,Gaussian=False )
    # CalMostPostProcessTanTaiResult(root_path5, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path2, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path3, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path4, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path1, Gaussian=False)
    # CalMostPostProcessTanTaiResult(root_path2, Gaussian=False)
    # CalMostPostProcessTanTaiResult(root_path3, Gaussian=False)
    # CalMostPostProcessTanTaiResult(root_path4, Gaussian=False)

    # MostPostProcessTanTaiResult(root_path5,Gaussian=True )
    # MostPostProcessTanTaiResult(root_path6,Gaussian=True )
    # MostPostProcessTanTaiResult(root_path5,Gaussian=False )
    # MostPostProcessTanTaiResult(root_path6,Gaussian=False )
    # CalMostPostProcessTanTaiResult(root_path5, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path6, Gaussian=True)
    # CalMostPostProcessTanTaiResult(root_path5, Gaussian=False)
    # CalMostPostProcessTanTaiResult(root_path6, Gaussian=False)


    '''
    paths = [root_path2, root_path3, root_path4, root_path5]
    for path in paths:
        ShowFinalBestResult(path)  # 读取所有结果
        print("===============================")
    # paths = [  root_path5] '''
