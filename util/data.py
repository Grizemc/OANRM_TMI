#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/22 16:04
# @Author  : 沈子明
# @File    : medicine_data.py
# @Software: PyCharm
import copy
import glob
import os
import numpy as np
import open3d
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def rotation_augment():
    # generate random rotation matrix
    rot_source = Rotation.random().as_matrix()
    rot_target = Rotation.random().as_matrix()
    return rot_source, rot_target


def augmentation_onlyRTcS_data(pos1, pos2, ground_truth):
    # random rotation
    rot_source, rot_target = rotation_augment()
    # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
    pos1 = np.einsum('ij,kj->ki', rot_source, pos1).astype('float32')
    pos2 = np.einsum('ij,kj->ki', rot_target, pos2).astype('float32')
    ground_truth = np.einsum('ij,kj->ki', rot_target, ground_truth).astype('float32')

    # random offset
    offset1 = np.random.rand(1, 3)
    offset2 = np.random.rand(1, 3)
    pos1 += offset1
    pos2 += offset2
    ground_truth += offset2

    # random scale
    scale_coefficient1 = np.random.randint(1, 100) / 10  # 0.1-10
    pos1 = pos1 * scale_coefficient1
    return pos1, pos2, ground_truth


def augmentation_onlyRT_data(pos1, pos2, ground_truth):
    # random rotation
    rot_source = Rotation.random().as_matrix()
    rot_target = Rotation.random().as_matrix()
    # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
    pos1 = np.einsum('ij,kj->ki', rot_source, pos1).astype('float32')
    pos2 = np.einsum('ij,kj->ki', rot_target, pos2).astype('float32')
    ground_truth = np.einsum('ij,kj->ki', rot_target, ground_truth).astype('float32')

    # random offset
    offset1 = np.random.rand(1, 3)
    offset2 = np.random.rand(1, 3)
    pos1 += offset1
    pos2 += offset2
    ground_truth += offset2
    return pos1, pos2, ground_truth


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


class HumanMarkDataSingleFilterFpfh(Dataset):
    def __init__(self, root="/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/", sample_num="None",
                 mode="multi", fpfh=False, normalize=True):
        self.root = root
        self.fpfh = fpfh
        self.normalize = normalize

        self.data_path = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))

        self.fpfh_path = glob.glob(os.path.join(self.root, "test", "fpfh", '*.npz'))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))
        if sample_num == "None":
            self.sample_path = self.data_path
            self.fpfh_path = self.fpfh_path
        else:
            self.sample_path = [self.data_path[sample_num]]
            self.fpfh_path = [self.fpfh_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def __getitem__(self, index):
        fn = self.sample_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            points1 = fp["points1"].astype('float32')
            points2 = fp["points2"].astype('float32')
            colors2 = fp["colors2"].astype('float32') / 255
            colors1 = fp["colors1"].astype('float32') / 255
            label_xyz1 = fp["label_xyz1"].astype('float32')
            label_xyz2 = fp["label_xyz2"].astype('float32')
            label_color1 = fp["label_color1"].astype('float32') / 255
            label_color2 = fp["label_color2"].astype('float32') / 255
            fp.close()
        points1 = np.concatenate((points1, label_xyz1), axis=0)
        points2 = np.concatenate((points2, label_xyz2), axis=0)
        colors1 = np.concatenate((colors1, label_color1), axis=0)
        colors2 = np.concatenate((colors2, label_color2), axis=0)
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)

        fn = self.fpfh_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            matches_list0 = fp['matches_list0']
        if self.normalize:
            points2, relax_ratio2 = xyz_to_normalization(points2)
            points1, relax_ratio1 = xyz_to_normalization(points1)
            rot_source, rot_target = rotation_augment()
            # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
            # points1 = np.einsum('ij,kj->ki', rot_target, points1).astype('float32')
            pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
            pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
            mask_pseudo1 = matches_list0[:, 0]
            mask_pseudo2 = matches_list0[:, 1]
            gt_pseudo = np.array(pcd2.points)[matches_list0[:, 1]]
            return points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, label_xyz1.shape[0], \
                mask_pseudo1, mask_pseudo2, normal2, gt_pseudo

        else:
            pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
            pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
            mask_pseudo1 = matches_list0[:, 0]
            mask_pseudo2 = matches_list0[:, 1]
            gt_pseudo = np.array(pcd2.points)[matches_list0[:, 1]]
            return points1, points2, colors1, colors2, label_xyz1.shape[
                0], mask_pseudo1, mask_pseudo2, normal2, gt_pseudo

    def __len__(self):
        return len(self.sample_path)


class HamlynArtificialFPFH(Dataset):
    def __init__(self, root="/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/test",
                 fpfh_path="/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/fpft_file",
                 sample_num="None",
                 normalize=True, continuous=True):
        self.root = root
        self.normalize = normalize

        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))

        self.fpfh_path = glob.glob(os.path.join(fpfh_path, '*.npz'))
        self.fpfh_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.fpfh_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))

        if sample_num == "None":
            self.sample_path = self.data_path
            self.fpfh_path = self.fpfh_path
        elif continuous:
            self.sample_path = self.data_path[sample_num[0]:sample_num[1]]
            self.fpfh_path = self.fpfh_path[sample_num[0]:sample_num[1]]
        else:
            self.sample_path = [self.data_path[sample_num]]
            self.fpfh_path = [self.fpfh_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def __getitem__(self, index):
        fn = self.sample_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            points1 = fp["mask_point1"].astype('float32')
            colors1 = fp["mask_color1"].astype('float32')
            points2 = fp["mask_point2"].astype('float32')
            colors2 = fp["mask_color2"].astype('float32')
            mask_gt1 = fp["mask_gt1"]
            mask_gt2 = fp["mask_gt2"]
            mask_gt_pc = fp["mask_gt_pc"].astype('float32')
            fp.close()
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)

        fn = self.fpfh_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            matches_list0 = fp['matches_list0']
        if self.normalize:
            pass
        else:
            pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
            pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
            mask_pseudo1 = matches_list0[:, 0]
            mask_pseudo2 = matches_list0[:, 1]
            gt_pseudo = np.array(pcd2.points)[matches_list0[:, 1]]
            return points1, points2, colors1, colors2, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo, mask_gt1, mask_gt2, mask_gt_pc

    def __len__(self):
        return len(self.sample_path)


class HumanMarkDataSingleFilterFpfhSL(Dataset):
    def __init__(self, root="/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/", sample_num="None",
                 fpfh_path=r"/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/test/fpfh/",
                 mode="multi", fpfh=False, normalize=True, continuous=True):
        self.root = root
        self.fpfh = fpfh
        self.normalize = normalize

        self.data_path = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))

        self.fpfh_path = glob.glob(os.path.join(fpfh_path, '*.npz'))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
        self.fpfh_path.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))
        if sample_num == "None":
            self.sample_path = self.data_path
            self.fpfh_path = self.fpfh_path
        elif continuous:
            self.sample_path = self.data_path[sample_num[0]:sample_num[1]]
            self.fpfh_path = self.fpfh_path[sample_num[0]:sample_num[1]]
        else:
            self.sample_path = [self.data_path[sample_num]]
            self.fpfh_path = [self.data_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def __getitem__(self, index):
        fn = self.sample_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            points1 = fp["points1"].astype('float32')
            points2 = fp["points2"].astype('float32')
            colors2 = fp["colors2"].astype('float32') / 255
            colors1 = fp["colors1"].astype('float32') / 255
            label_xyz1 = fp["label_xyz1"].astype('float32')
            label_xyz2 = fp["label_xyz2"].astype('float32')
            label_color1 = fp["label_color1"].astype('float32') / 255
            label_color2 = fp["label_color2"].astype('float32') / 255
            fp.close()
        points1 = np.concatenate((points1, label_xyz1), axis=0)
        points2 = np.concatenate((points2, label_xyz2), axis=0)
        colors1 = np.concatenate((colors1, label_color1), axis=0)
        colors2 = np.concatenate((colors2, label_color2), axis=0)
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)

        fn = self.fpfh_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp)
            matches_list0 = fp['matches_list0']
        if self.normalize:
            points2, relax_ratio2 = xyz_to_normalization(points2)
            points1, relax_ratio1 = xyz_to_normalization(points1)
            rot_source, rot_target = rotation_augment()
            # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
            # points1 = np.einsum('ij,kj->ki', rot_target, points1).astype('float32')
            pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
            pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
            mask_pseudo1 = matches_list0[:, 0]
            mask_pseudo2 = matches_list0[:, 1]
            gt_pseudo = np.array(pcd2.points)[matches_list0[:, 1]]
            return points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, label_xyz1.shape[0], \
                mask_pseudo1, mask_pseudo2, normal2, gt_pseudo

        else:
            pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
            pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
            mask_pseudo1 = matches_list0[:, 0]
            mask_pseudo2 = matches_list0[:, 1]
            gt_pseudo = np.array(pcd2.points)[matches_list0[:, 1]]
            return points1, points2, colors1, colors2, label_xyz1.shape[
                0], mask_pseudo1, mask_pseudo2, normal2, gt_pseudo

    def __len__(self):
        return len(self.sample_path)


class HumanMarkDataSingle(Dataset):
    def __init__(self, root='/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmNpz', type="train", sample_num="None",
                 mode="multi", fpfh=False, normalize=True, continuous=False):
        self.root = root
        self.mode = mode
        self.fpfh = fpfh
        self.normalize = normalize
        if type == "train":
            self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        else:
            self.data_path = glob.glob(os.path.join(self.root, "test", '*.npz'))
            # self.data_path = glob.glob(os.path.join(self.root, "test", 'dataset7_keyframe1*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('label')[-1].split('.')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('keyframe')[-1].split('_')[0]))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split('dataset')[-1].split('_')[0]))
        if sample_num == "None":
            self.sample_path = self.data_path
        elif continuous:
            self.sample_path = [self.data_path[sample_num[0]:sample_num[1]]]
        else:
            self.sample_path = [self.data_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def FhFHSinglePcd(self, source_cloud, target_cloud, distance_threshold):
        # 为两个点云计算FPFH特征
        radius_normal = 0.2  # 法线估计半径
        radius_feature = 0.25  # FPFH计算半径
        # FPFH距离阈值，控制匹配程度

        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            # o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
             ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        matches = np.asarray(result.correspondence_set)
        return matches[:, 0], matches[:, 1], np.array(target_cloud.points)[matches[:, 1]]

    def __getitem__(self, index):
        fn = self.sample_path[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            points1 = fp["points1"].astype('float32')
            points2 = fp["points2"].astype('float32')
            colors2 = fp["colors2"].astype('float32') / 255
            colors1 = fp["colors1"].astype('float32') / 255
            label_xyz1 = fp["label_xyz1"].astype('float32')
            label_xyz2 = fp["label_xyz2"].astype('float32')
            label_color1 = fp["label_color1"].astype('float32') / 255
            label_color2 = fp["label_color2"].astype('float32') / 255
            fp.close()
        points1 = np.concatenate((points1, label_xyz1), axis=0)
        points2 = np.concatenate((points2, label_xyz2), axis=0)
        colors1 = np.concatenate((colors1, label_color1), axis=0)
        colors2 = np.concatenate((colors2, label_color2), axis=0)
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)
        if self.normalize:
            points2, relax_ratio2 = xyz_to_normalization(points2)
            points1, relax_ratio1 = xyz_to_normalization(points1)
            rot_source, rot_target = rotation_augment()
            # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
            # points1 = np.einsum('ij,kj->ki', rot_target, points1).astype('float32')
            if self.fpfh:
                pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
                pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
                mask_pseudo1, mask_pseudo2, gt_pseudo = self.FhFHSinglePcd(pcd1, pcd2, 0.2)
                return points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, label_xyz1.shape[0], \
                    mask_pseudo1, mask_pseudo2, normal2, gt_pseudo
            else: \
                    return points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, normal2, label_xyz1.shape[0]
        else:
            if self.fpfh:
                pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
                pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
                mask_pseudo1, mask_pseudo2, gt_pseudo = self.FhFHSinglePcd(pcd1, pcd2, 0.2)
                return points1, points2, colors1, colors2, label_xyz1.shape[
                    0], mask_pseudo1, mask_pseudo2, normal2, gt_pseudo
            else: \
                    return points1, points2, colors1, colors2, normal2, label_xyz1.shape[0]

    def __len__(self):
        return len(self.sample_path)


# root='/big_data/szm/szm_MICCAI_Hamlyn/HuaXi/'
class HuaXiFpfh(Dataset):
    def __init__(self, root=r"/big_data/szm/szm_MICCAI_Hamlyn/HuaXiDownSample_new", sample_num="None",
                 mode="multi", fpfh=False, normalize=True, continuous=False, interval=50):
        self.root = root
        self.mode = mode
        self.fpfh = fpfh
        self.normalize = normalize
        print(os.path.exists(root))
        if not os.path.exists(root):
            print(cwz)
            print("{} does not exist".format(root))
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split(".")[0]))
        self.true_data_path = []
        for index in range(0, len(self.data_path) - interval, 1):
            fn1 = self.data_path[index]
            fn2 = self.data_path[index + interval]
            if os.path.exists(fn1) and os.path.exists(fn2):
                self.true_data_path.append([fn1, fn2])

        if sample_num == "None":
            self.sample_path = self.true_data_path
        elif continuous:
            self.sample_path = [self.true_data_path[sample_num[0]:sample_num[1]]]
        else:
            self.sample_path = [self.true_data_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def FhFHSinglePcd(self, source_cloud, target_cloud, distance_thresholds):
        def teturn_color(result, source_cloud, target_cloud):
            matches = np.asarray(result.correspondence_set)
            if len(matches) == 0:
                color_error = 2238
            else:
                idx_source = matches[:, 0]
                idx_target = matches[:, 1]
                color_source = np.array(source_cloud.colors)[idx_source]
                color_target = np.array(target_cloud.colors)[idx_target]
                color_error = np.mean(np.linalg.norm((color_target - color_source), axis=1))
            return color_error

        # 　在非刚性数据集上不好用
        # 为两个点云计算FPFH特征
        radius_normal = 2  # 法线估计半径
        radius_feature = 3  # FPFH计算半径

        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=200))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=200))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        matches_list = []
        transformations_list = []
        color_errors_list = []
        fitness_list = []
        inlier_rmse_list = []
        for distance_threshold in distance_thresholds:
            need_fpfh = True
            while need_fpfh:
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_cloud, target_cloud, source_fpfh, target_fpfh, True, distance_threshold,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                    # o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9),
                     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
                )
                if len(np.asarray(result.correspondence_set)) != 0 and np.max(
                        np.abs(R.from_matrix(result.transformation[:3, :3].copy()).as_euler("xyz", degrees=True))) < 45:
                    need_fpfh = False

            fitness_list.append(result.fitness)
            inlier_rmse_list.append(result.inlier_rmse)
            matches = np.asarray(result.correspondence_set)
            transformations_list.append(np.asarray(result.transformation))
            color1_temp = np.array(source_cloud.colors)
            color2_temp = np.array(target_cloud.colors)
            idx_source = matches[:, 0]
            idx_target = matches[:, 1]
            color_source = np.array(source_cloud.colors)[idx_source]
            color_target = np.array(target_cloud.colors)[idx_target]
            color_error = np.mean(np.linalg.norm((color_target - color_source), axis=1))
            color_errors_list.append(color_error)
            matches_list.append(matches)
        o3d.visualization.draw_geometries([source_cloud, target_cloud])
        color_errors_list = np.array(color_errors_list)
        fitness_list = np.array(fitness_list)
        inlier_rmse_list = np.array(inlier_rmse_list)
        transformations_list = np.array(transformations_list)

        index = np.argsort(-fitness_list)
        color_errors_list = color_errors_list[index]
        fitness_list = fitness_list[index]
        inlier_rmse_list = inlier_rmse_list[index]
        transformations_list = transformations_list[index]
        matches_list = [matches_list[i] for i in index]
        distance_thresholds = [distance_thresholds[i] for i in index]
        return matches[:, 0], matches[:, 1], np.array(target_cloud.points)[matches[:, 1]]

    def __getitem__(self, index):
        fn1, fn2 = self.sample_path[index]
        with open(fn1, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            points1 = fp["point1"].astype('float32')
            colors1 = fp["color1"].astype('float32')
            if self.root == '/big_data/szm/szm_MICCAI_Hamlyn/HuaXi':
                result = np.concatenate((points1, colors1), axis=1)
                mask = result[:, 2] < 10
                result_filtered = result[~mask]
                points1 = result_filtered[:, :3]
                colors1 = result_filtered[:, 3:]
        fp.close()
        with open(fn2, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            points2 = fp["point1"].astype('float32')
            colors2 = fp["color1"].astype('float32')
            if self.root == '/big_data/szm/szm_MICCAI_Hamlyn/HuaXi':
                result = np.concatenate((points2, colors2), axis=1)
                mask = result[:, 2] < 10
                result_filtered = result[~mask]
                points2 = result_filtered[:, :3]
                colors2 = result_filtered[:, 3:]
        fp.close()
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)
        if self.normalize:
            points2, relax_ratio2 = xyz_to_normalization(points2)
            points1, relax_ratio1 = xyz_to_normalization(points1)
            rot_source, rot_target = rotation_augment()
            # pos1 = np.matmul(rot_source, pos1.T).T.astype('float32')  cpu too high
            # points1 = np.einsum('ij,kj->ki', rot_target, points1).astype('float32')
            if self.fpfh:
                pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
                pcd1.colors = o3d.pybind.utility.Vector3dVector(colors1)
                pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
                pcd2.colors = o3d.pybind.utility.Vector3dVector(colors2)
                mask_pseudo1, mask_pseudo2, gt_pseudo = self.FhFHSinglePcd(pcd1, pcd2, [4])
                return (points1, points2, colors1, colors2, relax_ratio1, relax_ratio2,
                        mask_pseudo1, mask_pseudo2, normal2, gt_pseudo)
            else: \
                    return points1, points2, colors1, colors2, relax_ratio1, relax_ratio2, normal2
        else:
            if self.fpfh:
                pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
                pcd1.colors = o3d.pybind.utility.Vector3dVector(colors1)
                pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
                pcd2.colors = o3d.pybind.utility.Vector3dVector(colors2)
                mask_pseudo1, mask_pseudo2, gt_pseudo = self.FhFHSinglePcd(pcd1, pcd2, [4])
                return points1, points2, colors1, colors2, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo
            else: \
                    return points1, points2, colors1, colors2, normal2

    def __len__(self):
        return len(self.sample_path)


class HuaXiFpfhContinuous(Dataset):
    def __init__(self, temp_result, root='/big_data/szm/szm_MICCAI_Hamlyn/continuous_HuaXi/'):
        self.root = root
        self.temp_result = temp_result
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('/')[-1].split(".")[0]))
        self.sample_path = []
        for index in range(0, len(self.data_path) - 1, 1):
            fn1 = self.data_path[index]
            fn2 = self.data_path[index + 1]
            if os.path.exists(fn1) and os.path.exists(fn2):
                self.sample_path.append([fn1, fn2])
        self.len = len(self.sample_path)
        self.cache = {}

    def FhFHSinglePcd(self, source_cloud, target_cloud, distance_thresholds):
        def teturn_color(result, source_cloud, target_cloud):
            matches = np.asarray(result.correspondence_set)
            if len(matches) == 0:
                color_error = 2238
            else:
                idx_source = matches[:, 0]
                idx_target = matches[:, 1]
                color_source = np.array(source_cloud.colors)[idx_source]
                color_target = np.array(target_cloud.colors)[idx_target]
                color_error = np.mean(np.linalg.norm((color_target - color_source), axis=1))
            return color_error

        # 　在非刚性数据集上不好用
        # 为两个点云计算FPFH特征
        radius_normal = 2  # 法线估计半径
        radius_feature = 3  # FPFH计算半径

        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=200))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=200))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        matches_list = []
        transformations_list = []
        color_errors_list = []
        fitness_list = []
        inlier_rmse_list = []
        for distance_threshold in distance_thresholds:
            need_fpfh = True
            while need_fpfh:
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_cloud, target_cloud, source_fpfh, target_fpfh, True, distance_threshold,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                    # o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9),
                     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
                )
                if len(np.asarray(result.correspondence_set)) != 0 and np.max(
                        np.abs(R.from_matrix(result.transformation[:3, :3].copy()).as_euler("xyz", degrees=True))) < 45:
                    need_fpfh = False

            fitness_list.append(result.fitness)
            inlier_rmse_list.append(result.inlier_rmse)
            matches = np.asarray(result.correspondence_set)
            transformations_list.append(np.asarray(result.transformation))
            color1_temp = np.array(source_cloud.colors)
            color2_temp = np.array(target_cloud.colors)
            idx_source = matches[:, 0]
            idx_target = matches[:, 1]
            color_source = np.array(source_cloud.colors)[idx_source]
            color_target = np.array(target_cloud.colors)[idx_target]
            color_error = np.mean(np.linalg.norm((color_target - color_source), axis=1))
            color_errors_list.append(color_error)
            matches_list.append(matches)
        o3d.visualization.draw_geometries([source_cloud, target_cloud])
        color_errors_list = np.array(color_errors_list)
        fitness_list = np.array(fitness_list)
        inlier_rmse_list = np.array(inlier_rmse_list)
        transformations_list = np.array(transformations_list)

        index = np.argsort(-fitness_list)
        color_errors_list = color_errors_list[index]
        fitness_list = fitness_list[index]
        inlier_rmse_list = inlier_rmse_list[index]
        transformations_list = transformations_list[index]
        matches_list = [matches_list[i] for i in index]
        distance_thresholds = [distance_thresholds[i] for i in index]
        return matches[:, 0], matches[:, 1], np.array(target_cloud.points)[matches[:, 1]]

    def __getitem__(self, index):
        fn1, fn2 = self.sample_path[index]
        if index != 0:
            fn1 = self.temp_result + "/index_{}.npz".format(index - 1)
        with open(fn1, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            points1 = fp["point1"].astype('float32')
            colors1 = fp["color1"].astype('float32')
        fp.close()
        with open(fn2, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            points2 = fp["point1"].astype('float32')
            colors2 = fp["color1"].astype('float32')
        fp.close()
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.estimate_normals()
        normal2 = np.array(pcd2.normals).astype(np.float32)
        pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
        pcd1.colors = o3d.pybind.utility.Vector3dVector(colors1)
        pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points2))
        pcd2.colors = o3d.pybind.utility.Vector3dVector(colors2)
        mask_pseudo1, mask_pseudo2, gt_pseudo = self.FhFHSinglePcd(pcd1, pcd2, [4])
        return points1, points2, colors1, colors2, mask_pseudo1, mask_pseudo2, normal2, gt_pseudo

    def __len__(self):
        return len(self.sample_path)


class PostTrainData(Dataset):
    def __init__(self, root='/big_data/szm/PostTrain/szm_MICCAI_Hamlyn/Hamlyn', sample_num=50, interval=50):
        self.root = root
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        if self.root == '/big_data/szm/PostTrain/Hamlyn':
            self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
            self.data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
        elif self.root == '/big_data/szm/PostTrain/HmalynConsecutive':
            self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
            self.data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
        elif self.root == "/big_data/szm/szm_MICCAI_Hamlyn/HuaXi/":
            self.data_path.sort(key=lambda x: int(x.split('/')[-1].split(".")[0]))
        self.true_data_path = []
        for index in range(len(self.data_path) - interval):
            fn1 = self.data_path[index]
            fn2 = self.data_path[index + interval]
            if os.path.exists(fn1) and os.path.exists(fn2):
                self.true_data_path.append([fn1, fn2])
        if sample_num == "None":
            self.sample_path = self.true_data_path[::10]
        else:
            self.sample_path = [self.true_data_path[sample_num]]
        self.len = len(self.sample_path)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            mask_point1, mask_color1, mask_point2, mask_color2, normal1, normal2 = self.cache[index]
        else:
            fn1, fn2 = self.sample_path[index]
            with open(fn1, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point1 = fp["point1"].astype('float32')
                mask_color1 = fp["color1"].astype('float32')
                pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point1))
                pcd1.estimate_normals()
                normal1 = np.array(pcd1.normals).astype(np.float32)
            fp.close()
            with open(fn2, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point2 = fp["point1"].astype('float32')
                mask_color2 = fp["color1"].astype('float32')
                pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(mask_point2))
                pcd2.estimate_normals()
                normal2 = np.array(pcd2.normals).astype(np.float32)
            fp.close()
            self.cache[index] = (mask_point1, mask_color1, mask_point2, mask_color2, normal1, normal2)
        mask_point2, relax_ratio = xyz_to_normalization(mask_point2)
        mask_point1, _ = xyz_to_normalization(mask_point1)
        return mask_point1, mask_color1, mask_point2, mask_color2, relax_ratio, normal1, normal2

    def __len__(self):
        return len(self.sample_path)


class PostTrainSpecialData(PostTrainData):
    def __init__(self, root='/big_data/szm/PostTrain/Hamlyn', sample_num=50):
        self.root = root
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-2]))
        self.data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
        self.true_data_path = []
        for index in range(0, len(self.data_path) - 1, 2):
            fn1 = self.data_path[index]
            fn2 = self.data_path[index + 1]
            if os.path.exists(fn1) and os.path.exists(fn2):
                self.true_data_path.append([fn1, fn2])
        self.sample_path = [self.true_data_path[sample_num]]
        self.len = len(self.sample_path)

        """
        [['/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_000_11976.npz',
  '/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_000_12001.npz'],
 ['/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_005_11976.npz',
  '/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_005_12001.npz'],
 ['/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_007_11976.npz',
  '/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_007_12001.npz'],
 ['/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_010_11976.npz',
  '/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_010_12001.npz'],
 ['/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_015_11976.npz',
  '/big_data/szm/H50000amlyn_mask_mutual/special_half/rectified08_015_12001.npz']]
        """


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


class ChromaticJitters(object):
    def __init__(self, p=0.95, std=0.005, **kwargs):
        self.p = p
        self.std = std

    def __call__(self, colors):
        if np.random.rand() < self.p:
            noise = np.random.randn(colors.shape[0], 3)
            noise *= self.std
            colors[:, :3] = np.clip(noise + colors[:, :3], 0, 1)
        return colors


class MaskMICCAIMutualTest(Dataset):
    def __init__(self, partition, num_points=57344, root='/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual', aug=True,
                 if_test_type=False, color_aug=False, average_coordinate=False):
        self.num_points = num_points
        self.root = root
        self.aug = aug
        self.partition = partition
        self.if_test_type = if_test_type
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, "train", '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.len = len(self.datapath)

        self.color_aug = color_aug
        if color_aug:
            self.color_aug_fun = ChromaticJitters()
        self.average_coordinate = average_coordinate

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn, 'rb') as fp:
            fp = np.load(fp, allow_pickle=True)
            mask_point1 = fp["mask_point1"].astype('float32')
            mask_color1 = fp["mask_color1"].astype('float32')
            mask_point2 = fp["mask_point2"].astype('float32')
            mask_color2 = fp["mask_color2"].astype('float32')
            mask_gt1 = fp["mask_gt1"]
            mask_gt2 = fp["mask_gt2"]
            mask_gt_pc = fp["mask_gt_pc"].astype('float32')
        if self.if_test_type:
            mask_point1_source, mask_point2, mask_gt_pc = augmentation_onlyRT_test_data(mask_point1, mask_point2,
                                                                                        mask_gt_pc)
        return mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc

    def __len__(self):
        return len(self.datapath)


class MaskMICCAIMutual(Dataset):
    def __init__(self, partition, num_points=57344, root='/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual', aug=True,
                 color_aug=False, average_coordinate=False):
        self.num_points = num_points
        self.root = root
        self.aug = aug
        self.partition = partition

        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, "train", '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.len = len(self.datapath)
        self.cache = {}
        self.color_aug = color_aug
        if color_aug:
            self.color_aug_fun = ChromaticJitters()
        self.average_coordinate = average_coordinate

    def __getitem__(self, index):
        if index in self.cache:
            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point1 = fp["mask_point1"].astype('float32')
                mask_color1 = fp["mask_color1"].astype('float32')
                mask_point2 = fp["mask_point2"].astype('float32')
                mask_color2 = fp["mask_color2"].astype('float32')
                mask_gt1 = fp["mask_gt1"]
                mask_gt2 = fp["mask_gt2"]
                mask_gt_pc = fp["mask_gt_pc"].astype('float32')
            self.cache[index] = (mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc)

        if self.partition == 'train':
            n1 = mask_point1.shape[0]
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]

            n2 = mask_point2.shape[0]
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
        else:
            n1 = mask_point1.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]

            n2 = mask_point2.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
        if self.aug:
            mask_point1_source, mask_point2, mask_gt_pc = augmentation_onlyRT_data(mask_point1, mask_point2, mask_gt_pc)
        if self.color_aug:
            mask_color2 = self.color_aug_fun(mask_color2)
            mask_color1 = self.color_aug_fun(mask_color1)
        if self.average_coordinate:
            mask_point1_mean = np.mean(mask_point1, axis=0)
            mask_point2_mean = np.mean(mask_point2, axis=0)
            mask_point1 = mask_point1 - mask_point1_mean
            mask_point2 = mask_point2 - mask_point2_mean
            mask_gt_pc = mask_gt_pc - mask_point2_mean
        else:
            pass
        return mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc

    def __len__(self):
        return len(self.datapath)


class MaskMICCAIMutualNormalized(Dataset):
    # num_points: 8192  color_aug: True
    # data_dir: /big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_mix  train_batch: 32
    # aug = True partition='train'
    def __init__(self, partition, num_points=8192, root='/big_data/szm/M6ICCAI_60000_Mask_18879_new', aug=True,
                 color_aug=False):
        self.num_points = num_points
        self.root = root
        print(root)
        self.aug = aug
        self.partition = partition
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, "train", '*.npz'))
        else:
            # 用 M8ICCAI_8192_Mask_19055_new_mutual中的test进行测试
            self.datapath = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.len = len(self.datapath)
        self.cache = {}
        self.color_aug = color_aug
        if color_aug:
            self.color_aug_fun = ChromaticJitters()

    def __getitem__(self, index):
        if index in self.cache:
            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point1 = fp["mask_point1"].astype('float32')
                mask_color1 = fp["mask_color1"].astype('float32')
                mask_point2 = fp["mask_point2"].astype('float32')
                mask_color2 = fp["mask_color2"].astype('float32')
                mask_gt1 = fp["mask_gt1"]
                mask_gt2 = fp["mask_gt2"]
                mask_gt_pc = fp["mask_gt_pc"].astype('float32')
            self.cache[index] = (mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc)

        if self.partition == 'train':
            n1 = mask_point1.shape[0]
            # 随机选取8192个点
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]

            n2 = mask_point2.shape[0]
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
        else:
            n1 = mask_point1.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]

            n2 = mask_point2.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
        if self.aug:
            mask_point1_source, mask_point2, mask_gt_pc = augmentation_onlyRT_data(mask_point1, mask_point2, mask_gt_pc)
        if self.color_aug:
            mask_color2 = self.color_aug_fun(mask_color2)
            mask_color1 = self.color_aug_fun(mask_color1)

        mask_point1, _ = xyz_to_normalization(mask_point1_source)
        nor_mask_point2, nor_ground_truth, relax_ratio = pcd2_gt_normalization(mask_point2, mask_gt_pc)
        return mask_point1, mask_color1, nor_mask_point2, mask_color2, mask_gt1, mask_gt2, nor_ground_truth, mask_gt_pc, relax_ratio, mask_point1_source

    def __len__(self):
        return len(self.datapath)


# 数据集的构造，以传入dataloader
class MaskMICCAIMutualNormalizedSpecial(Dataset):
    def __init__(self, partition, num_points=8192,
                 root='/big_data/szm/szm_MICCAI_Hamlyn/MSTargetSourceNpz_8192_train18133_val3668', aug=True,
                 color_aug=False):
        self.num_points = num_points
        self.root = root
        self.aug = aug
        self.partition = partition
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, "train", '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.len = len(self.datapath)
        self.cache = {}
        self.color_aug = color_aug
        if color_aug:
            self.color_aug_fun = ChromaticJitters()

    def __getitem__(self, index):
        if index in self.cache:
            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, \
                mask_gt2, mask_gt_pc, point1, color1, point2, color2, ground_truth = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point1 = fp["mask_point1"].astype('float32')
                mask_color1 = fp["mask_color1"].astype('float32')
                mask_point2 = fp["mask_point2"].astype('float32')
                mask_color2 = fp["mask_color2"].astype('float32')
                point1 = fp["point1"].astype('float32')
                color1 = fp["color1"].astype('float32')
                point2 = fp["point2"].astype('float32')
                color2 = fp["color2"].astype('float32')
                ground_truth = fp["ground_truth"].astype('float32')
                mask_gt1 = fp["mask_gt1"]
                mask_gt2 = fp["mask_gt2"]
                mask_gt_pc = fp["mask_gt_pc"].astype('float32')
            self.cache[index] = (
                mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc, point1, color1,
                point2,
                color2, ground_truth)

        if self.partition == 'train':
            random_point = np.random.rand()
            if random_point < 0.34:
                n1 = mask_point1.shape[0]
                sample_idx = np.random.choice(n1, self.num_points, replace=False)
                mask_point1 = mask_point1[sample_idx, :]
                mask_color1 = mask_color1[sample_idx, :]
                mask_gt_pc = mask_gt_pc[sample_idx, :]
                mask_gt1 = mask_gt1[sample_idx]

                n2 = mask_point2.shape[0]
                sample_idx = np.random.choice(n2, self.num_points, replace=False)
                mask_point2 = mask_point2[sample_idx, :]
                mask_color2 = mask_color2[sample_idx, :]
                mask_gt2 = mask_gt2[sample_idx]
            elif random_point < 0.68:
                n1 = point1.shape[0]
                sample_idx = np.random.choice(n1, self.num_points, replace=False)
                mask_point1 = point1[sample_idx, :]
                mask_color1 = color1[sample_idx, :]
                mask_gt_pc = ground_truth[sample_idx, :]
                mask_gt1 = np.ones_like(mask_point1[:, 0]).astype(bool)

                n2 = mask_point2.shape[0]
                sample_idx = np.random.choice(n2, self.num_points, replace=False)
                mask_point2 = mask_point2[sample_idx, :]
                mask_color2 = mask_color2[sample_idx, :]
                mask_gt2 = mask_gt2[sample_idx]
            else:
                n1 = mask_point1.shape[0]
                sample_idx = np.random.choice(n1, self.num_points, replace=False)
                mask_point1 = mask_point1[sample_idx, :]
                mask_color1 = mask_color1[sample_idx, :]
                mask_gt_pc = mask_gt_pc[sample_idx, :]
                mask_gt1 = mask_gt1[sample_idx]
                n2 = point2.shape[0]
                sample_idx = np.random.choice(n2, self.num_points, replace=False)
                mask_point2 = point2[sample_idx, :]
                mask_color2 = color2[sample_idx, :]
                mask_gt2 = np.ones_like(mask_point2[:, 0]).astype(bool)
        else:
            n1 = mask_point1.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]

            n2 = mask_point2.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
        if self.aug:
            mask_point1_source, mask_point2, mask_gt_pc = augmentation_onlyRT_data(mask_point1, mask_point2, mask_gt_pc)
        if self.color_aug:
            mask_color2 = self.color_aug_fun(mask_color2)
            mask_color1 = self.color_aug_fun(mask_color1)

        mask_point1, _ = xyz_to_normalization(mask_point1_source)
        # _, relax_ratio = mask_xyz_to_normalization(mask_point2, mask_gt2)
        # mask_point2, _ = xyz_to_normalization(mask_point2)
        mask_point2, relax_ratio = xyz_to_normalization(mask_point2)
        nor_ground_truth, _ = xyz_to_normalization(mask_gt_pc)
        return mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_ground_truth, mask_gt_pc, relax_ratio, mask_point1_source

    def __len__(self):
        return len(self.datapath)


class LargeMaskMICCAIMutualNormalized(Dataset):
    def __init__(self, partition, num_points=57344, root='/big_data/szm/M6ICCAI_60000_Mask_18879_new', aug=True):
        self.num_points = num_points
        self.root = root
        self.aug = aug
        self.partition = partition
        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root, "train", '*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "test", '*.npz'))
        self.len = len(self.datapath)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc, mask_weight_1, mask_weight_2 = \
                self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                fp = np.load(fp, allow_pickle=True)
                mask_point1 = fp["mask_point1"].astype('float32')
                mask_color1 = fp["mask_color1"].astype('float32')
                mask_point2 = fp["mask_point2"].astype('float32')
                mask_color2 = fp["mask_color2"].astype('float32')
                mask_gt1 = fp["mask_gt1"]
                mask_gt2 = fp["mask_gt2"]
                mask_gt_pc = fp["mask_gt_pc"].astype('float32')
                mask_weight_1 = fp["mask_weight_1"].astype('float32')
                mask_weight_2 = fp["mask_weight_2"].astype('float32')
            self.cache[index] = (
                mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc, mask_weight_1,
                mask_weight_2)

        if self.partition == 'train':
            n1 = mask_point1.shape[0]
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]
            mask_weight_1 = mask_weight_1[sample_idx]

            n2 = mask_point2.shape[0]
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
            mask_weight_2 = mask_weight_2[sample_idx]
        else:
            n1 = mask_point1.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n1, self.num_points, replace=False)
            mask_point1 = mask_point1[sample_idx, :]
            mask_color1 = mask_color1[sample_idx, :]
            mask_gt_pc = mask_gt_pc[sample_idx, :]
            mask_gt1 = mask_gt1[sample_idx]
            mask_weight_1 = mask_weight_1[sample_idx]

            n2 = mask_point2.shape[0]
            np.random.seed(10)
            sample_idx = np.random.choice(n2, self.num_points, replace=False)
            mask_point2 = mask_point2[sample_idx, :]
            mask_color2 = mask_color2[sample_idx, :]
            mask_gt2 = mask_gt2[sample_idx]
            mask_weight_2 = mask_weight_2[sample_idx]
        if self.aug:
            mask_point1_source, mask_point2, mask_gt_pc = augmentation_onlyRT_data(mask_point1, mask_point2, mask_gt_pc)
        mask_point1, _ = xyz_to_normalization(mask_point1_source)

        # _, relax_ratio = mask_xyz_to_normalization(mask_point2, mask_gt2)
        # mask_point2, _ = xyz_to_normalization(mask_point2)
        mask_point2, relax_ratio = xyz_to_normalization(mask_point2)

        nor_ground_truth, _ = xyz_to_normalization(mask_gt_pc)
        return mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_ground_truth, mask_gt_pc, relax_ratio, mask_point1_source, mask_weight_1, mask_weight_2

    def __len__(self):
        return len(self.datapath)


def mask_xyz_to_normalization(xyz_in, mask):
    xyz_in = xyz_in[mask]
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    len_x = x.max() - x.min()
    len_y = y.max() - y.min()
    len_z = z.max() - z.min()
    new_x = (((x - x.min()) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y.min()) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z.min()) / len_z - 0.5
    result = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return result, np.array([len_x, len_y, len_z, x.min(), y.min(), z.min()])


def xyz_to_normalization(xyz_in):
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    len_x = x.max() - x.min()
    len_y = y.max() - y.min()
    len_z = z.max() - z.min()
    new_x = (((x - x.min()) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y.min()) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z.min()) / len_z - 0.5
    result = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return result, np.array([len_x, len_y, len_z, x.min(), y.min(), z.min()])


def pcd2_gt_normalization(xyz_2, xyz_gt):
    x = xyz_2[:, 0].reshape(-1, 1)
    y = xyz_2[:, 1].reshape(-1, 1)
    z = xyz_2[:, 2].reshape(-1, 1)
    len_x = x.max() - x.min()
    len_y = y.max() - y.min()
    len_z = z.max() - z.min()
    x_min = x.min()
    y_min = y.min()
    z_min = z.min()
    new_x = (((x - x_min) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y_min) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z_min) / len_z - 0.5
    result1 = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)

    x = xyz_gt[:, 0].reshape(-1, 1)
    y = xyz_gt[:, 1].reshape(-1, 1)
    z = xyz_gt[:, 2].reshape(-1, 1)
    new_x = (((x - x_min) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y_min) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z_min) / len_z - 0.5
    result2 = np.concatenate((new_x, new_y, new_z), axis=1).reshape(-1, 3)
    return result1, result2, np.array([len_x, len_y, len_z, x_min, y_min, z_min])


if __name__ == "__main__":
    # test_loader = DataLoader(
    #     MaskNormalizedMICCAI(partition="train", num_points=57344, root="/big_data/szm/M6ICCAI_60000_Mask_18879_new/"),
    #     batch_size=20, shuffle=False, drop_last=False)
    test_loader = DataLoader(MaskMICCAIMutual(partition="train", num_points=57344),
                             batch_size=1)
    # test_loader = DataLoader(BigOrderMICCAI(),
    #                          num_workers=0, batch_size=2, shuffle=False, drop_last=False)
    for index, datas in enumerate(test_loader):
        pointxyzs1, pointxyzs2, colorss1, colorss2 = datas
        points1 = torch.cat([pointxyzs1, colorss1], dim=2)
        points2 = torch.cat([pointxyzs2, colorss2], dim=2)
        print(index)

    print('Hello world')
