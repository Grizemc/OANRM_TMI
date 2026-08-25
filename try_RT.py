import argparse
import glob
import multiprocessing
import os
import numpy as np
import open3d
from scipy.spatial.transform import Rotation


def apply_rigid_transform(mask_point2,mask_gt_pc,point2,ground_truth):
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

    # 对点云应用旋转和位移  # mask_point2,mask_gt_pc,point2,ground_truth
    mask_point2 = np.dot(mask_point2, R.T) + t
    mask_gt_pc = np.dot(mask_gt_pc, R.T) + t
    point2 = np.dot(point2, R.T) + t
    ground_truth = np.dot(ground_truth, R.T) + t
    return mask_point2,mask_gt_pc,point2,ground_truth

# H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap
root = r'/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_80/test1'
data_paths = glob.glob(os.path.join(root, '*.npz'))
data_paths.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
data_paths.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
target_path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_80/test"
if not os.path.exists(target_path):
    os.mkdir(target_path)
else:
    pass

count = 0
for data_path in data_paths:
    fp = np.load(data_path)
    mask_point1 = fp["mask_point1"].astype('float32')
    mask_color1 = fp["mask_color1"].astype('float32')
    mask_point2 = fp["mask_point2"].astype('float32')
    mask_color2 = fp["mask_color2"].astype('float32')
    mask_gt1 = fp["mask_gt1"]
    mask_gt2 = fp["mask_gt2"]
    mask_gt_pc = fp["mask_gt_pc"].astype('float32')
    point1 = fp["point1"].astype('float32')
    color1 = fp["color1"].astype('float32')
    point2 = fp["point2"].astype('float32')
    color2 = fp["color2"].astype('float32')
    ground_truth = fp["ground_truth"].astype('float32')

    mask_point2,mask_gt_pc,point2,ground_truth = apply_rigid_transform(mask_point2,mask_gt_pc,point2,ground_truth)

    target_path_in = os.path.join(target_path, data_path.split('/')[-1])
    np.savez_compressed(target_path_in,
                        point1=point1,
                        color1=color1,
                        ground_truth=ground_truth,
                        point2=point2,
                        color2=color2,
                        mask_point1=mask_point1,
                        mask_color1=mask_color1,
                        mask_point2=mask_point2,
                        mask_color2=mask_color2,
                        mask_gt1=mask_gt1,
                        mask_gt2=mask_gt2,
                        mask_gt_pc=mask_gt_pc)
    print("finish {}".format(count))
    count +=1
