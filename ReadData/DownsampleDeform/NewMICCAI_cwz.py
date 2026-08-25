import glob
import multiprocessing
import os
import numpy as np
import open3d
from GenerateMaskUtil import quickRemove000, VoxelDownSample, GendeformSourceSize, New_Mutual_Mask

"""
参数设置：main函数设置源文件夹 ，GenerateDataset设置目标文件夹。
"""
# souce的文件为源数据集文件，被读取后，形成处理后可用于训练的数据集，放在target_path下
# 关键是找到源数据集文件
# 完成任务的file，移动到wancheng中，原位置没有了
# 本文件的file成功的，放在wancheng中 /big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/wancheng
# 生成结果
# 人工形变数据集
# try在循环内部，从try跳到except后，会继续执行下一个循环操作

# file_list

z_std = 2.0  # mm
c_std = 0.10  # RGB noise

cfg = {
    "z_noise_std": z_std,
    "color_noise_std": c_std,
    "random_seed": 42
}


def GenerateDataset(file):
    # target_path = r"/big_data/szm/szm_MICCAI_Hamlyn/MICCAI_8192_Train"
    target_path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85noise/test"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # target_path = r"/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_90"
    source_pcd_num = 8192
    voxel_size = 400
    # for file in file_list:
    # file.split('/')[-1]，
    # file_name = file.split('/')[-1]
    # # 检查最后一个元素的前八个字母是否为 'dataset7'
    # if file_name[:8] == 'dataset7':
    #     target_path_in = os.path.join(target_path, 'test', file_name)
    #     target_path_in_dir = os.path.join(target_path, 'test')
    #     os.makedirs(target_path_in_dir, exist_ok=True)
    # else:
    #     target_path_in = os.path.join(target_path, 'train', file_name)
    #     target_path_in_dir = os.path.join(target_path, 'train')
    #     os.makedirs(target_path_in_dir, exist_ok=True)
    target_path_in = os.path.join(target_path, file.split('/')[-1])

    # try在循环内部，从try跳到except后，会继续执行下一个循环操作
    try:
        with np.load(file) as npz:
            # 从每个file中提取xyz和rgb
            xyz = npz["xyz"]
            rgb = npz["rgb"]
        # Remove 000 point & outliers
        """在一个点周围选择若干个点，计算它们距离的统计参数，如果某个点偏离平均值超过stdio_ratio倍的方差
        则认为是离群点并进行删除。std_ratio实际上是指偏离标准差的倍数。因此，这种方法也可以称为邻域滤波。"""
        point_cloud = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(xyz))
        point_cloud.colors = open3d.pybind.utility.Vector3dVector(rgb)

        #
        point_cloud_remove000 = quickRemove000(point_cloud)

        # 基于距离统计
        point_cloud_remove = open3d.geometry.PointCloud.remove_statistical_outlier(
            point_cloud_remove000, nb_neighbors=100, std_ratio=1)[0]

        # Sample，从均匀分布中采样随机数,为index_percentage=（1-重叠区域的概率）
        # !!!!!!!!!!!!!!!!!!!!!!!!!!  0.92+0.8   /2  85
        # index_percentage = np.random.uniform(0.08, 0.2)  # Probability of non-overlap
        # 下采样  0.05, 0.1  0.2 0.15   0.10, 0.15
        index_percentage = np.random.uniform(0.08, 0.2)
        # index_percentage = np.random.uniform(0.8,0.92)

        pcd_voxel, pcd_type, voxel_size = VoxelDownSample(point_cloud_remove, voxel_size, index_percentage,
                                                          source_pcd_num)

        z_noise_std = cfg["z_noise_std"]
        color_noise_std = cfg["color_noise_std"]

        # 下采样成功时ls
        if pcd_type:
            # 　No Mask
            point1 = np.array(pcd_voxel.points)
            color1 = np.array(pcd_voxel.colors)

            p_noise = np.random.normal(0, z_noise_std, size=point1.shape[0])
            c_noise = np.random.normal(0, color_noise_std, size=color1.shape)

            point1[:, 2] += p_noise
            color1 = np.array(pcd_voxel.colors) + c_noise
            # 生成虚拟数据集后，点坐标改变，但是对应索引的颜色值不变

            point1, ground_truth = GendeformSourceSize(point1)
            color2 = color1.copy()
            point2 = ground_truth.copy()
            # 按列拼接  首先要保持两片点云的数据索引肯定是乱序的。
            result_points_colors = np.concatenate((point2, color2), axis=1)
            np.random.shuffle(result_points_colors)  # 乱序，只有行顺序被打乱
            # 点坐标为前三列，颜色坐标为后三列
            point2 = result_points_colors[:, 0:3]
            color2 = result_points_colors[:, 3:6]

            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, mask_gt_pc = New_Mutual_Mask(
                point1,
                color1,
                ground_truth,
                index_percentage)

            # 会对数据进行压缩，以减少文件的大小。
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
            print(
                "{} is stored, source num is {},  input num is {}, mask num is {}, percentage is {}".format(
                    file, xyz.shape[0], color1.shape[0], mask_point1.shape[0], (1 - index_percentage) * 100))
            # 完成任务的file，移动到wancheng中，原位置没有了
            # os.system("mv {} {}".format(file,os.path.join("/big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/wancheng1",file.split('/')[-1])))
        else:
            # 点数太少，不行
            print("{} is less".format(file))
    except:
        # 失败
        print("file {} is fail".format(file.split("/")[-1]))
        # break
        # 将失败的file移动到/big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/fail下
        # os.system("mv {} {}".format(file, os.path.join("/big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/fail1",
        #                                          file.split('/')[-1])))


if __name__ == "__main__":
    # config 某些设置文件写在 GenerateDataset 函数的里面
    # root_path = r"/big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/all/wancheng"
    # root_path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/source_Hamlyn_sample_npz/wancheng"
    # all_split_num = 30
    # pool_num = 10
    # # Read dataset
    # file_list = glob.glob(os.path.join(root_path + "/*.npz"))
    # # key：指定一个函数，该函数会作用于列表中的每个元素上，并返回一个用于排序的值。
    # file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    # file_list.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
    # # split dataset
    # all_len = len(file_list)
    # temp_len = all_len // all_split_num
    # file_group_list = []
    # last_number = 0
    # for i in range(0, all_len, temp_len):
    #     if i != 0:
    #         temp_list = file_list[last_number:i]
    #         # file_group_list中有好几个列表，每个列表有几个路径
    #         file_group_list.append(temp_list)
    #     last_number = i
    # if last_number != all_len:
    #     temp_list = file_list[last_number:all_len]
    #     file_group_list.append(temp_list)
    # # temp_list = ["/big_data/szm/szm_MICCAI_Hamlyn/MICCAISourceNpz/dataset2_keyframe_4_939.npz"]
    # # GenerateDataset(temp_list)
    # #
    # multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    # pool = multiprocessing.Pool(pool_num)
    # pool.map(GenerateDataset, file_group_list)
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出

    root_path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/source_Hamlyn_sample_npz/wancheng"
    all_split_num = 30

    # Read dataset
    file_list = glob.glob(os.path.join(root_path + "/*.npz"))

    # Sort the file list
    file_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    file_list.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))

    # Process each file directly
    for file_path in file_list:
        try:
            GenerateDataset(file_path)  # Pass the file as a single-item list if required
        except Exception as e:
            print("{} is fail".format(GenerateDataset))