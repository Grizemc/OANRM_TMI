
import glob
import multiprocessing
import os
import numpy as np
import open3d

def quickRemove000(pcd):
    """
    :param pcd: point cloud open3d
    :return: point cloud open3d without (0,0,0)
    """
    xyz_in = np.array(pcd.points)  # 读取点云中的点
    rgb = np.array(pcd.colors)  # 读取点云中的点
    # 生成需要删除的点的索引
    temp_mask = (xyz_in == [0, 0, 0])
    mask = temp_mask[:, 0] * temp_mask[:, 1] * temp_mask[:, 2]
    mask = ~ mask
    xyz_remove_0 = xyz_in[mask, :]
    rgb_remove_0 = rgb[mask, :]
    point_cloud = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(xyz_remove_0))
    point_cloud.colors = open3d.pybind.utility.Vector3dVector(rgb_remove_0)
    return point_cloud
def Read_xyz_len(xyz_in):
    x = xyz_in[:, 0].reshape(-1, 1)
    y = xyz_in[:, 1].reshape(-1, 1)
    z = xyz_in[:, 2].reshape(-1, 1)
    len_x = np.max(x) - np.min(x)
    len_y = np.max(y) - np.min(y)
    len_z = np.max(z) - np.min(z)
    return x, y, z, len_x, len_y, len_z
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


def HuaXiDownSample(file_list_in):
    target_path_in = r"/big_data/szm/cwz_stereomis_P2-5_0.01_new"
    if not os.path.exists(target_path_in):
        os.mkdir(target_path_in)
    else:
        pass
    log_path_in = os.path.join(target_path_in, "log.txt")
    log = IOStream(log_path_in)
    for file in file_list_in:
        with np.load(file) as npz:
            # xyz = npz["point1"]
            # color = npz["color1"]
            xyz = npz["xyz"]
            color = npz["rgb"]
            point_cloud = open3d.geometry.PointCloud(open3d.pybind.utility.Vector3dVector(xyz))
            point_cloud.colors = open3d.pybind.utility.Vector3dVector(color)
            point_cloud_remove000 = quickRemove000(point_cloud)
            point_cloud_remove = open3d.geometry.PointCloud.remove_statistical_outlier(
                point_cloud_remove000, nb_neighbors=20, std_ratio=1.0)[0]
            # Sample
            voxel_size = 0.01 # 0.5 # 0.35
            pcd_voxel = point_cloud.voxel_down_sample(voxel_size=voxel_size)  # 体素均匀下采样
            point1 = np.array(pcd_voxel.points)
            color1 = np.array(pcd_voxel.colors)
            np.savez_compressed(os.path.join(target_path_in, file.split("/")[-1]),
                                point1=point1,
                                color1=color1)
            log.cprint(
                "{} is stored, source num is {},  input num is {}".format(
                    file, xyz.shape[0], color1.shape[0]))
    print("Finished")


if __name__ == "__main__":
    # path = r"/big_data/szm/szm_MICCAI_Hamlyn/HuaXi/"
    path = r"/big_data/szm/cwz_stereomis_P2-5/"
    path = r"/big_data/szm/cwz_stereomis_new0610"
    file_list = glob.glob(os.path.join(path, "*.npz"))
    # group
    all_len = len(file_list)
    temp_len = all_len // 20
    file_group_list = []
    last_number = 0
    for i in range(0, all_len, temp_len):
        if i != 0:
            temp_list = file_list[last_number:i]
            file_group_list.append(temp_list)
            last_number = i
    if last_number != all_len:
        temp_list = file_list[last_number:all_len]
        file_group_list.append(temp_list)


    multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    pool = multiprocessing.Pool(5)
    pool.map(HuaXiDownSample, file_group_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    # for file_path in file_group_list:
    #      HuaXiDownSample(file_path)
    # print("All finished")
