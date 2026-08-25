import numpy as np

# a = np.load(r"D:\try\try\szmCode\paconv_\BCPD\rectified01_1000.npz")
# print(a)

import os
import re
import glob

# 假设 file_list 是包含所有文件名的列表
# root = "/big_data/szm/szm_MICCAI_Hamlyn/TanTaiBiaoZhu/szmFilteNpz/"
root = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/source_Hamlyn_sample_npz/wancheng"

# data_path = glob.glob(os.path.join(root, "test", '*.npz'))
data_path = glob.glob(os.path.join(root, '*.npz'))
data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))

print(data_path[155])
print(data_path[1891])
print(data_path[2507])

num_root = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85noise/test_0.5_0.03"
data_path2 = glob.glob(os.path.join(num_root, '*.npz'))
print("num_sample", len(data_path2))
# a = data_path[2516]
# os.system("cp {} /home/szm/Paconv_730/".format(a))



# import os
# root_dir = "/big_data/szm/cwz_stereomis_P2-5_0.35"
#
# # 获取所有 npz 文件
# npz_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".npz")])
#
# # 重命名文件
# for i, old_file_name in enumerate(npz_files):
#     # 构建新文件名
#     new_file_name = f"{i:05d}.npz"
#
#     # 获取旧文件的完整路径
#     old_file_path = os.path.join(root_dir, old_file_name)
#
#     # 获取新文件的完整路径
#     new_file_path = os.path.join(root_dir, new_file_name)
#
#     # 重命名文件
#     os.rename(old_file_path, new_file_path)
#     print(f"Renamed: {old_file_name} -> {new_file_name}")

