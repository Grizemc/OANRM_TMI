import multiprocessing
import os.path
import itertools
import glob
import numpy as np
import random


def copy_files(sampled_files):
    for file in sampled_files:
        # 假设 road 是一个已定义的变量，指向目标目录
        destination = os.path.join(road, "train", file.split('/')[-1])
        os.system(f"cp {file} {destination}")
        print(f"{file} copied to {destination}")


if __name__ == '__main__':

    road = '/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_mix'
    if not os.path.exists(road):
        os.mkdir(road)

    a = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_65", "train", "/*.npz"))
    b = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_75", "train", "/*.npz"))
    c = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_80", "train", "/*.npz"))
    d = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_85", "train", "/*.npz"))
    e = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual_90", "train", "/*.npz"))
    f = glob.glob(os.path.join("/big_data/szm/M8ICCAI_8192_Mask_19055_new_mutual", "train", "/*.npz"))

    num_samples_1 = len(a) // 6
    num_samples_2 = len(b) // 6
    num_samples_3 = len(c) // 6
    num_samples_4 = len(d) // 6
    num_samples_5 = len(e) // 6
    num_samples_6 = len(f) // 6

    cwz = []
    # 使用random.sample()进行随机抽样
    sampled_files1 = random.sample(a, num_samples_1)
    sampled_files2 = random.sample(b, num_samples_2)
    sampled_files3 = random.sample(c, num_samples_3)
    sampled_files4 = random.sample(d, num_samples_4)
    sampled_files5 = random.sample(e, num_samples_5)
    sampled_files6 = random.sample(f, num_samples_6)

    # cwz.extend(sampled_files1)
    # cwz.extend(sampled_files2)
    # cwz.extend(sampled_files3)
    # cwz.extend(sampled_files4)
    # cwz.extend(sampled_files5)
    # cwz.extend(sampled_files6)

    cwz.append(sampled_files1)
    cwz.append(sampled_files2)
    cwz.append(sampled_files3)
    cwz.append(sampled_files4)
    cwz.append(sampled_files5)
    cwz.append(sampled_files6)


    def copy(sampled_files1):
        os.system("cp {} {}".format(file, os.path.join(road, "train", file.split('/')[-1])) for file \
                  in sampled_files1)
        print("{} final".format(file))


    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=6)
    pool.map(copy_files, cwz)
    pool.close()
    pool.join()

# for file in cwz:
#     os.system("cp {} {}".format(file,os.path.join(road,"train",file.split('/')[-1])))
#     print("{} final".format(file))

# cwz.extend(itertools.chain(sampled_files1, sampled_files2, sampled_files3, sampled_files4, sampled_files5, sampled_files6))

# 打印抽样结果
# print(sampled_files)