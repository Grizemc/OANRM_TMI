#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/2 20:54
# @Author  : 沈子明
# @File    : MappingOverlapAblation.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 不同重叠比率对应的消融实验的图
# 以及在不同重叠比率下的MACC误差  应该是大论文的

if __name__ == "__main__":
    # 75 重叠率的数据集
    root_path1 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path2 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path3 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_low_overlap"
    root_path4 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_low_overlap"
    # 80重叠率的数据集
    root_path5 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path6 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path7 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_80"
    root_path8 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_80"
    # 85 重叠率数据集
    root_path9 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path10 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path11 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_85"
    root_path12 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_85"
    # 　90 重叠率的数据集
    root_path13 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path14 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path15 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation_90"
    root_path16 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation_90"
    #  94
    root_path17 = r"/home/szm/Paconv/checkpoints/Zall/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path18 = r"/home/szm/Paconv/checkpoints/Zall_focal_loss1/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path19 = r"/home/szm/Paconv/checkpoints/Zall_no_fuse/fpfh_Post_Train_Hamlyn_no_rotation"
    root_path20 = r"/home/szm/Paconv/checkpoints/Zall/Hamlyn_DircetTest_no_rotation"
    # 低重叠率数据集
    root_zall = [root_path1, root_path5, root_path9, root_path13, root_path17]
    root_focal = [root_path2, root_path6, root_path10, root_path14, root_path18]
    root_no_fuze = [root_path3, root_path7, root_path11, root_path15, root_path19]
    root_direct = [root_path4, root_path8, root_path12, root_path16, root_path20]
    all_path = [root_zall, root_focal, root_no_fuze, root_direct]

    """
    竖版的折线图
    """
    # Create a new figure and arrange the subplots vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7.4),gridspec_kw={'height_ratios': [1.1, 1], 'hspace': 0.48})
    # Plot the lines for the first graph with the same color
    # colors = plt.cm.viridis(np.linspace(0, 1, 5))  # Generate 5 different colors
    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    fig.subplots_adjust(top=0.97)
    text_fontsize = 10
    labelsize = 10
    """
    ======================================================ACC================================================
    """
    # 低重叠率数据集
    root_zall = [root_path1, root_path5, root_path9, root_path13, root_path17]
    root_focal = [root_path2, root_path6, root_path10, root_path14, root_path18]
    root_no_fuze = [root_path3, root_path7, root_path11, root_path15, root_path19]
    root_direct = [root_path4, root_path8, root_path12, root_path16, root_path20]
    all_path = [root_zall, root_focal, root_no_fuze, root_direct]
    point_acc_list = []

    # root_focal为缺的距离图损失，root_no_fuze为缺的MF
    # 这里分别是root_zall root_focal root_no_fuze root_direct的消融实验某四个的精度
    for single_root_path in all_path:
        temp_point_acc = []
        for single_path in single_root_path:
            target_npz_path = single_path + "/GaussianInter.npy"
            print_data = np.load(target_npz_path)
            temp_point_acc.append(print_data[-1])
        point_acc_list.append(temp_point_acc)
    temp_point_acc = []
    # 这里用的是无后处理时的精度 （仅仅有监督+无监督）的精度
    for single_path in root_zall:
        target_npz_path = single_path + "/EvalHamlyn.npz"
        print_data = np.load(target_npz_path)
        final_acc = print_data['final_acc']
        temp_point_acc.append(final_acc[-1])
    point_acc_list.append(temp_point_acc)

    point_acc_list = np.array(point_acc_list)
    name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT","Full w/o TDDP"]
    x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    y = point_acc_list  # Y coordinates from the remaining rows
    # Plot the line graphs
    for i in range(y.shape[0]):
        ax1.plot(x, y[i, :], label=f'Line {name[i]}',color=colors[i], marker='o')
    # Add a legend between the subplots with adjusted position


    # Add legend, x-axis label, y-axis label, and title
    ax1.set_xlabel('Mean overlapping probability %',fontsize=text_fontsize)
    ax1.set_ylabel('EPE mm',fontsize=text_fontsize)
    ax1.tick_params(axis='both', labelsize=labelsize)
    fig.legend(loc='upper center',prop={'size':8.4},
               bbox_to_anchor=(0.5, -0.2),
               bbox_transform=ax1.transAxes,
               shadow=False, ncol=3)
    """
    ======================================================MASK================================================
    """
    mask_acc_list = []
    for single_root_path in all_path:
        temp_mask_acc = []
        for single_path in single_root_path:
            target_npz_path = single_path + "/EvalHamlyn.npz"
            print_data = np.load(target_npz_path)
            mask_acc = print_data['mask_acc']
            temp_mask_acc.append(mask_acc[-1])
        mask_acc_list.append(temp_mask_acc)
    mask_acc_array = np.array(mask_acc_list)
    name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT"]
    x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    y = mask_acc_array  # Y coordinates from the remaining rows
    # Plot the line graphs
    for i in range(y.shape[0]):
        ax2.plot(x, y[i, :], label=f'Line {name[i]}',color=colors[i], marker='o')

    # Add legend, x-axis label, y-axis label, and title
    ax2.set_xlabel('Mean overlapping probability %',fontsize=text_fontsize)
    ax2.set_ylabel('MAcc %',fontsize=text_fontsize)
    ax2.tick_params(axis='both', labelsize=labelsize)
    # Adjust layout and display the plot
    # 保存为高清TIFF图片，dpi为600
    plt.savefig('Figure_more/overlap_ablation.tiff', dpi=600, format='tiff')
    plt.show()
    """
    横板的小论文结果图
    """
    # font = FontProperties(fname="HarmonyOS_Sans_SC_Regular.ttf")
    # # Create a new figure and arrange the subplots horizontally
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
    #                                gridspec_kw={'width_ratios': [1, 1],
    #                                             'bottom': 0.22,
    #                                             'wspace': 0.3})
    # # Create a new figure and arrange the subplots vertically
    # colors = plt.cm.Set1(np.linspace(0, 1, 5))
    # text_fontsize = 14
    # labelsize = 14
    # """
    # ======================================================ACC================================================
    # """
    # # 低重叠率数据集
    # root_zall = [root_path1, root_path5, root_path9, root_path13, root_path17]
    # root_focal = [root_path2, root_path6, root_path10, root_path14, root_path18]
    # root_no_fuze = [root_path3, root_path7, root_path11, root_path15, root_path19]
    # root_direct = [root_path4, root_path8, root_path12, root_path16, root_path20]
    # all_path = [root_zall, root_focal, root_no_fuze, root_direct]
    # point_acc_list = []
    # for single_root_path in all_path:
    #     temp_point_acc = []
    #     for single_path in single_root_path:
    #         target_npz_path = single_path + "/GaussianInter.npy"
    #         print_data = np.load(target_npz_path)
    #         temp_point_acc.append(print_data[-1])
    #     point_acc_list.append(temp_point_acc)
    # temp_point_acc = []
    # for single_path in root_zall:
    #     target_npz_path = single_path + "/EvalHamlyn.npz"
    #     print_data = np.load(target_npz_path)
    #     final_acc = print_data['final_acc']
    #     temp_point_acc.append(final_acc[-1])
    # point_acc_list.append(temp_point_acc)
    #
    # point_acc_list = np.array(point_acc_list)
    # name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT","Full w/o TDDP"]
    # x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    # y = point_acc_list  # Y coordinates from the remaining rows
    # # Plot the line graphs
    # for i in range(y.shape[0]):
    #     ax1.plot(x, y[i, :], label=f'{name[i]}',color=colors[i], marker='o')
    # # Add a legend between the subplots with adjusted position
    #
    #
    # # Add legend, x-axis label, y-axis label, and title
    # ax1.set_xlabel('Mean overlapping probability %',fontsize=text_fontsize, fontproperties=font, labelpad=10)
    # ax1.set_ylabel('EPE mm',fontsize=text_fontsize, fontproperties=font)
    # ax1.tick_params(axis='both', labelsize=labelsize)
    # fig.legend(loc='lower center', prop={'size': 14},
    #            shadow=False, ncol=5)
    # """
    # ======================================================MASK================================================
    # """
    # mask_acc_list = []
    # for single_root_path in all_path:
    #     temp_mask_acc = []
    #     for single_path in single_root_path:
    #         target_npz_path = single_path + "/EvalHamlyn.npz"
    #         print_data = np.load(target_npz_path)
    #         mask_acc = print_data['mask_acc']
    #         temp_mask_acc.append(mask_acc[-1])
    #     mask_acc_list.append(temp_mask_acc)
    # mask_acc_array = np.array(mask_acc_list)
    # name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT"]
    # x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    # y = mask_acc_array  # Y coordinates from the remaining rows
    # # Plot the line graphs
    # for i in range(y.shape[0]):
    #     ax2.plot(x, y[i, :], label=f'{name[i]}',color=colors[i], marker='o')
    #
    # # Add legend, x-axis label, y-axis label, and title
    # ax2.set_xlabel('Mean overlapping probability %',fontsize=text_fontsize, fontproperties=font, labelpad=10)
    # ax2.set_ylabel('MAcc %',fontsize=text_fontsize, fontproperties=font)
    # ax2.tick_params(axis='both', labelsize=labelsize)
    # plt.savefig('Figure_more/大论文横板.tiff',dpi=600)
    # plt.show()
    """
    大论文中的结果图
    """

    # font = FontProperties(fname="HarmonyOS_Sans_SC_Regular.ttf")
    # # Create a new figure and arrange the subplots horizontally
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
    #                                gridspec_kw={'width_ratios': [1, 1],
    #                                             'bottom': 0.22,
    #                                             'wspace': 0.3})
    # # Create a new figure and arrange the subplots vertically
    # colors = plt.cm.Set1(np.linspace(0, 1, 5))
    # text_fontsize = 14
    # labelsize = 14
    # """
    # ======================================================ACC================================================
    # """
    # # 低重叠率数据集
    # root_zall = [root_path1, root_path5, root_path9, root_path13, root_path17]
    # root_focal = [root_path2, root_path6, root_path10, root_path14, root_path18]
    # root_no_fuze = [root_path3, root_path7, root_path11, root_path15, root_path19]
    # root_direct = [root_path4, root_path8, root_path12, root_path16, root_path20]
    # all_path = [root_zall, root_focal, root_no_fuze, root_direct]
    # point_acc_list = []
    # for single_root_path in all_path:
    #     temp_point_acc = []
    #     for single_path in single_root_path:
    #         target_npz_path = single_path + "/GaussianInter.npy"
    #         print_data = np.load(target_npz_path)
    #         temp_point_acc.append(print_data[-1])
    #     point_acc_list.append(temp_point_acc)
    # temp_point_acc = []
    # for single_path in root_zall:
    #     target_npz_path = single_path + "/EvalHamlyn.npz"
    #     print_data = np.load(target_npz_path)
    #     final_acc = print_data['final_acc']
    #     temp_point_acc.append(final_acc[-1])
    # point_acc_list.append(temp_point_acc)
    #
    # point_acc_list = np.array(point_acc_list)
    # name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT","Full w/o TDDP"]
    # x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    # y = point_acc_list  # Y coordinates from the remaining rows
    # # Plot the line graphs
    # for i in range(y.shape[0]):
    #     ax1.plot(x, y[i, :], label=f'{name[i]}',color=colors[i], marker='o')
    # # Add a legend between the subplots with adjusted position
    #
    #
    # # Add legend, x-axis label, y-axis label, and title
    # ax1.set_xlabel('Overlap ratio %',fontsize=text_fontsize, fontproperties=font, labelpad=10)
    # ax1.set_ylabel('EPE mm',fontsize=text_fontsize, fontproperties=font)
    # ax1.tick_params(axis='both', labelsize=labelsize)
    # fig.legend(loc='lower center', prop={'size': 14},
    #            shadow=False, ncol=5)
    # """
    # ======================================================MASK================================================
    # """
    # mask_acc_list = []
    # for single_root_path in all_path:
    #     temp_mask_acc = []
    #     for single_path in single_root_path:
    #         target_npz_path = single_path + "/EvalHamlyn.npz"
    #         print_data = np.load(target_npz_path)
    #         mask_acc = print_data['mask_acc']
    #         temp_mask_acc.append(mask_acc[-1])
    #     mask_acc_list.append(temp_mask_acc)
    # mask_acc_array = np.array(mask_acc_list)
    # name = ['Full', "Full w/o DMS", "Full w/o MF", "Full w/o UFT"]
    # x = [75, 80, 85, 90, 95]  # X coordinates from the first row
    # y = mask_acc_array  # Y coordinates from the remaining rows
    # # Plot the line graphs
    # for i in range(y.shape[0]):
    #     ax2.plot(x, y[i, :], label=f'{name[i]}',color=colors[i], marker='o')
    #
    # # Add legend, x-axis label, y-axis label, and title
    # ax2.set_xlabel('Overlap ratio %',fontsize=text_fontsize, fontproperties=font, labelpad=10)
    # ax2.set_ylabel('OAcc %',fontsize=text_fontsize, fontproperties=font)
    # ax2.tick_params(axis='both', labelsize=labelsize)
    # plt.savefig('Figure_more/大论文.jpg',dpi=600)
    # plt.show()



