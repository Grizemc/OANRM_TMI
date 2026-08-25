#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 16:36
# @Author  : 沈子明
# @File    : Big_main.py
# @Software: PyCharm
import glob
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from model.backbone_new import PTEnetBase, PTFlow
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from util.util import load_cfg_from_cfg_file, IOStream

"""
使用澹台佑彤师姐标注的数据集，进行评价
直接测试，不进行在线学习微调
"""


class HamlynTest(Dataset):
    def __init__(self, root="/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/test"):
        self.root = root
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.data_path.sort(key=lambda x: int(x.split('rectified')[-1].split('_')[0]))
        self.len = len(self.data_path)

    def __getitem__(self, index):
        fn = self.data_path[index]
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
        return points1, points2, colors1, colors2, mask_gt1, mask_gt2, mask_gt_pc

    def __len__(self):
        return len(self.data_path)


def init(type, type_name):
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml',
                        help='config file')
    args_l = parser.parse_args()
    assert args_l.config is not None
    args = load_cfg_from_cfg_file(args_l.config)
    # -----------------------------------------------------------------------------
    # backup the running files
    if not os.path.exists('checkpoints/' + args.exp_name + '/saved_model'):
        raise SystemExit('No model')
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + type_name):
        os.makedirs('checkpoints/' + args.exp_name + '/' + type_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + type_name + '/npz_result'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + type_name + '/npz_result')
    source_io = IOStream(
        'checkpoints/' + args.exp_name + '/' + type_name + '/log.log'.format(
            type))
    source_io.cprint(
        'checkpoints/' + args.exp_name + '/' + type_name + '/log{}_single_relax.log'.format(
            type))
    file_name = os.path.basename(sys.argv[0])
    os.system(
        'cp -r {} checkpoints/'.format(file_name) + args.exp_name + '/' + type_name + '/{}.txt'.format(file_name))
    # -----------------------------------------------------------------------------
    # set random seed
    # -----------------------------------------------------------------------------
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        source_io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        source_io.cprint('Using CPU')
    source_io.cprint(str(args))
    args.source_io = source_io
    args.dataset_type = type
    args.directory = type_name
    return args


def HamlynDirectTest(dataloader, model, args):
    if args.model_test_type == "Train":
        model.train()
        args.source_io.cprint("model.train()")
    elif args.model_test_type == "eval":
        model.eval()
        args.source_io.cprint("model.eval()")
    with torch.no_grad():
        for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            points1, points2, colors1, colors2, mask_gt1, mask_gt2, mask_gt_pc = data
            points1 = points1.to(args.device)
            points2 = points2.to(args.device)
            colors1 = colors1.to(args.device)
            colors2 = colors2.to(args.device)
            mask_gt1 = mask_gt1.to(args.device)
            mask_gt2 = mask_gt2.to(args.device)
            mask_gt_pc = mask_gt_pc.to(args.device)
            l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(points1, points2, colors1,
                                                                                   colors2)
            save_path = 'checkpoints/{}/{}/npz_result/sample_num_{}_best.npz'.format(
                str(args.exp_name), str(args.directory), index)
            np.savez(save_path,
                     points2=points2.cpu(),
                     points1=points1.cpu(),
                     colors1=colors1.cpu(),
                     colors2=colors2.cpu(),
                     pred_xyz=l_pred_xyz[0].detach().cpu(),
                     mask_gt1=mask_gt1.cpu(),
                     mask_gt2=mask_gt2.cpu(),
                     mask_gt_pc=mask_gt_pc.detach().cpu(),
                     pred_mask1=(torch.sigmoid(l_pred_mask1[0])).detach().cpu(),
                     pred_mask2=(torch.sigmoid(l_pred_mask2[0])).detach().cpu())


if __name__ == "__main__":
    # dataset_Path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual/test"
    # dataset_Path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_Low_Overlap/test"
    # dataset_Path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_80/test"
    # dataset_Path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/test"
    dataset_Path = r"/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_90/test"

    mask_ratio = 0.9
    dataset_type = "test"
    args = init(dataset_type, "Hamlyn_DircetTest_no_rotation_90")
    args.mask_ratio = mask_ratio
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.model_test_type = "eval"
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.model_type == "Base":
        model = PTEnetBase(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow":
        model = PTFlow(c=6, args=args).to(args.device)
    else:
        raise SystemExit('Not impletion')
    model_path = os.path.join('checkpoints/', args.exp_name, "saved_model/best_model.t7")
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    total = sum([param.nelement() for param in model.parameters()])
    args.source_io.cprint("Number of parameter: %.2fM" % (total / 1e6))
    test_hamlyn_loader = DataLoader(HamlynTest(root=dataset_Path))
    HamlynDirectTest(test_hamlyn_loader, model, args)
