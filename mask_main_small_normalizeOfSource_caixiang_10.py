    #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 16:36
# @Author  : 沈子明
# @File    : Big_main.py
# @Software: PyCharm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from model.paconv import PAConv
from model.backbone_new import PTEnetBase, PTFlow
from model.loss import Only_mask, Only_point, Only_one_loss, Multi_Loss, WeightedFocalLossAll, \
    WeightedDistance_mapLossAll, Weighted_Mask_Loss_All, one_loss, one_loss_focal_loss, one_loss_dual, \
    one_gradient_loss, one_loss_smooth, one_loss_focal
import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from util.data import MaskMICCAIMutualNormalized, MaskMICCAIMutualNormalizedSpecial
from util.util import load_cfg_from_cfg_file, IOStream, CfgNode, weight_init
from torch import nn


def init():
    parser = argparse.ArgumentParser(description='The Pytorch porgramme Point Cloud correspondence')
    parser.add_argument('--config', type=str, default='/home/szm/Paconv_730/config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_10.yaml',
                        help='config file')
    args_l = parser.parse_args()
    assert args_l.config is not None
    args = load_cfg_from_cfg_file(args_l.config)
    # -----------------------------------------------------------------------------
    # backup the running files
    # -----------------------------------------------------------------------------
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if args.train:
        if not os.path.exists('checkpoints/' + args.exp_name + '/train'):
            os.makedirs('checkpoints/' + args.exp_name + '/train')
            os.makedirs('checkpoints/' + args.exp_name + '/pythonfile')
            os.makedirs('checkpoints/' + args.exp_name + '/saved_model')
        io = IOStream('checkpoints/' + args.exp_name + '/train.log')
        writer = SummaryWriter('checkpoints/' + args.exp_name + '/train')
        pythonfile = '/' + args.exp_name + '/pythonfile' + '/'
        os.system('cp -r model checkpoints' + pythonfile)
        os.system('cp -r util checkpoints' + pythonfile)
    # -----------------------------------------------------------------------------
    # set random seed
    # -----------------------------------------------------------------------------
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')
    io.cprint(str(args))
    args.correlation_loss = args.get('correlation_loss', False)
    args.color_aug = args.get('color_aug', False)
    args.dataset = args.get('dataset', 'normal')
    return args, io, writer


def Miccai_absolute_strength_evaluate_result(pcd1, pcd2, mask):
    """
    predict is success when the predicted value is evaluate_indicator%(default 0.2) above or below the true value.
    Args:
        pcd1:  source point cloud xyz  [bs, num_points, 3]
        pcd2: target point cloud xyz
    Returns: Acc
    """
    pcd1 = pcd1[mask, :]
    pcd2 = pcd2[mask, :]
    num = mask.sum()
    displace = torch.norm(pcd1 - pcd2, dim=1)
    relax_error = [1, 2, 3, 5, 10]
    acc_list = []
    for error in relax_error:
        acc_list.append((displace < error).sum() / num)
    acc_list.append(displace.mean() / 100)
    acc_list = torch.tensor(acc_list).to(pcd1.device)
    return acc_list


def writer_info(out_info, epoch, iol, writerl, type="Train"):
    if type == "Test":
        out_info.acc = out_info.test_acc
        out_info.loss_sum = out_info.test_loss_sum
        out_info.loss = out_info.test_loss
    outstr1 = "{}: {} epoch, time consumption is {} ".format(type, epoch,
                                                             (out_info.end_time - out_info.start_time).seconds)
    outstr2 = "{} loss is :{}, The {} acc is {}%".format(type, out_info.loss, type, out_info.acc)
    writerl.add_scalars("{}_loss".format(type), {'{}_loss'.format(type): out_info.loss,
                                                 '{}_gt_loss'.format(type): out_info.loss_sum[0],
                                                 '{}_mask_loss'.format(type): out_info.loss_sum[1],
                                                 }, global_step=epoch)
    writerl.add_scalars("{}_Single_loss".format(type),
                        {"{}_smooth_loss".format(type): out_info.loss_sum[2],
                         "{}_corr_loss".format(type): out_info.loss_sum[3]},
                        global_step=epoch)
    writerl.add_scalars("{}_point_acc".format(type),
                        {'1': out_info.acc[0][0], '2': out_info.acc[0][1], '3': out_info.acc[0][2],
                         '5': out_info.acc[0][3], '10': out_info.acc[0][4]}, global_step=epoch)
    writerl.add_scalars("point_error".format(type), {'{}_point_error'.format(type): out_info.acc[0][5],
                                                     '{}_normalized_point_error'.format(type): out_info.acc[3]},
                        global_step=epoch)
    writerl.add_scalars("{}_mask_acc".format(type),
                        {'0.7': out_info.acc[1][0], '0.75': out_info.acc[1][1], '0.8': out_info.acc[1][2],
                         '0.85': out_info.acc[1][3], '0.9': out_info.acc[1][4]}, global_step=epoch)
    iol.cprint(outstr1 + outstr2)


def xyz_restore(xyz_in_all, relax_proportion_all):
    B = xyz_in_all.shape[0]
    result = []
    for i in range(B):
        xyz_in = xyz_in_all[i, :]
        relax_proportion = relax_proportion_all[i]
        len_x, len_y, len_z, x_min, y_min, z_min = relax_proportion
        x = xyz_in[:, 0].reshape(-1, 1)
        y = xyz_in[:, 1].reshape(-1, 1)
        z = xyz_in[:, 2].reshape(-1, 1)
        new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
        new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
        new_z = (z + 0.5) * len_z + z_min
        temp = torch.cat((new_x, new_y, new_z), axis=1).reshape(-1, 3)
        result.append(temp)
    result = torch.stack(result)
    return result


def train_one_epoch(model, optimizer, epoch, args):
    ####################
    # Train
    ####################
    train_loss = 0.0
    acc = 0.0
    gt_loss_sum = 0.0
    mask_loss_sum = 0.0
    smooth_loss_sum = 0.0
    mask_acc = 0.0
    corr_loss_sum = 0.0
    precise_mask_acc = 0.0
    normalized_acc = 0.0
    model.train()
    loop = tqdm(enumerate(args.train_loader), total=len(args.train_loader))
    for index, data in loop:
        mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_gt_pc, gt_pc, _ = data
        mask_point1 = mask_point1.to(args.device)
        mask_point2 = mask_point2.to(args.device)
        mask_color1 = mask_color1.to(args.device)
        mask_color2 = mask_color2.to(args.device)
        mask_gt1 = mask_gt1.to(args.device)
        mask_gt2 = mask_gt2.to(args.device)
        nor_gt_pc = nor_gt_pc.to(args.device)
        gt_pc = gt_pc.to(args.device)
        # relax_ratio = relax_ratio.to(args.device)
        # opt
        optimizer.zero_grad()
        l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(mask_point1, mask_point2, mask_color1,
                                                                               mask_color2)

        loss, gt_loss, mask_loss, smooth_loss = args.loss_fn.losscal(l_pc1=l_xyz1, l_pred=l_pred_xyz, pc2=mask_point2,
                                                                     l_idx1=l_idx1, l_idx2=l_idx2,
                                                                     l_pred_mask2=l_pred_mask2,
                                                                     l_pred_mask1=l_pred_mask1,
                                                                     gt_pc=nor_gt_pc, mask_gt1=mask_gt1
                                                                     , mask_gt2=mask_gt2, epoch=epoch)
        corr_loss = 0.0
        if args.correlation_loss:
            for m in model.module.SA_modules.named_modules():
                if isinstance(m[-1], PAConv):
                    kernel_matrice, output_dim, m_dim = m[-1].weightbank, m[-1].output_dim, m[-1].m
                    new_kernel_matrice = kernel_matrice.view(-1, m_dim, output_dim).permute(1, 0, 2).reshape(m_dim, -1)
                    cost_matrice = torch.matmul(new_kernel_matrice, new_kernel_matrice.T) / torch.matmul(
                        torch.sqrt(torch.sum(new_kernel_matrice ** 2, dim=-1, keepdim=True)),
                        torch.sqrt(torch.sum(new_kernel_matrice.T ** 2, dim=0, keepdim=True)))
                    corr_loss += torch.sum(torch.triu(cost_matrice, diagonal=1) ** 2)
            loss = loss + corr_loss * args.paloss_factor
            corr_loss_sum += corr_loss.item()
        loss.backward()
        optimizer.step()
        # record loss
        train_loss += loss.item()
        gt_loss_sum += gt_loss.item()
        mask_loss_sum += mask_loss.item()
        smooth_loss_sum += smooth_loss.item()

        # record network acc
        # acc += Miccai_absolute_strength_evaluate_result(xyz_restore(l_pred_xyz[0], relax_ratio), gt_pc, mask_gt1)
        acc += Miccai_absolute_strength_evaluate_result(l_pred_xyz[0], gt_pc, mask_gt1)
        normalized_acc += Miccai_absolute_strength_evaluate_result(l_pred_xyz[0], nor_gt_pc, mask_gt1)[-1]
        mask_acc += Mask_evaluate_result(torch.sigmoid(l_pred_mask1[0]), mask_gt1)
        # 更新信息
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item())
    acc = acc / (index + 1) * 100
    mask_acc = mask_acc / (index + 1) * 100
    precise_mask_acc = precise_mask_acc / (index + 1) * 100
    normalized_acc = normalized_acc / (index + 1) * 100
    return train_loss, [acc, mask_acc, precise_mask_acc, normalized_acc], [gt_loss_sum, mask_loss_sum, smooth_loss_sum,
                                                                       corr_loss_sum]


def Mask_evaluate_result(pred, gt):
    pred = pred.squeeze()
    B, N = gt.shape[0], gt.shape[1]
    truncation_nums = [0.7, 0.75, 0.8, 0.85, 0.9]
    success_ratio = []
    for truncation_num in truncation_nums:
        pred_trun = pred > truncation_num
        success_ratio.append(pred_trun.eq(gt).sum() / (B * N))
    success_ratio = torch.tensor(success_ratio)
    return success_ratio


def test_one_epoch(model, args):
    test_loss = 0.0
    test_acc = 0.0
    gt_loss_sum = 0.0
    mask_loss_sum = 0.0
    smooth_loss_sum = 0.0
    test_mask_acc = 0.0
    test_precise_mask_acc = 0.0
    corr_loss_sum = 0.0
    normalized_acc = 0.0
    with torch.no_grad():
        for index, data in tqdm(enumerate(args.test_loader), total=len(args.test_loader)):
            mask_point1, mask_color1, mask_point2, mask_color2, mask_gt1, mask_gt2, nor_gt_pc, gt_pc, _ = data
            mask_point1 = mask_point1.to(args.device)
            mask_point2 = mask_point2.to(args.device)
            mask_color1 = mask_color1.to(args.device)
            mask_color2 = mask_color2.to(args.device)
            mask_gt1 = mask_gt1.to(args.device)
            mask_gt2 = mask_gt2.to(args.device)
            nor_gt_pc = nor_gt_pc.to(args.device)
            gt_pc = gt_pc.to(args.device)
            # relax_ratio = relax_ratio.to(args.device)
            # opt
            l_xyz1, l_pred_xyz, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2 = model(mask_point1, mask_point2,
                                                                                   mask_color1,
                                                                                   mask_color2)

            loss, gt_loss, mask_loss, smooth_loss = args.loss_fn.losscal(l_pc1=l_xyz1, l_pred=l_pred_xyz,
                                                                         pc2=mask_point2,
                                                                         l_idx1=l_idx1, l_idx2=l_idx2,
                                                                         l_pred_mask2=l_pred_mask2,
                                                                         l_pred_mask1=l_pred_mask1,
                                                                         gt_pc=nor_gt_pc, mask_gt1=mask_gt1
                                                                         , mask_gt2=mask_gt2, epoch=0)
            corr_loss = 0.0
            if args.correlation_loss:
                for m in model.module.SA_modules.named_modules():
                    if isinstance(m[-1], PAConv):
                        kernel_matrice, output_dim, m_dim = m[-1].weightbank, m[-1].output_dim, m[-1].m
                        new_kernel_matrice = kernel_matrice.view(-1, m_dim, output_dim).permute(1, 0, 2).reshape(m_dim,
                                                                                                                 -1)
                        cost_matrice = torch.matmul(new_kernel_matrice, new_kernel_matrice.T) / torch.matmul(
                            torch.sqrt(torch.sum(new_kernel_matrice ** 2, dim=-1, keepdim=True)),
                            torch.sqrt(torch.sum(new_kernel_matrice.T ** 2, dim=0, keepdim=True)))
                        corr_loss += torch.sum(torch.triu(cost_matrice, diagonal=1) ** 2)
                loss = loss + corr_loss * args.paloss_factor
                corr_loss_sum += corr_loss.item()
            test_loss += loss.item()
            gt_loss_sum += gt_loss.item()
            mask_loss_sum += mask_loss.item()
            smooth_loss_sum += smooth_loss.item()

            # record network acc
            test_acc += Miccai_absolute_strength_evaluate_result(l_pred_xyz[0],
                                                                 gt_pc, mask_gt1)
            normalized_acc += Miccai_absolute_strength_evaluate_result(l_pred_xyz[0], nor_gt_pc, mask_gt1)[-1]
            test_mask_acc += Mask_evaluate_result(torch.sigmoid(l_pred_mask1[0]), mask_gt1)
    normalized_acc = normalized_acc / (index + 1) * 100
    test_acc = test_acc / (index + 1) * 100
    test_mask_acc = test_mask_acc / (index + 1) * 100
    test_precise_mask_acc = test_precise_mask_acc / (index + 1) * 100
    return test_loss, [test_acc, test_mask_acc, test_precise_mask_acc, normalized_acc], [gt_loss_sum, mask_loss_sum,
                                                                                     smooth_loss_sum, corr_loss_sum]


def train(starting_epoch, model, optimizer, scheduler, io, writer, args):
    # =========== Dataloader =================
    if args.dataset == 'normal':
        args.train_loader = DataLoader(
            MaskMICCAIMutualNormalized(partition='train', num_points=args.num_points, color_aug=args.color_aug,
                                       root=args.data_dir), num_workers=8, batch_size=args.train_batch, shuffle=True,
            drop_last=True)
        args.test_loader = DataLoader(
            MaskMICCAIMutualNormalized(partition='test', num_points=args.num_points, color_aug=args.color_aug,
                                       root=args.data_dir), num_workers=8, batch_size=args.test_batch, shuffle=True,
            drop_last=True)
    else:
        args.train_loader = DataLoader(
            MaskMICCAIMutualNormalizedSpecial(partition='train', num_points=args.num_points, color_aug=args.color_aug,
                                              root=args.data_dir), num_workers=8, batch_size=args.train_batch,
            shuffle=True, drop_last=True)
        args.test_loader = DataLoader(
            MaskMICCAIMutualNormalizedSpecial(partition='test', num_points=args.num_points, color_aug=args.color_aug,
                                              root=args.data_dir), num_workers=8, batch_size=args.test_batch,
            shuffle=True, drop_last=True)
    best_acc = 1e-5
    out_info = CfgNode()
    for epoch in range(starting_epoch, args.epochs):
        # Train
        out_info.start_time = datetime.datetime.now()
        out_info.loss, out_info.acc, out_info.loss_sum = train_one_epoch(model, optimizer, epoch, args)
        out_info.end_time = datetime.datetime.now()
        writer_info(out_info, epoch, iol=io, writerl=writer, type="Train")

        # Test
        out_info.start_time = datetime.datetime.now()
        out_info.test_loss, out_info.test_acc, out_info.test_loss_sum = test_one_epoch(model.eval(), args)
        out_info.end_time = datetime.datetime.now()
        writer_info(out_info, epoch, iol=io, writerl=writer, type="Test")
        scheduler.step()

        # store
        # if epoch % args.step_epoch == 0:
        #     if isinstance(model, nn.DataParallel):
        #         model_to_save = model.module
        #     else:
        #         model_to_save = model
        #     save_name = os.path.join('checkpoints', args.exp_name,
        #                              "saved_model/train_epoch_{}_end.pth".format(str(epoch)))
        #     torch.save({'epoch': epoch, 'state_dict': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(), }, save_name)
        if out_info.test_acc[0][0] >= best_acc:
            best_acc = out_info.test_acc[0][0]
            torch.save(model.module.state_dict(), 'checkpoints/{}/saved_model/best_model.t7'.format(str(args.exp_name)))
            io.cprint('The new best model is created in {} epoch'.format(epoch))


def main(args, io, writer):
    # ============= Model ===================
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.model_type == "Base":
        model = PTEnetBase(c=6, args=args).to(args.device)
    elif args.model_type == "Base_flow":
        model = PTFlow(c=6, args=args).to(args.device)
    else:
        raise SystemExit('Not impletion')
    model.apply(weight_init)
    # io.cprint(str(model))
    total = sum([param.nelement() for param in model.parameters()])
    io.cprint("Number of parameter: %.2fM" % (total / 1e6))
    # ============= Optimizer ================
    if args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args["MOMENTUM"])
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = None
        exit(0)
    # ============= scheduler ================
    if args.learning_ratedeacy == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    elif args.learning_ratedeacy == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 1e20)
    else:
        scheduler = None
        exit(0)

    # ============= resume ================
    if args.resume is not None:
        resume_filename = os.path.join("../checkpoints", args.exp_name, "saved_model",
                                       "train_epoch_{}_end.pth".format(args.resume))
        io.cprint("Resuming From {}".format(resume_filename))
        checkpoint = torch.load(resume_filename)
        starting_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        starting_epoch = 0

    model = torch.nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # ============= loss ================
    if args.loss_type == "multi_loss":
        args.loss_fn = Multi_Loss(gt_factor=args.gt_factor,
                                  smooth_factor=args.smooth_factor,
                                  mask_factor=args.mask_factor)
    elif args.loss_type == "only_mask":
        args.loss_fn = Only_mask(gt_factor=args.gt_factor,
                                 smooth_factor=args.smooth_factor,
                                 mask_factor=args.mask_factor)
    elif args.loss_type == "only_point":
        args.loss_fn = Only_point(gt_factor=args.gt_factor,
                                  smooth_factor=args.smooth_factor,
                                  mask_factor=args.mask_factor)
    elif args.loss_type == "only_one_loss":
        args.loss_fn = Only_one_loss(gt_factor=args.gt_factor,
                                     smooth_factor=args.smooth_factor,
                                     mask_factor=args.mask_factor)
    elif args.loss_type == "focal_loss":
        args.loss_fn = WeightedFocalLossAll(alpha=args.alpha)
    elif args.loss_type == "one_focal_loss":
        args.loss_fn = one_loss_focal_loss(alpha=args.alpha)
    elif args.loss_type == "Distance_mapLoss":
        args.loss_fn = WeightedDistance_mapLossAll()
    elif args.loss_type == "Weighted_focal_distance_loss":
        args.loss_fn = Weighted_Mask_Loss_All()
    elif args.loss_type == "one_loss_base":
        args.loss_fn = one_loss(gt_factor=args.gt_factor, smooth_factor=args.smooth_factor, mask_factor=args.mask_factor, pos_weight=None)
    elif args.loss_type == "one_loss_focal":
        args.loss_fn = one_loss_focal(gt_factor=args.gt_factor, smooth_factor=args.smooth_factor, mask_factor=args.mask_factor, alpha=args.alpha)
    elif args.loss_type == "one_loss_base_smooth":
        args.loss_fn = one_loss_smooth(gt_factor=args.gt_factor, smooth_factor=args.smooth_factor, mask_factor=args.mask_factor, pos_weight=None)
    elif args.loss_type == "one_gradient_loss":
        args.loss_fn = one_gradient_loss(gt_factor=args.gt_factor,smooth_factor=args.smooth_factor, mask_factor=args.mask_factor,
                                         pos_weight=None)
    elif args.loss_type == "one_loss_dual":
        args.loss_fn = one_loss_dual(gt_factor=args.gt_factor,smooth_factor=args.smooth_factor, mask_factor=args.mask_factor, pos_weight=None)
    elif args.loss_type == "one_loss_base_weight":
        args.loss_fn = one_loss(gt_factor=args.gt_factor, smooth_factor=args.smooth_factor, pos_weight=0.1111)
    else:
        raise SystemExit('Loss not impletion')

    if args.train:
        train(starting_epoch, model, optimizer, scheduler, io, writer, args)
    else:
        print("Hello")


if __name__ == "__main__":
    args_g, io_g, writer_g = init()
    main(args_g, io_g, writer_g)
    print('FINISH')
