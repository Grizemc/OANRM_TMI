# -*- coding: utf-8 -*-
# @Time : 2022/5/8 15:35
# @Author : 8515
# @File : loss.py
import random

import torch
from lib.pointops.functions import pointops
import torch.nn as nn
import torch.nn.functional as F


class Only_point:
    """
        仅对mask重叠区域的点进行邻域约束，仅对 最高层级的点 约束mask
    """

    def __init__(self, gt_factor=0.0001, smooth_factor=10.0, mask_factor=10.0, multi_ratio=0.9, pos_weight_num=0.111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.mask_loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))
        self.mask_loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))

    def losscal(self, l_pc1, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean()
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean()
        smooth_loss = gt_loss
        mask_loss_sum = gt_loss
        gt_loss_sum = gt_loss
        loss = gt_loss
        # l_gt_pc = [gt_pc]
        # l_gt_mask1 = [mask_gt1]
        # l_gt_mask2 = [mask_gt2]
        # for i in range(len(l_idx1)):
        #     l_gt_pc.append(torch.gather(l_gt_pc[i], 1, l_idx1[i].long().unsqueeze(-1).expand(-1, -1, 3)))
        #     l_gt_mask1.append(torch.gather(l_gt_mask1[i], 1, l_idx1[i].long()))
        #     l_gt_mask2.append(torch.gather(l_gt_mask2[i], 1, l_idx2[i].long()))
        # loss = 0.
        # for i in range(len(l_pc1)):
        #     pred = l_pred[i][l_gt_mask1[i].squeeze()]
        #     gt_pc = l_gt_pc[i][l_gt_mask1[i].squeeze()]
        #     gt_loss = ((pred - gt_pc) ** 2).sum(dim=1).mean()
        #     if i == 0:
        #         gt_loss_sum = gt_loss
        #     elif i == 1:
        #         mask_loss_sum = gt_loss
        #     elif i == 2:
        #         smooth_loss = gt_loss
        #     loss += gt_loss * (self.multi_ratio ** (i + 1))
        return loss, gt_loss_sum.detach(), mask_loss_sum.detach(), smooth_loss.detach()


class Only_mask:
    """
        仅对mask重叠区域的点进行邻域约束，仅对 最高层级的点 约束mask
    """

    def __init__(self, gt_factor=0.0001, smooth_factor=10.0, mask_factor=10.0, multi_ratio=0.9, pos_weight_num=0.111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.mask_loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))
        self.mask_loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))

    def losscal(self, l_pc1, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        l_gt_pc = [gt_pc]
        l_gt_mask1 = [mask_gt1.float()]
        l_gt_mask2 = [mask_gt2.float()]
        for i in range(len(l_idx1)):
            l_gt_pc.append(torch.gather(l_gt_pc[i], 1, l_idx1[i].long().unsqueeze(-1).expand(-1, -1, 3)))
            l_gt_mask1.append(torch.gather(l_gt_mask1[i], 1, l_idx1[i].long()))
            l_gt_mask2.append(torch.gather(l_gt_mask2[i], 1, l_idx2[i].long()))
        loss = 0.
        for i in range(len(l_pc1)):
            mask_loss = self.mask_loss1(l_pred_mask1[i].squeeze(), l_gt_mask1[i]) * self.mask_factor
            mask_loss += self.mask_loss2(l_pred_mask2[i].squeeze(), l_gt_mask2[i]) * self.mask_factor
            loss += mask_loss * (self.multi_ratio ** (i + 1))
            if i == 0:
                gt_loss_sum = mask_loss
            elif i == 1:
                mask_loss_sum = mask_loss
            elif i == 2:
                smooth_loss = mask_loss
        return loss, gt_loss_sum.detach(), mask_loss_sum.detach(), smooth_loss.detach()

    ''


class Multi_Loss:
    """
        仅对mask重叠区域的点进行邻域约束，仅对 最高层级的点 约束mask
    """

    def __init__(self, gt_factor=0.0001, smooth_factor=10.0, mask_factor=10.0, multi_ratio=0.9, pos_weight_num=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.mask_loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))
        self.mask_loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))

    def losscal(self, l_pc1, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        l_gt_pc = [gt_pc]
        l_gt_mask1 = [mask_gt1]
        l_gt_mask2 = [mask_gt2]
        for i in range(len(l_idx1)):
            l_gt_pc.append(torch.gather(l_gt_pc[i], 1, l_idx1[i].long().unsqueeze(-1).expand(-1, -1, 3)))
            l_gt_mask1.append(torch.gather(l_gt_mask1[i], 1, l_idx1[i].long()))
            l_gt_mask2.append(torch.gather(l_gt_mask2[i], 1, l_idx2[i].long()))
        loss = 0.
        gt_loss_sum = 0.
        mask_loss_sum = 0.
        for i in range(len(l_pc1)):
            pc1 = l_pc1[i]
            gt = l_gt_pc[i]
            pred = l_pred[i]
            # gt_loss = (((gt - pred) ** 2).sum(dim=2) * l_gt_mask1[i]).mean() * self.gt_a
            gt_loss = (((gt[l_gt_mask1[i]] - pred[l_gt_mask1[i]]) ** 2).sum(dim=1)).mean() * self.gt_a
            gt_loss_sum += gt_loss
            if i == 0:  # 有正则项,mask项
                smooth_loss = 0.
                for j in range(pred.shape[0]):
                    temp_mask = l_gt_mask1[i][j, :].bool()
                    temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
                    temp_pred = pred[j][temp_mask].unsqueeze(0)
                    smooth_loss += NewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
                smooth_loss = smooth_loss / pred.shape[0]
                mask_loss = self.mask_loss1(l_pred_mask1[i].squeeze(), l_gt_mask1[i].float()) * self.mask_factor
                mask_loss += self.mask_loss2(l_pred_mask2[i].squeeze(), l_gt_mask2[i].float()) * self.mask_factor
                mask_loss_sum += mask_loss
                loss += (gt_loss + smooth_loss + mask_loss_sum) * (self.multi_ratio ** (i + 1))
            else:  # 无正则项
                mask_loss = self.mask_loss1(l_pred_mask1[i].squeeze(), l_gt_mask1[i].float()) * self.mask_factor
                mask_loss += self.mask_loss2(l_pred_mask2[i].squeeze(), l_gt_mask2[i].float()) * self.mask_factor
                mask_loss_sum += mask_loss
                loss += (mask_loss_sum + gt_loss) * (self.multi_ratio ** (i + 1))
        return loss, gt_loss_sum.detach(), mask_loss_sum.detach(), smooth_loss.detach()


class Only_one_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=0.0001, smooth_factor=10.0, mask_factor=10.0, multi_ratio=0.9, pos_weight_num=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.mask_loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))
        self.mask_loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_num))

    def losscal(self, l_pc1, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += NewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        mask_loss = self.mask_loss1(pred_mask1.squeeze(), mask_gt1.float()) * self.mask_factor
        mask_loss += self.mask_loss2(pred_mask2.squeeze(), mask_gt2.float()) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()





class one_loss_smooth:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, pos_weight=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1

    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        smooth_loss = NewSmoothLoss(pc1, pred, smooth_num=7).mean() * self.smooth_a
        if self.pos_weight is not None:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, self.pos_weight) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, self.pos_weight) * self.mask_factor
        else:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, 1) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, 1) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()


class one_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, pos_weight=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1

    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += FlowNewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        if self.pos_weight is not None:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, self.pos_weight) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, self.pos_weight) * self.mask_factor
        else:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, 1) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, 1) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()

class  one_loss_bce_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, pos_weight, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.pos_weight = pos_weight
        if self.pos_weight == 'None':
            self.mask_loss1 = torch.nn.BCEWithLogitsLoss()
            self.mask_loss2 = torch.nn.BCEWithLogitsLoss()
        else:
            self.mask_loss1 = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor(pos_weight))
            self.mask_loss2 = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor(pos_weight))
    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += FlowNewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        mask_loss = self.mask_loss1(pred_mask1.squeeze(), mask_gt1.type(torch.float)) * self.mask_factor
        mask_loss += self.mask_loss2(pred_mask2.squeeze(), mask_gt2.type(torch.float)) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()
class one_loss_focal:
    """
        不需要使用的函数，仅用于防止报错
    """

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, alpha=0., gamma=0.):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.focal_loss1 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_loss2 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += FlowNewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        mask_loss = self.focal_loss1(pred_mask1.squeeze(), mask_gt1) * self.mask_factor
        mask_loss += self.focal_loss2(pred_mask2.squeeze(), mask_gt2) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()

class one_loss_focal_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, alpha=0., gamma=0.):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        self.focal_loss1 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_loss2 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += FlowNewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        mask_loss = self.focal_loss1(pred_mask1.squeeze(), mask_gt1) * self.mask_factor
        mask_loss += self.focal_loss2(pred_mask2.squeeze(), mask_gt2) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()
class one_gradient_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, pos_weight=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1

    def gradient_and_det_loss(self, pred, xyz1, smooth_num):
        B = pred.shape[0]
        flow = pred - xyz1
        # idx1 = pointops.ballquery(0.05, smooth_num, xyz1, xyz1)
        idx1 = pointops.knnquery(6, xyz1, xyz1)
        idx1 = idx1[:, :, 1:].contiguous()
        neigh_flow = pointops.grouping(flow.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3, 1).contiguous()
        neigh_xyz1 = pointops.grouping(xyz1.transpose(1, 2).contiguous(), idx1.int()).permute(0, 2, 3, 1).contiguous()
        relative_xyz = neigh_xyz1 - xyz1.unsqueeze(-2)
        relative_flow = neigh_flow - flow.unsqueeze(-2)
        graident_list = []
        error_sum = 0.
        for i in range(3):
            flow_neigh_single = relative_flow[:, :, :, i].reshape(B, relative_flow.shape[1], relative_flow.shape[2], 1)
            xyz_relative_trans = relative_xyz.permute(0, 1, 3, 2).contiguous()
            coefficients_front = torch.linalg.pinv(torch.matmul(xyz_relative_trans, relative_xyz) + 1e-10)
            coefficients = torch.matmul(torch.matmul(coefficients_front, xyz_relative_trans), flow_neigh_single)
            flow_neigh_single_restore = torch.matmul(relative_xyz, coefficients)
            single_error = (flow_neigh_single_restore - flow_neigh_single).mean()
            graident_list.append(coefficients)
            error_sum += single_error
        jacob_mat = torch.cat((graident_list[0], graident_list[1], graident_list[2]), dim=3).transpose(2,
                                                                                                       3).contiguous()
        smooth_gradient_loss = jacob_mat.norm(dim=-1).mean()
        return smooth_gradient_loss

    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        if epoch != 0 and epoch % 40 == 0:
            self.smooth_a = self.smooth_a * 0.5
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = self.gradient_and_det_loss(pc1, pred, smooth_num=5) * self.smooth_a
        if self.pos_weight is not None:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, self.pos_weight) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, self.pos_weight) * self.mask_factor
        else:
            mask_loss = Distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, 1) * self.mask_factor
            mask_loss += Distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, 1) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()


class one_loss_dual:
    """
        仅对最高层添加各种约束
    """

    def dual_distance_map_loss(self, pcd, mask_pred, mask_gt, pos_weight):
        B = mask_pred.shape[0]
        Distance_map_loss_sum = 0.
        mask_pred = torch.sigmoid(mask_pred) + 1e-10
        for i in range(B):
            single_pcd = pcd[i, :]
            single_mask_pred = mask_pred[i, :]
            single_mask_gt = mask_gt[i, :]

            true_point = single_pcd[single_mask_gt]
            true_mask_pred = single_mask_pred[single_mask_gt]
            false_point = single_pcd[~single_mask_gt]
            false_mask_pred = single_mask_pred[~single_mask_gt]
            distance = torch.norm(true_point.unsqueeze(dim=1) - false_point.unsqueeze(dim=0), dim=2)
            true_distance = distance.min(dim=1)[0]
            false_distance = distance.min(dim=0)[0]
            true_sort_index = torch.sort(true_distance, dim=0, descending=True)[1]
            false_sort_index = torch.sort(false_distance, dim=0, descending=True)[1]

            true_factor = (torch.arange(0.1, 1, 0.9 / true_point.shape[0]) + 1).to(true_point.device)[
                          :true_point.shape[0]]
            false_factor = (torch.arange(0.1, 1, 0.9 / false_point.shape[0]) + 1).to(true_point.device)[
                           :false_point.shape[0]]
            mask_factor = torch.cat((true_factor, false_factor), dim=0)
            BCE_mask_pred = torch.cat((true_mask_pred[true_sort_index], false_mask_pred[false_sort_index]), dim=0)
            BCE_mask_gt = torch.cat((torch.ones_like(true_mask_pred), torch.zeros_like(false_mask_pred)), dim=0)
            Bce_loss = F.binary_cross_entropy(input=BCE_mask_pred, target=BCE_mask_gt, reduction='none')
            Distance_map_loss_sum += (Bce_loss * mask_factor).mean()
        return Distance_map_loss_sum / B

        # Bce_loss = F.binary_cross_entropy_with_logits(mask_pred_single, mask_gt_single.bool(),
        #                                                                 reduction='none', pos_weight=0.111)

    def __init__(self, gt_factor=8.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, pos_weight=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1

    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += NewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        if self.pos_weight is not None:
            mask_loss = self.dual_distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1,
                                                    self.pos_weight) * self.mask_factor
            mask_loss += self.dual_distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2,
                                                     self.pos_weight) * self.mask_factor
        else:
            mask_loss = self.dual_distance_map_loss(pc1, pred_mask1.squeeze(), mask_gt1, 1) * self.mask_factor
            mask_loss += self.dual_distance_map_loss(pc2, pred_mask2.squeeze(), mask_gt2, 1) * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()


class large_one_loss:
    """
        仅对最高层添加各种约束
    """

    def __init__(self, gt_factor=1.0, smooth_factor=10.0, mask_factor=1.0, multi_ratio=0.9, pos_weight=0.1111):
        self.gt_a = gt_factor
        self.smooth_a = smooth_factor
        self.multi_ratio = multi_ratio
        self.mask_factor = mask_factor
        if pos_weight is not None:
            self.pos_weight = pos_weight
        else:
            self.pos_weight = 1

    def losscal(self, l_pc1, l_pred, pc2, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2,
                mask_weight_1, mask_weight_2, epoch):
        pred = l_pred[0]
        pred_mask1 = l_pred_mask1[0]
        pred_mask2 = l_pred_mask2[0]
        pc1 = l_pc1[0]
        gt_loss = torch.norm(gt_pc[mask_gt1] - pred[mask_gt1], dim=1).mean() * self.gt_a
        # gt_loss = (((gt_pc[mask_gt1] - pred[mask_gt1]) ** 2).sum(dim=1)).mean() * self.gt_a
        smooth_loss = 0.
        for j in range(pred.shape[0]):
            temp_mask = mask_gt1[j, :].bool()
            temp_pc1 = pc1[j][temp_mask].unsqueeze(0)
            temp_pred = pred[j][temp_mask].unsqueeze(0)
            smooth_loss += NewSmoothLoss(temp_pc1, temp_pred, smooth_num=7).mean() * self.smooth_a
        smooth_loss = smooth_loss / pred.shape[0]
        temp_mask_loss1 = F.binary_cross_entropy_with_logits(pred_mask1.squeeze(), mask_gt1.float(),
                                                             reduction='none',
                                                             pos_weight=torch.tensor(1))
        mask_loss = (temp_mask_loss1.unsqueeze(-1) * (mask_weight_1)).mean() * self.mask_factor
        temp_mask_loss2 = F.binary_cross_entropy_with_logits(pred_mask2.squeeze(), mask_gt2.float(),
                                                             reduction='none',
                                                             pos_weight=torch.tensor(1))
        mask_loss += (temp_mask_loss2.unsqueeze(-1) * (mask_weight_2)).mean() * self.mask_factor
        loss = gt_loss + smooth_loss + mask_loss
        return loss, gt_loss.detach(), mask_loss.detach(), smooth_loss.detach()


def Distance_map_loss(pcd, mask_pred, mask_gt, pos_weight):
    B = mask_pred.shape[0]
    Distance_map_loss_sum = 0.
    for i in range(B):
        single_pcd = pcd[i, :]
        single_mask_pred = mask_pred[i, :]
        single_mask_gt = mask_gt[i, :]
        if single_mask_gt.sum() == single_mask_gt.shape[0]:
            Bce_loss = F.binary_cross_entropy_with_logits(input=single_mask_pred, target=single_mask_gt.float())
            Distance_map_loss_sum += Bce_loss
        else:
            true_point = single_pcd[single_mask_gt]
            true_mask_pred = single_mask_pred[single_mask_gt]
            false_point = single_pcd[~single_mask_gt]
            false_mask_pred = single_mask_pred[~single_mask_gt]
            distance = torch.norm(true_point.unsqueeze(dim=1) - false_point.unsqueeze(dim=0), dim=2)
            true_distance = distance.min(dim=1)[0]
            false_distance = distance.min(dim=0)[0]
            true_sort_index = torch.sort(true_distance, dim=0, descending=True)[1]
            false_sort_index = torch.sort(false_distance, dim=0, descending=True)[1]
            true_factor = (torch.arange(0.1, 1, 0.9 / true_point.shape[0]) + 1).to(true_point.device)[:true_point.shape[0]]
            false_factor = (torch.arange(0.1, 1, 0.9 / false_point.shape[0]) + 1).to(true_point.device)[:false_point.shape[0]]
            mask_factor = torch.cat((true_factor, false_factor), dim=0)
            BCE_mask_pred = torch.cat((true_mask_pred[true_sort_index], false_mask_pred[false_sort_index]), dim=0)
            BCE_mask_gt = torch.cat((torch.ones_like(true_mask_pred), torch.zeros_like(false_mask_pred)), dim=0)
            Bce_loss = F.binary_cross_entropy_with_logits(BCE_mask_pred, BCE_mask_gt, reduction='none',
                                                          pos_weight=torch.tensor(pos_weight))
            Distance_map_loss_sum += (Bce_loss * mask_factor).mean()
    return Distance_map_loss_sum / B

    # Bce_loss = F.binary_cross_entropy_with_logits(mask_pred_single, mask_gt_single.bool(),
    #                                                                 reduction='none', pos_weight=0.111)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.1, gamma=2):
        """
        Binary weighted cross entropy loss

        Parameters
        ----------
        alpha: weight of the positive sample
        gamma
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([1 - alpha, alpha]).cuda()
        self.gamma = gamma
        self.mask_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        targets = targets.type(torch.float)
        BCE_loss = self.mask_loss(inputs, targets).view(-1)
        at = self.alpha.gather(0, targets.data.view(-1).type(torch.long))
        pt = torch.exp(-BCE_loss).view(-1)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# Final_BCELoss
class WeightedFocalLossAll(nn.Module):
    def __init__(self, alpha=.889, gamma=2):
        """
        Binary weighted cross entropy loss

        Parameters
        ----------
        alpha: Overlap specific gravity
        gamma
        """
        super(WeightedFocalLossAll, self).__init__()
        self.focal_loss1 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_loss2 = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def losscal(self, l_pc1, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2):
        focal_loss = self.focal_loss1(l_pred_mask1[0].squeeze(), mask_gt1.float())
        focal_loss += self.focal_loss1(l_pred_mask2[0].squeeze(), mask_gt2.float())
        return focal_loss, focal_loss, focal_loss, focal_loss


class Weighted_Mask_Loss_All(nn.Module):
    def __init__(self, alpha=.111, gamma=2):
        super(Weighted_Mask_Loss_All, self).__init__()
        self.focal_loss1 = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_loss2 = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def losscal(self, l_pc1, pc2, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2):
        distance_map_loss = Distance_map_loss(l_pc1[0], l_pred_mask1[0].squeeze(), mask_gt1)
        distance_map_loss += Distance_map_loss(pc2, l_pred_mask2[0].squeeze(), mask_gt2)
        focal_loss = self.focal_loss1(l_pred_mask1[0].squeeze(), mask_gt1.float())
        focal_loss += self.focal_loss1(l_pred_mask2[0].squeeze(), mask_gt2.float())
        return focal_loss + distance_map_loss, distance_map_loss, focal_loss, distance_map_loss


class WeightedDistance_mapLossAll(nn.Module):
    def __init__(self, alpha=.111, gamma=2):
        """
        Binary weighted cross entropy loss

        Parameters
        ----------
        alpha: Overlap specific gravity
        gamma
        """
        super(WeightedDistance_mapLossAll, self).__init__()

    def losscal(self, l_pc1, pc2, l_pred, l_idx1, l_idx2, l_pred_mask1, l_pred_mask2, gt_pc, mask_gt1, mask_gt2):
        distance_map_loss = Distance_map_loss(l_pc1[0], l_pred_mask1[0].squeeze(), mask_gt1)
        distance_map_loss += Distance_map_loss(pc2, l_pred_mask2[0].squeeze(), mask_gt2)
        return distance_map_loss, distance_map_loss, distance_map_loss, distance_map_loss


def FlowNewSmoothLoss(pc1, pred, smooth_num=7):
    """
    使用点之间的绝对距离约束
    """
    # B N N
    # Smoothness
    topk_idx = pointops.knnquery(smooth_num, pc1, pc1)[:, :, 1:] .contiguous() # remove center point self
    flow = pred - pc1
    source_grouped_point = pointops.grouping(pc1.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    source_distance = torch.norm(source_grouped_point - pc1.unsqueeze(2), dim=3)
    # Reciprocal
    dist_recip = 1.0 / (source_distance + 1e-10)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    flow_neigh = pointops.grouping(flow.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    flow_self = flow.unsqueeze(dim=-2)
    smooth_loss = torch.norm((flow_neigh - flow_self), dim=-1) * weight
    return smooth_loss


def NewSmoothLoss(pc1, pred, smooth_num=7):
    """
    使用点之间的绝对距离约束
    """
    # B N N
    # Smoothness
    topk_idx = pointops.knnquery(smooth_num, pc1, pc1)[:, :, 1:]  # remove center point self
    topk_idx = topk_idx[:, :, 1:].contiguous()
    source_grouped_point = pointops.grouping(pc1.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    source_distance = torch.norm(source_grouped_point - pc1.unsqueeze(2), dim=3)
    # Reciprocal
    dist_recip = 1.0 / (source_distance + 1e-10)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    source_grouped_point = pointops.grouping(pc1.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    source_distance = torch.norm(source_grouped_point - pc1.unsqueeze(2), dim=3)

    pred_grouped_point = pointops.grouping(pred.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    pred_distance = torch.norm(pred_grouped_point - pred.unsqueeze(2), dim=3)

    smooth_loss = (pred_distance - source_distance).abs() * weight
    return smooth_loss


def square_distance(pc1, pc2):
    """
    Calculate Euclid distance between each two points.

    pc1^T * pc2 = xn * xm + yn * ym + zn * zm；
    sum(pc1^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(pc2^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(pc1**2,dim=-1)c2+sum(pc2**2,dim=-1)-2*pc1^T*p

    Input:
        pc1: source points, [B, N, C]
        pc2: target points, [B, M, C]
    Output:
        dist: Tensor per-point square distance, [B, N, M]
    """
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    dist = -2 * torch.matmul(pc1, pc2.permute(0, 2, 1))
    dist += torch.sum(pc1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(pc2 ** 2, -1).view(B, 1, M)
    return dist


def computeSmooth(pc1, pred_flow, smooth_num):
    """
    pc1: B N 3
    pred_flow: B N 3
    仅使用了点云直之间的相对坐标，而不是距离。
    """
    dist = F.relu(square_distance(pc1, pc1) - 1e-10)
    # B N N
    # Smoothness
    sqrdist_small, topk_idx = torch.topk(dist, smooth_num, dim=2, largest=False, sorted=False)
    sqrdist_small = sqrdist_small[:, :, 1:]  # remove point self
    topk_idx = topk_idx[:, :, 1:]
    # Reciprocal
    dist_recip = 1.0 / (sqrdist_small + 1e-10)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    grouped_flow = pointops.grouping(pred_flow.permute(0, 2, 1).contiguous(), topk_idx.int()).permute(0, 2, 3, 1)
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim=3) * weight
    # diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim=3)
    diff_flow = diff_flow.sum(dim=2) / (smooth_num - 1)
    return diff_flow


if __name__ == "__main__":
    # xyz1 = torch.tensor([[0],[0],[1],[1.]]).cuda()
    # xyz2 = torch.tensor([[0],[1],[0],[1.]]).cuda()
    # fn_WeightedFocalLoss = WeightedFocalLoss()
    # result = fn_WeightedFocalLoss(xyz1,xyz2)
    import numpy as np

    xyz1 = torch.randn(3, 4096, 3).cuda()
    predmask = torch.tensor(np.random.choice(a=[False, True], size=(3, 4096))).cuda()
    gtmask = torch.tensor(np.random.choice(a=[False, True], size=(3, 4096))).cuda()
    fn_WeightedFocalLoss = Distance_map_loss(xyz1, predmask, gtmask)
