#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Register one point cloud by split:
- source: first 2/3 points
- target: last 2/3 points
from a single input ply file.
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d
import torch

from util.util import load_cfg_from_cfg_file
from run_itags_mix_pair_registration import (
    PairPostLoss,
    build_model,
    compute_fpfh_matches,
    compute_nn_gt,
    gaussian_postprocess,
    load_model_weights,
    pcd2_gt_normalization,
    set_seed,
    symmetric_nn_distance,
    voxel_and_sample,
    xyz_restore,
    xyz_to_normalization,
)


def load_and_split_ply(ply_path: str):
    if not os.path.exists(ply_path):
        raise RuntimeError(f"PLY not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors).astype(np.float32)
    if pts.shape[0] < 100:
        raise RuntimeError(f"Too few points in ply: {pts.shape[0]}")
    if cols.shape[0] != pts.shape[0]:
        cols = np.ones_like(pts, dtype=np.float32) * 0.5

    n = pts.shape[0]
    src_end = (2 * n) // 3
    tgt_start = n // 3
    source = pts[:src_end]
    source_c = cols[:src_end]
    target = pts[tgt_start:]
    target_c = cols[tgt_start:]
    split_meta = {
        "total_points": int(n),
        "source_range": [0, int(src_end)],
        "target_range": [int(tgt_start), int(n)],
    }
    return source, source_c, target, target_c, split_meta


def run(args):
    cfg = load_cfg_from_cfg_file(args.config)
    if args.post_lr is None:
        args.post_lr = float(cfg.post_lr)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    source_raw, source_col_raw, target_raw, target_col_raw, split_meta = load_and_split_ply(args.ply_path)
    source_ds, source_col = voxel_and_sample(source_raw, source_col_raw, args.voxel_size, args.num_points, rng)
    target_ds, target_col = voxel_and_sample(target_raw, target_col_raw, args.voxel_size, args.num_points, rng)

    # Build pseudo gt / normalized inputs
    mask_gt_pc_raw = compute_nn_gt(source_ds, target_ds)
    source_norm, _ = xyz_to_normalization(source_ds)
    target_norm, mask_gt_pc_norm, relax_ratio = pcd2_gt_normalization(target_ds, mask_gt_pc_raw)

    pseudo1, pseudo2 = compute_fpfh_matches(
        source_norm,
        source_col,
        target_norm,
        target_col,
        args.fpfh_thresholds,
        args.max_fpfh_corr,
    )
    gt_pseudo = target_norm[pseudo2].astype(np.float32)
    
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")
    print(
        f"[INFO] source_points={source_norm.shape[0]}, target_points={target_norm.shape[0]}, "
        f"sample_num_points={args.num_points}, pseudo_corr={len(pseudo1)}"
    )

    source_t = torch.from_numpy(source_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_t = torch.from_numpy(target_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    source_col_t = torch.from_numpy(source_col).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_col_t = torch.from_numpy(target_col).unsqueeze(0).to(device=device, dtype=torch.float32)
    pseudo1_t = torch.from_numpy(pseudo1).unsqueeze(0).to(device=device, dtype=torch.long)
    gt_pseudo_t = torch.from_numpy(gt_pseudo).unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model(cfg, device)
    model = load_model_weights(model, args.best_model, device)

    with torch.no_grad():
        model.eval()
        pre_out = model(source_t, target_t, source_col_t, target_col_t)
        pred_before = pre_out[1][0][0].detach().cpu().numpy().astype(np.float32)
        pred_mask_before = torch.sigmoid(pre_out[4][0])[0, :, 0].detach().cpu().numpy().astype(np.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.post_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    post_loss = PairPostLoss(
        gt_factor=args.gt_factor,
        smooth_factor=args.smooth_factor,
        pseudo_factor=args.pseudo_factor,
    )

    best_eval = float("inf")
    best_pred = pred_before.copy()
    best_mask = pred_mask_before.copy()
    best_epoch = 0

    for epoch in range(args.pt_epoch):
        optimizer.zero_grad()
        out = model(source_t, target_t, source_col_t, target_col_t)
        total_loss, logs = post_loss(
            l_xyz1=out[0],
            l_pred_xyz=out[1],
            xyz2=target_t,
            mask_pseudo1=pseudo1_t,
            gt_pseudo=gt_pseudo_t,
        )
        if not torch.isfinite(total_loss):
            print(f"[WARN] non-finite loss at epoch={epoch}, stop fine-tuning.")
            break
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        eval_val = float(logs["chamfer_eval"].item())
        print(
            f"[FT] epoch={epoch:02d} total={float(logs['total'].item()):.6f} "
            f"chamfer={eval_val:.6f} smooth={float(logs['smooth_loss'].item()):.6f} "
            f"pseudo={float(logs['pseudo_loss'].item()):.6f}"
        )
        if eval_val < best_eval:
            best_eval = eval_val
            best_epoch = epoch
            best_pred = out[1][0][0].detach().cpu().numpy().astype(np.float32)
            best_mask = torch.sigmoid(out[4][0])[0, :, 0].detach().cpu().numpy().astype(np.float32)

    pred_post = gaussian_postprocess(source_norm, best_pred, knn_num=args.gaussian_knn, sigma=args.gaussian_sigma)

    pred_before_raw = xyz_restore(pred_before, relax_ratio)
    pred_after_raw = xyz_restore(best_pred, relax_ratio)
    pred_post_raw = xyz_restore(pred_post, relax_ratio)
    metric_before = symmetric_nn_distance(pred_before_raw, target_ds)
    metric_after = symmetric_nn_distance(pred_after_raw, target_ds)
    metric_post = symmetric_nn_distance(pred_post_raw, target_ds)

    out_dir = os.path.join(args.output_root, "ply_split_2over3")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "pair_registration_result.npz")
    np.savez_compressed(
        out_npz,
        ply_path=args.ply_path,
        split_meta=json.dumps(split_meta, ensure_ascii=False),
        source_points_raw=source_ds,
        target_points_raw=target_ds,
        source_colors=source_col,
        target_colors=target_col,
        source_points_norm=source_norm,
        target_points_norm=target_norm,
        relax_ratio=relax_ratio,
        mask_gt_pc_norm=mask_gt_pc_norm,
        fpfh_source_idx=pseudo1,
        fpfh_target_idx=pseudo2,
        pred_before_norm=pred_before,
        pred_after_norm=best_pred,
        pred_post_norm=pred_post,
        pred_before_raw=pred_before_raw,
        pred_after_raw=pred_after_raw,
        pred_post_raw=pred_post_raw,
        pred_mask_before=pred_mask_before,
        pred_mask_after=best_mask,
        metric_before=metric_before,
        metric_after=metric_after,
        metric_post=metric_post,
        best_epoch=best_epoch,
    )

    summary = {
        "config": args.config,
        "best_model": args.best_model,
        "ply_path": args.ply_path,
        "split": split_meta,
        "sample_num_points": int(args.num_points),
        "source_points_num": int(source_norm.shape[0]),
        "target_points_num": int(target_norm.shape[0]),
        "voxel_size": float(args.voxel_size),
        "post_lr": float(args.post_lr),
        "pt_epoch": int(args.pt_epoch),
        "best_epoch": int(best_epoch),
        "fpfh_corr_num": int(len(pseudo1)),
        "metric_symmetric_nn_before": float(metric_before),
        "metric_symmetric_nn_after": float(metric_after),
        "metric_symmetric_nn_post": float(metric_post),
        "output_npz": out_npz,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Registration finished.")
    print(f"[DONE] Output directory: {out_dir}")
    print(
        f"[DONE] symmetric_nn: before={metric_before:.6f}, "
        f"after={metric_after:.6f}, post={metric_post:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Register split source/target from one points3D.ply")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/szm/Paconv_730/config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        default="/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/saved_model/best_model.t7",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/sparse_inter_3/0/points3D.ply",
    )
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--num_points", type=int, default=-1, help="<=0 keep all points")
    parser.add_argument("--post_lr", type=float, default=None)
    parser.add_argument("--pt_epoch", type=int, default=10)
    parser.add_argument("--gt_factor", type=float, default=2.0)
    parser.add_argument("--smooth_factor", type=float, default=1.0)
    parser.add_argument("--pseudo_factor", type=float, default=1.0)
    parser.add_argument("--fpfh_thresholds", type=float, nargs="+", default=[0.08, 0.12, 0.2, 0.3])
    parser.add_argument("--max_fpfh_corr", type=int, default=2048)
    parser.add_argument("--gaussian_knn", type=int, default=5)
    parser.add_argument("--gaussian_sigma", type=float, default=0.2)
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/ply_split_registration",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
