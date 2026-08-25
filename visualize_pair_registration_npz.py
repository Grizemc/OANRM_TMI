#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


DEFAULT_NPZ_ROOT = "/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/itags_mix_pair_registration"


def to_nx3(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Point cloud must be Nx3, got shape={arr.shape}")
    return arr.astype(np.float32)


def to_color_nx3(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Color must be Nx3, got shape={arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_source_target_pred(
    npz_path: str, pred_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    source = None
    target = None
    pred = None
    used_pred_key = ""
    source_color = None
    target_color = None

    source_candidates = [
        "source_points_raw",
        "points1",
        "mask_point1",
    ]
    target_candidates = [
        "target_points_raw",
        "points2",
        "mask_point2",
    ]
    pred_candidates_auto = [
        "pred_post_raw",
        "pred_after_raw",
        "pred_before_raw",
        "pred_xyz",
    ]
    source_color_candidates = [
        "source_colors",
        "colors1",
        "mask_color1",
    ]
    target_color_candidates = [
        "target_colors",
        "colors2",
        "mask_color2",
    ]

    for k in source_candidates:
        if k in keys:
            source = to_nx3(data[k])
            break
    for k in target_candidates:
        if k in keys:
            target = to_nx3(data[k])
            break
    for k in source_color_candidates:
        if k in keys:
            try:
                source_color = to_color_nx3(data[k])
                break
            except Exception:
                pass
    for k in target_color_candidates:
        if k in keys:
            try:
                target_color = to_color_nx3(data[k])
                break
            except Exception:
                pass

    if pred_key == "auto":
        for k in pred_candidates_auto:
            if k in keys:
                pred = to_nx3(data[k])
                used_pred_key = k
                break
    else:
        if pred_key not in keys:
            raise KeyError(f"pred_key='{pred_key}' not found. available keys: {sorted(keys)}")
        pred = to_nx3(data[pred_key])
        used_pred_key = pred_key

    if source is None:
        raise KeyError(f"Cannot find source cloud key in npz. available keys: {sorted(keys)}")
    if target is None:
        raise KeyError(f"Cannot find target cloud key in npz. available keys: {sorted(keys)}")
    if pred is None:
        raise KeyError(f"Cannot find predicted cloud key in npz. available keys: {sorted(keys)}")

    if source_color is not None and source_color.shape[0] != source.shape[0]:
        source_color = None
    if target_color is not None and target_color.shape[0] != target.shape[0]:
        target_color = None

    return source, target, pred, source_color, target_color, used_pred_key


def make_cloud(
    points: np.ndarray,
    color: Tuple[float, float, float],
    colors: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color(color)
    return pcd


def visualize_overlay(
    source: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    source_color: Optional[np.ndarray],
    target_color: Optional[np.ndarray],
    pred_color: Optional[np.ndarray],
) -> None:
    source_pcd = make_cloud(source, (1.0, 0.2, 0.2), source_color)   # red fallback
    target_pcd = make_cloud(target, (0.2, 1.0, 0.2), target_color)   # green fallback
    pred_pcd = make_cloud(pred, (0.2, 0.4, 1.0), pred_color)         # blue fallback
    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd, pred_pcd],
        window_name="Overlay: source / target / pred",
    )


def visualize_separate(
    source: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    source_color: Optional[np.ndarray],
    target_color: Optional[np.ndarray],
    pred_color: Optional[np.ndarray],
) -> None:
    all_pts = np.vstack([source, target, pred])
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    span = np.linalg.norm(bbox_max - bbox_min)
    if span <= 1e-8:
        span = 1.0
    shift = np.array([span * 1.4, 0.0, 0.0], dtype=np.float32)

    source_pcd = make_cloud(source.copy(), (1.0, 0.2, 0.2), source_color)
    target_pcd = make_cloud(target.copy() + shift, (0.2, 1.0, 0.2), target_color)
    pred_pcd = make_cloud(pred.copy() + shift * 2.0, (0.2, 0.4, 1.0), pred_color)
    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd, pred_pcd],
        window_name="Separate: source | target | pred",
    )


def save_ply(
    save_dir: str,
    source: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    source_color: Optional[np.ndarray],
    target_color: Optional[np.ndarray],
    pred_color: Optional[np.ndarray],
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_point_cloud(
        os.path.join(save_dir, "source.ply"),
        make_cloud(source, (1.0, 0.2, 0.2), source_color),
    )
    o3d.io.write_point_cloud(
        os.path.join(save_dir, "target.ply"),
        make_cloud(target, (0.2, 1.0, 0.2), target_color),
    )
    o3d.io.write_point_cloud(
        os.path.join(save_dir, "pred.ply"),
        make_cloud(pred, (0.2, 0.4, 1.0), pred_color),
    )


def resolve_npz_path(npz_path: str, npz_root: str) -> str:
    if npz_path:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"npz file not found: {npz_path}")
        return npz_path

    pattern = os.path.join(npz_root, "src_*_tgt_*", "pair_registration_result.npz")
    candidates = glob.glob(pattern)
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No pair_registration_result.npz found under default root: {npz_root}"
        )
    # Use latest modified result
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize source/target/pred point clouds from pair_registration_result.npz")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="",
        help="Path to pair_registration_result.npz. If empty, auto-find latest result in --npz_root",
    )
    parser.add_argument(
        "--npz_root",
        type=str,
        default=DEFAULT_NPZ_ROOT,
        help="Default root for auto-search when --npz_path is empty",
    )
    parser.add_argument(
        "--pred_key",
        type=str,
        default="auto",
        help="Prediction key in npz. default=auto (pred_post_raw -> pred_after_raw -> pred_before_raw -> pred_xyz)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlay",
        choices=["overlay", "separate"],
        help="overlay: three clouds in one coordinate frame; separate: three clouds side-by-side",
    )
    parser.add_argument(
        "--save_ply_dir",
        type=str,
        default="",
        help="Optional directory to export source/target/pred ply files",
    )
    parser.add_argument(
        "--use_original_color",
        action="store_true",
        default=True,
        help="Use original RGB colors from npz when available",
    )
    parser.add_argument(
        "--disable_original_color",
        dest="use_original_color",
        action="store_false",
        help="Disable original color and use fallback pure colors",
    )
    args = parser.parse_args()

    npz_path = resolve_npz_path(args.npz_path, args.npz_root)
    source, target, pred, source_color, target_color, used_pred_key = load_source_target_pred(npz_path, args.pred_key)
    pred_color = None
    if source_color is not None and source_color.shape[0] == pred.shape[0]:
        pred_color = source_color.copy()
    if not args.use_original_color:
        source_color = None
        target_color = None
        pred_color = None

    print(f"[INFO] npz: {npz_path}")
    print(f"[INFO] source points: {source.shape[0]}")
    print(f"[INFO] target points: {target.shape[0]}")
    print(f"[INFO] pred points:   {pred.shape[0]} (key={used_pred_key})")
    print(
        f"[INFO] source_color={'yes' if source_color is not None else 'no'}, "
        f"target_color={'yes' if target_color is not None else 'no'}, "
        f"pred_color={'yes' if pred_color is not None else 'no'}"
    )

    if args.save_ply_dir:
        save_ply(args.save_ply_dir, source, target, pred, source_color, target_color, pred_color)
        print(f"[INFO] ply saved to: {args.save_ply_dir}")

    if args.mode == "overlay":
        visualize_overlay(source, target, pred, source_color, target_color, pred_color)
    else:
        visualize_separate(source, target, pred, source_color, target_color, pred_color)


if __name__ == "__main__":
    main()
