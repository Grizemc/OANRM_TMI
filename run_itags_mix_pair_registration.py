#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pair registration for ITAGS data with PAConv:
1) Depth + RGB -> point clouds
2) Downsample each cloud
3) Run pretrained PAConv model (mix yaml + best_model)
4) Unsupervised fine-tuning on this pair
5) Gaussian post-processing on predicted flow
"""

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from torch.optim.lr_scheduler import StepLR

from util.util import load_cfg_from_cfg_file

pointops = None
HAS_POINTOPS = False
dist_chamfer_3D = None
HAS_CHAMFER_OP = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_pointops() -> bool:
    global pointops, HAS_POINTOPS
    if HAS_POINTOPS and pointops is not None:
        return True
    try:
        from lib.pointops.functions import pointops as pointops_mod
        pointops = pointops_mod
        HAS_POINTOPS = True
        return True
    except Exception as e:
        print(f"[WARN] pointops unavailable, smooth regularization disabled. detail: {e}")
        pointops = None
        HAS_POINTOPS = False
        return False


def ensure_chamfer() -> bool:
    global dist_chamfer_3D, HAS_CHAMFER_OP
    if HAS_CHAMFER_OP and dist_chamfer_3D is not None:
        return True
    try:
        import lib.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer_mod
        dist_chamfer_3D = chamfer_mod
        HAS_CHAMFER_OP = True
        return True
    except Exception as e:
        print(f"[WARN] chamfer cuda op unavailable, fallback to cdist. detail: {e}")
        dist_chamfer_3D = None
        HAS_CHAMFER_OP = False
        return False


def parse_int_from_path(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


def resolve_frame_path(
    depth_files: List[str],
    requested_index: int,
    index_mode: str,
    strict_index: bool,
) -> Tuple[str, int]:
    if len(depth_files) == 0:
        raise RuntimeError("No depth .npy files found.")

    if index_mode == "ordinal":
        if 1 <= requested_index <= len(depth_files):
            chosen = depth_files[requested_index - 1]
            return chosen, parse_int_from_path(chosen)
        if strict_index:
            raise RuntimeError(
                f"Requested ordinal index {requested_index} is out of range. "
                f"Available count: {len(depth_files)}."
            )
        clamped = max(1, min(requested_index, len(depth_files)))
        chosen = depth_files[clamped - 1]
        print(
            f"[WARN] ordinal index {requested_index} out of range, "
            f"fallback to nearest valid ordinal {clamped}."
        )
        return chosen, parse_int_from_path(chosen)

    # frame_id mode
    by_id = {parse_int_from_path(p): p for p in depth_files}
    if requested_index in by_id:
        chosen = by_id[requested_index]
        return chosen, requested_index

    if strict_index:
        raise RuntimeError(
            f"Requested frame_id {requested_index} not found in depth files."
        )

    all_ids = np.array(sorted(by_id.keys()))
    nearest_id = int(all_ids[np.argmin(np.abs(all_ids - requested_index))])
    print(
        f"[WARN] frame_id {requested_index} not found, "
        f"fallback to nearest frame_id {nearest_id}."
    )
    return by_id[nearest_id], nearest_id


def parse_pair_spec(spec: str) -> List[int]:
    s = str(spec).strip()
    if len(s) == 0:
        raise ValueError("Empty pair spec.")
    for sep in [",", "-", "_", " "]:
        if sep in s:
            ids = [int(x) for x in s.split(sep) if len(x.strip()) > 0]
            if len(ids) != 2:
                raise ValueError(f"Pair spec must contain exactly 2 ids, got: {spec}")
            return ids
    # compact form: "12" -> [1,2], "23" -> [2,3]
    if s.isdigit() and len(s) == 2:
        return [int(s[0]), int(s[1])]
    raise ValueError(
        f"Unsupported pair spec: {spec}. Use '12' or '1,2' style."
    )


def resolve_image_path(image_dir: str, frame_id: int, fallback_order_idx: Optional[int] = None) -> str:
    candidates = [
        os.path.join(image_dir, f"{frame_id:07d}.jpg"),
        os.path.join(image_dir, f"{frame_id:07d}.png"),
        os.path.join(image_dir, f"{frame_id:07d}.jpeg"),
        os.path.join(image_dir, f"{frame_id:010d}.jpg"),
        os.path.join(image_dir, f"{frame_id:010d}.png"),
        os.path.join(image_dir, f"{frame_id:010d}.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    if fallback_order_idx is not None:
        imgs = []
        imgs.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
        imgs.extend(glob.glob(os.path.join(image_dir, "*.png")))
        imgs.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))
        imgs = sorted(imgs)
        if 0 <= fallback_order_idx < len(imgs):
            p = imgs[fallback_order_idx]
            print(
                f"[WARN] image frame_id={frame_id} not found by name, "
                f"fallback by ordinal index -> {os.path.basename(p)}"
            )
            return p
    raise RuntimeError(f"Cannot find matching RGB image for frame_id={frame_id}.")


def fallback_intrinsic(h: int, w: int) -> np.ndarray:
    fx = float(max(w, h))
    fy = float(max(w, h))
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    # COLMAP qvec format: [qw, qx, qy, qz], world->camera rotation.
    qw, qx, qy, qz = qvec.tolist()
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def parse_colmap_cameras_txt(cameras_txt: str) -> Dict[int, np.ndarray]:
    if not os.path.exists(cameras_txt):
        return {}
    cam_map: Dict[int, np.ndarray] = {}
    with open(cameras_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            cam_id = int(parts[0])
            model = parts[1].upper()
            if model != "PINHOLE":
                continue
            fx = float(parts[4])
            fy = float(parts[5])
            cx = float(parts[6])
            cy = float(parts[7])
            k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            cam_map[cam_id] = k
    return cam_map


def parse_colmap_images_txt(images_txt: str) -> Dict[str, Dict[str, np.ndarray]]:
    if not os.path.exists(images_txt):
        return {}
    info: Dict[str, Dict[str, np.ndarray]] = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [x.rstrip("\n") for x in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if len(line) == 0 or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32)
        camera_id = int(parts[8])
        name = parts[9]

        r_wc = qvec_to_rotmat(qvec)  # world->camera
        r_cw = r_wc.T  # camera->world
        t_cw = -r_cw @ tvec.reshape(3, 1)  # camera->world translation

        info[name] = {
            "image_id": np.array([image_id], dtype=np.int32),
            "camera_id": np.array([camera_id], dtype=np.int32),
            "r_cw": r_cw.astype(np.float32),
            "t_cw": t_cw.reshape(3).astype(np.float32),
        }

        # Skip POINTS2D line if present.
        if i < len(lines):
            i += 1
    return info


def parse_intrinsic_from_cam(cam_dir: str, h: int, w: int) -> np.ndarray:
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "*_cam.txt")))
    if len(cam_files) == 0:
        print("[WARN] No cam txt found, fallback to image-center intrinsic.")
        return fallback_intrinsic(h, w)

    with open(cam_files[0], "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    try:
        idx = lines.index("intrinsic")
        k = []
        for i in range(1, 4):
            k.append([float(v) for v in lines[idx + i].split()])
        k = np.array(k, dtype=np.float32)
        return k
    except Exception:
        print("[WARN] Failed to parse intrinsic from cam txt, fallback to image-center intrinsic.")
        return fallback_intrinsic(h, w)


def resolve_cam_file(cam_dir: str, frame_id: int) -> Optional[str]:
    if not os.path.isdir(cam_dir):
        return None
    candidates = [
        os.path.join(cam_dir, f"{frame_id:07d}_cam.txt"),
        os.path.join(cam_dir, f"{frame_id:010d}_cam.txt"),
        os.path.join(cam_dir, f"{frame_id}_cam.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    all_cam = sorted(glob.glob(os.path.join(cam_dir, "*_cam.txt")))
    if len(all_cam) == 0:
        return None

    parsed = []
    for p in all_cam:
        name = os.path.basename(p).split("_cam.txt")[0]
        try:
            parsed.append((int(name), p))
        except Exception:
            continue
    if len(parsed) == 0:
        return None
    parsed.sort(key=lambda x: x[0])
    ids = np.array([x[0] for x in parsed], dtype=np.int64)
    nearest_id = int(ids[np.argmin(np.abs(ids - frame_id))])
    for fid, path in parsed:
        if fid == nearest_id:
            print(f"[WARN] exact cam for frame {frame_id} not found, fallback nearest cam frame {nearest_id}")
            return path
    return None


def parse_cam_txt(cam_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(cam_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    idx_e = lines.index("extrinsic")
    extr = []
    for i in range(1, 5):
        extr.append([float(v) for v in lines[idx_e + i].split()])
    extr = np.array(extr, dtype=np.float32)

    idx_k = lines.index("intrinsic")
    k = []
    for i in range(1, 4):
        k.append([float(v) for v in lines[idx_k + i].split()])
    k = np.array(k, dtype=np.float32)

    r_wc = extr[:3, :3]
    t_wc = extr[:3, 3]
    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc.reshape(3, 1)
    return k, r_cw.astype(np.float32), t_cw.reshape(3).astype(np.float32)


def get_intrinsic_and_pose_for_image(
    image_path: str,
    cameras_map: Dict[int, np.ndarray],
    images_map: Dict[str, Dict[str, np.ndarray]],
    fallback_k: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    name = os.path.basename(image_path)
    if name in images_map:
        item = images_map[name]
        cam_id = int(item["camera_id"][0])
        k = cameras_map.get(cam_id, fallback_k)
        r_cw = item["r_cw"].copy()
        t_cw = item["t_cw"].copy()
        return k, r_cw, t_cw
    return fallback_k, None, None


def depth_rgb_to_point_cloud(
    depth_path: str,
    image_path: str,
    intrinsic: np.ndarray,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    r_cw: Optional[np.ndarray] = None,
    t_cw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.load(depth_path).astype(np.float32)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if depth.shape[:2] != image.shape[:2]:
        raise RuntimeError(
            f"Depth and RGB shape mismatch: depth={depth.shape}, rgb={image.shape}."
        )

    z = depth * depth_scale
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if valid.sum() < 10:
        raise RuntimeError(
            f"Too few valid depth pixels in {depth_path}. valid={int(valid.sum())}"
        )

    h, w = depth.shape
    v, u = np.indices((h, w), dtype=np.float32)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1)[valid].astype(np.float32)
    if r_cw is not None and t_cw is not None:
        xyz = (xyz @ r_cw.T) + t_cw.reshape(1, 3)
    rgb = image[valid].astype(np.float32)
    return xyz, rgb


def voxel_and_sample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] != colors.shape[0]:
        raise RuntimeError("points/colors size mismatch.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    p = np.asarray(pcd.points).astype(np.float32)
    c = np.asarray(pcd.colors).astype(np.float32)

    if p.shape[0] < 10:
        raise RuntimeError(
            f"Too few points after downsampling: {p.shape[0]}. "
            "Try smaller voxel_size."
        )

    # Keep all points when num_points <= 0.
    if num_points is None or num_points <= 0:
        return p, c

    if p.shape[0] >= num_points:
        idx = rng.choice(p.shape[0], size=num_points, replace=False)
    else:
        idx = rng.choice(p.shape[0], size=num_points, replace=True)
    return p[idx], c[idx]


def xyz_to_normalization(xyz_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = xyz_in[:, 0:1]
    y = xyz_in[:, 1:2]
    z = xyz_in[:, 2:3]
    len_x = float(x.max() - x.min())
    len_y = float(y.max() - y.min())
    len_z = float(z.max() - z.min())

    eps = 1e-8
    len_x = max(len_x, eps)
    len_y = max(len_y, eps)
    len_z = max(len_z, eps)

    new_x = (((x - x.min()) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y.min()) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z.min()) / len_z - 0.5
    result = np.concatenate((new_x, new_y, new_z), axis=1).astype(np.float32)
    relax = np.array([len_x, len_y, len_z, float(x.min()), float(y.min()), float(z.min())], dtype=np.float32)
    return result, relax


def pcd2_gt_normalization(
    xyz_2: np.ndarray,
    xyz_gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = xyz_2[:, 0:1]
    y = xyz_2[:, 1:2]
    z = xyz_2[:, 2:3]
    len_x = float(x.max() - x.min())
    len_y = float(y.max() - y.min())
    len_z = float(z.max() - z.min())

    eps = 1e-8
    len_x = max(len_x, eps)
    len_y = max(len_y, eps)
    len_z = max(len_z, eps)

    x_min = float(x.min())
    y_min = float(y.min())
    z_min = float(z.min())

    new_x = (((x - x_min) / len_x) - 0.5) * (len_x / len_z)
    new_y = (((y - y_min) / len_y) - 0.5) * (len_y / len_z)
    new_z = (z - z_min) / len_z - 0.5
    result1 = np.concatenate((new_x, new_y, new_z), axis=1).astype(np.float32)

    xg = xyz_gt[:, 0:1]
    yg = xyz_gt[:, 1:2]
    zg = xyz_gt[:, 2:3]
    new_xg = (((xg - x_min) / len_x) - 0.5) * (len_x / len_z)
    new_yg = (((yg - y_min) / len_y) - 0.5) * (len_y / len_z)
    new_zg = (zg - z_min) / len_z - 0.5
    result2 = np.concatenate((new_xg, new_yg, new_zg), axis=1).astype(np.float32)

    relax = np.array([len_x, len_y, len_z, x_min, y_min, z_min], dtype=np.float32)
    return result1, result2, relax


def xyz_restore(xyz_in: np.ndarray, relax: np.ndarray) -> np.ndarray:
    len_x, len_y, len_z, x_min, y_min, z_min = [float(x) for x in relax.tolist()]
    eps = 1e-8
    len_x = max(len_x, eps)
    len_y = max(len_y, eps)
    len_z = max(len_z, eps)
    x = xyz_in[:, 0:1]
    y = xyz_in[:, 1:2]
    z = xyz_in[:, 2:3]
    new_x = (x * (len_z / len_x) + 0.5) * len_x + x_min
    new_y = (y * (len_z / len_y) + 0.5) * len_y + y_min
    new_z = (z + 0.5) * len_z + z_min
    return np.concatenate((new_x, new_y, new_z), axis=1).astype(np.float32)


def compute_nn_gt(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = KDTree(target)
    _, idx = tree.query(source, k=1)
    return target[idx].astype(np.float32)


def compute_fpfh_matches(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    distance_thresholds: List[float],
    max_corr: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    source_cloud.colors = o3d.utility.Vector3dVector(source_colors)
    target_cloud.colors = o3d.utility.Vector3dVector(target_colors)

    radius_normal = 0.2
    radius_feature = 0.25
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    for dt in distance_thresholds:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud,
            target_cloud,
            source_fpfh,
            target_fpfh,
            True,
            dt,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dt),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
        )
        matches = np.asarray(result.correspondence_set)
        if matches.shape[0] > 0:
            if matches.shape[0] > max_corr:
                choose = np.random.choice(matches.shape[0], size=max_corr, replace=False)
                matches = matches[choose]
            return matches[:, 0].astype(np.int64), matches[:, 1].astype(np.int64)

    print("[WARN] FPFH did not find correspondences, fallback to nearest-neighbor pseudo pairs.")
    tree = KDTree(target_points)
    _, idx = tree.query(source_points, k=1)
    k = min(max_corr, source_points.shape[0], target_points.shape[0])
    source_idx = np.arange(source_points.shape[0], dtype=np.int64)[:k]
    target_idx = idx[:k].astype(np.int64)
    return source_idx, target_idx


def build_model(cfg, device: torch.device) -> nn.Module:
    try:
        from model.backbone_new import PTEnetBase, PTFlow, PTFlowmean
    except Exception as e:
        raise RuntimeError(
            "Failed to import PAConv backbone. "
            "Please ensure pointops/model dependencies are built in this environment."
        ) from e

    if cfg.model_type == "Base":
        return PTEnetBase(c=6, args=cfg).to(device)
    if cfg.model_type == "Base_flow":
        return PTFlow(c=6, args=cfg).to(device)
    if cfg.model_type == "Base_flow_mean":
        return PTFlowmean(c=6, args=cfg).to(device)
    raise RuntimeError(f"Unsupported model_type: {cfg.model_type}")


def load_model_weights(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"best_model not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
        return model
    except Exception:
        pass

    # Try remove/add "module." prefix.
    keys = list(state.keys())
    if len(keys) == 0:
        raise RuntimeError("Checkpoint state_dict is empty.")

    if keys[0].startswith("module."):
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped, strict=False)
    else:
        add_module = {f"module.{k}": v for k, v in state.items()}
        try:
            model.load_state_dict(add_module, strict=False)
        except Exception:
            model.load_state_dict(state, strict=False)
    return model


class PairPostLoss:
    def __init__(
        self,
        gt_factor: float,
        smooth_factor: float,
        pseudo_factor: float,
        max_cdist_points: int = 2048,
    ):
        self.gt_factor = gt_factor
        self.smooth_factor = smooth_factor
        self.pseudo_factor = pseudo_factor
        self.max_cdist_points = max_cdist_points
        self.chamfer_op = None
        if ensure_chamfer():
            try:
                self.chamfer_op = dist_chamfer_3D.chamfer_3DDist()
            except Exception as e:
                print(f"[WARN] Failed to init chamfer op, fallback to cdist. detail: {e}")
                self.chamfer_op = None

    def _chamfer(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.chamfer_op is not None:
            d1, d2, _, _ = self.chamfer_op(pred, target)
            d1 = torch.relu(d1 - 1e-5)
            d2 = torch.relu(d2 - 1e-5)
            return d1.sqrt().mean() + d2.sqrt().mean()

        # Fallback (subsample to avoid huge cdist memory)
        if pred.shape[1] > self.max_cdist_points:
            idx = torch.randperm(pred.shape[1], device=pred.device)[: self.max_cdist_points]
            pred = pred[:, idx, :]
        if target.shape[1] > self.max_cdist_points:
            idx = torch.randperm(target.shape[1], device=target.device)[: self.max_cdist_points]
            target = target[:, idx, :]
        dist = torch.cdist(pred, target, p=2.0)
        return dist.min(-1)[0].mean() + dist.min(-2)[0].mean()

    def _smooth(self, pred: torch.Tensor, xyz1: torch.Tensor) -> torch.Tensor:
        if not ensure_pointops():
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        flow = pred - xyz1
        idx = pointops.knnquery(6, xyz1, xyz1)[:, :, 1:].contiguous()
        neigh_flow = pointops.grouping(flow.transpose(1, 2).contiguous(), idx.int())
        neigh_flow = neigh_flow.permute(0, 2, 3, 1).contiguous()
        flow_center = flow.unsqueeze(2)
        smooth = torch.norm(neigh_flow - flow_center, dim=-1).mean()
        return smooth * self.smooth_factor

    def __call__(
        self,
        l_xyz1: List[torch.Tensor],
        l_pred_xyz: List[torch.Tensor],
        xyz2: torch.Tensor,
        mask_pseudo1: torch.Tensor,
        gt_pseudo: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xyz1 = l_xyz1[0]
        pred = l_pred_xyz[0]

        chamfer_eval = self._chamfer(pred, xyz2)
        chamfer_loss = chamfer_eval * self.gt_factor
        smooth_loss = self._smooth(pred, xyz1)

        if mask_pseudo1.numel() == 0:
            pseudo_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            pred_pseudo = pred[:, mask_pseudo1[0].long(), :]
            pseudo_loss = torch.norm(pred_pseudo - gt_pseudo, dim=2).mean()

        total = chamfer_loss + smooth_loss + self.pseudo_factor * pseudo_loss
        logs = {
            "chamfer_eval": chamfer_eval.detach(),
            "chamfer_loss": chamfer_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "pseudo_loss": pseudo_loss.detach(),
            "total": total.detach(),
        }
        return total, logs


def gaussian_postprocess(
    xyz1: np.ndarray,
    pred_xyz: np.ndarray,
    knn_num: int,
    sigma: float,
) -> np.ndarray:
    knn_num = max(2, min(knn_num, xyz1.shape[0]))
    flow = pred_xyz - xyz1
    tree = KDTree(xyz1)
    _, idx = tree.query(xyz1, k=knn_num)

    neigh_xyz = xyz1[idx]
    neigh_flow = flow[idx]
    rel = neigh_xyz - xyz1[:, None, :]

    sigma = max(float(sigma), 1e-6)
    w = np.exp(-(np.square(rel).sum(axis=-1)) / (2.0 * sigma * sigma))
    w_sum = np.sum(w, axis=-1, keepdims=True) + 1e-12
    w = w / w_sum
    flow_new = (neigh_flow * w[:, :, None]).sum(axis=1)
    return (xyz1 + flow_new).astype(np.float32)


def symmetric_nn_distance(a: np.ndarray, b: np.ndarray) -> float:
    tree_b = KDTree(b)
    d1, _ = tree_b.query(a, k=1)
    tree_a = KDTree(a)
    d2, _ = tree_a.query(b, k=1)
    return float(d1.mean() + d2.mean())


def run_pair_registration(args: argparse.Namespace) -> None:
    cfg = load_cfg_from_cfg_file(args.config)
    if args.num_points is None:
        args.num_points = -1
    if args.post_lr is None:
        args.post_lr = float(cfg.post_lr)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    depth_files = sorted(glob.glob(os.path.join(args.depth_dir, "*.npy")))
    if len(depth_files) == 0:
        raise RuntimeError(f"No depth npy found in {args.depth_dir}")

    source_pair_ids = parse_pair_spec(args.source_pair)
    target_pair_ids = parse_pair_spec(args.target_pair)
    if (not args.allow_same_frame) and (source_pair_ids == target_pair_ids):
        raise RuntimeError(
            f"source_pair and target_pair are identical: {source_pair_ids}. "
            "Please provide different pairs or pass --allow_same_frame."
        )

    pair_frame_ids = {"source": source_pair_ids, "target": target_pair_ids}
    pair_depth_paths = {"source": [], "target": []}
    pair_image_paths = {"source": [], "target": []}

    for role in ["source", "target"]:
        for fid in pair_frame_ids[role]:
            depth_path, resolved_fid = resolve_frame_path(
                depth_files, fid, "frame_id", args.strict_index
            )
            if resolved_fid != fid:
                print(f"[WARN] {role} pair frame {fid} resolved to nearest {resolved_fid}")
            order_idx = depth_files.index(depth_path)
            image_path = resolve_image_path(args.image_dir, resolved_fid, fallback_order_idx=order_idx)
            pair_depth_paths[role].append(depth_path)
            pair_image_paths[role].append(image_path)

    source_frame = int(source_pair_ids[0])
    target_frame = int(target_pair_ids[0])
    tmp_depth = np.load(pair_depth_paths["source"][0])
    h, w = tmp_depth.shape[:2]
    fallback_k = parse_intrinsic_from_cam(args.cam_dir, h, w)

    cameras_map = parse_colmap_cameras_txt(args.cameras_txt)
    images_map = parse_colmap_images_txt(args.images_txt)
    pair_cam_files = {"source": [], "target": []}
    source_xyz_list: List[np.ndarray] = []
    source_rgb_list: List[np.ndarray] = []
    target_xyz_list: List[np.ndarray] = []
    target_rgb_list: List[np.ndarray] = []

    for role in ["source", "target"]:
        for depth_path, image_path, fid in zip(
            pair_depth_paths[role], pair_image_paths[role], pair_frame_ids[role]
        ):
            k_i, r_i, t_i = fallback_k, None, None
            cam_file = resolve_cam_file(args.cam_dir, fid)
            if cam_file is not None:
                try:
                    k_i, r_i, t_i = parse_cam_txt(cam_file)
                    pair_cam_files[role].append(cam_file)
                except Exception:
                    pair_cam_files[role].append("")
            else:
                pair_cam_files[role].append("")
                k_i, r_i, t_i = get_intrinsic_and_pose_for_image(
                    image_path, cameras_map, images_map, fallback_k
                )

            if not args.use_cam_pose:
                r_i, t_i = None, None

            xyz_i, rgb_i = depth_rgb_to_point_cloud(
                depth_path,
                image_path,
                k_i,
                args.depth_scale,
                args.min_depth,
                args.max_depth,
                r_i,
                t_i,
            )
            if role == "source":
                source_xyz_list.append(xyz_i)
                source_rgb_list.append(rgb_i)
            else:
                target_xyz_list.append(xyz_i)
                target_rgb_list.append(rgb_i)

    source_xyz = np.concatenate(source_xyz_list, axis=0).astype(np.float32)
    source_rgb = np.concatenate(source_rgb_list, axis=0).astype(np.float32)
    target_xyz = np.concatenate(target_xyz_list, axis=0).astype(np.float32)
    target_rgb = np.concatenate(target_rgb_list, axis=0).astype(np.float32)

    source_ds, source_rgb_ds = voxel_and_sample(
        source_xyz, source_rgb, args.voxel_size, args.num_points, rng
    )
    target_ds_original, target_rgb_ds = voxel_and_sample(
        target_xyz, target_rgb, args.voxel_size, args.num_points, rng
    )
    # No extra rigid/non-rigid perturbation on target input.
    target_ds = target_ds_original.copy()

    # Source->target nearest pseudo "gt" for metrics only.
    mask_gt_pc_raw = compute_nn_gt(source_ds, target_ds)
    source_norm, _ = xyz_to_normalization(source_ds)
    target_norm, mask_gt_pc_norm, relax_ratio = pcd2_gt_normalization(target_ds, mask_gt_pc_raw)

    pseudo1, pseudo2 = compute_fpfh_matches(
        source_norm,
        source_rgb_ds,
        target_norm,
        target_rgb_ds,
        args.fpfh_thresholds,
        args.max_fpfh_corr,
    )
    gt_pseudo = target_norm[pseudo2].astype(np.float32)

    mask_gt1 = np.ones((source_norm.shape[0],), dtype=np.bool_)
    mask_gt2 = np.ones((target_norm.shape[0],), dtype=np.bool_)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"[INFO] device={device}")
    print(
        f"[INFO] source_pair={source_pair_ids}, target_pair={target_pair_ids}, "
        f"source_points={source_norm.shape[0]}, target_points={target_norm.shape[0]}, "
        f"sample_num_points={args.num_points}, pseudo_corr={len(pseudo1)}"
    )

    source_t = torch.from_numpy(source_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_t = torch.from_numpy(target_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    source_rgb_t = torch.from_numpy(source_rgb_ds).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_rgb_t = torch.from_numpy(target_rgb_ds).unsqueeze(0).to(device=device, dtype=torch.float32)
    pseudo1_t = torch.from_numpy(pseudo1).unsqueeze(0).to(device=device, dtype=torch.long)
    gt_pseudo_t = torch.from_numpy(gt_pseudo).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask_gt1_t = torch.from_numpy(mask_gt1).unsqueeze(0).to(device=device, dtype=torch.bool)
    mask_gt2_t = torch.from_numpy(mask_gt2).unsqueeze(0).to(device=device, dtype=torch.bool)
    mask_gt_pc_t = torch.from_numpy(mask_gt_pc_norm).unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model(cfg, device)
    model = load_model_weights(model, args.best_model, device)

    with torch.no_grad():
        model.eval()
        pre_out = model(source_t, target_t, source_rgb_t, target_rgb_t)
        pred_before = pre_out[1][0][0].detach().cpu().numpy().astype(np.float32)
        pred_mask_before = torch.sigmoid(pre_out[4][0])[0, :, 0].detach().cpu().numpy().astype(np.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.post_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
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
        out = model(source_t, target_t, source_rgb_t, target_rgb_t)
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

    pred_post = gaussian_postprocess(
        source_norm, best_pred, knn_num=args.gaussian_knn, sigma=args.gaussian_sigma
    )

    # Convert predicted points back to target coordinate system.
    pred_before_raw = xyz_restore(pred_before, relax_ratio)
    pred_after_raw = xyz_restore(best_pred, relax_ratio)
    pred_post_raw = xyz_restore(pred_post, relax_ratio)

    metric_before = symmetric_nn_distance(pred_before_raw, target_ds)
    metric_after = symmetric_nn_distance(pred_after_raw, target_ds)
    metric_post = symmetric_nn_distance(pred_post_raw, target_ds)

    pair_tag = (
        f"srcpair_{source_pair_ids[0]:07d}-{source_pair_ids[1]:07d}"
        f"_tgtpair_{target_pair_ids[0]:07d}-{target_pair_ids[1]:07d}"
    )
    out_dir = os.path.join(args.output_root, pair_tag)
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "pair_registration_result.npz"),
        source_pair_ids=np.array(source_pair_ids, dtype=np.int32),
        target_pair_ids=np.array(target_pair_ids, dtype=np.int32),
        source_depth_paths=np.array(pair_depth_paths["source"]),
        target_depth_paths=np.array(pair_depth_paths["target"]),
        source_image_paths=np.array(pair_image_paths["source"]),
        target_image_paths=np.array(pair_image_paths["target"]),
        source_cam_files=np.array(pair_cam_files["source"]),
        target_cam_files=np.array(pair_cam_files["target"]),
        source_points_raw=source_ds,
        target_points_raw=target_ds,
        target_points_raw_original=target_ds_original,
        source_colors=source_rgb_ds,
        target_colors=target_rgb_ds,
        source_points_norm=source_norm,
        target_points_norm=target_norm,
        relax_ratio=relax_ratio,
        mask_gt1=mask_gt1,
        mask_gt2=mask_gt2,
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
        "source_pair_ids": [int(x) for x in source_pair_ids],
        "target_pair_ids": [int(x) for x in target_pair_ids],
        "source_cam_files": [str(x) for x in pair_cam_files["source"]],
        "target_cam_files": [str(x) for x in pair_cam_files["target"]],
        "source_depth_paths": [str(x) for x in pair_depth_paths["source"]],
        "target_depth_paths": [str(x) for x in pair_depth_paths["target"]],
        "images_txt": args.images_txt,
        "cameras_txt": args.cameras_txt,
        "use_cam_pose": bool(args.use_cam_pose),
        "source_frame_id": int(source_frame),
        "target_frame_id": int(target_frame),
        "index_mode": args.index_mode,
        "num_points": int(args.num_points),
        "source_points_num": int(source_norm.shape[0]),
        "target_points_num": int(target_norm.shape[0]),
        "voxel_size": float(args.voxel_size),
        "post_lr": float(args.post_lr),
        "pt_epoch": int(args.pt_epoch),
        "best_epoch": int(best_epoch),
        "fpfh_corr_num": int(len(pseudo1)),
        "target_perturb_enabled": False,
        "metric_symmetric_nn_before": float(metric_before),
        "metric_symmetric_nn_after": float(metric_after),
        "metric_symmetric_nn_post": float(metric_post),
        "output_npz": os.path.join(out_dir, "pair_registration_result.npz"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Registration finished.")
    print(f"[DONE] Output directory: {out_dir}")
    print(
        f"[DONE] symmetric_nn: before={metric_before:.6f}, "
        f"after={metric_after:.6f}, post={metric_post:.6f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ITAGS pair registration with PAConv mix model + fine-tune + post-process"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/szm/Paconv_730/config/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix.yaml",
        help="Path to mix yaml config.",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        default="/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/saved_model/best_model.t7",
        help="Path to pretrained best_model.",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/depth",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/images",
    )
    parser.add_argument(
        "--cam_dir",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/cams",
        help="Fallback camera txt directory (legacy format).",
    )
    parser.add_argument(
        "--images_txt",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/sparse_inter_3/0/images.txt",
        help="COLMAP images.txt for per-frame extrinsic and camera id.",
    )
    parser.add_argument(
        "--cameras_txt",
        type=str,
        default="/home/szm/Paconv_730/Data_ITAGS/endo_2/sparse_inter_3/0/cameras.txt",
        help="COLMAP cameras.txt for per-camera intrinsics.",
    )
    parser.add_argument(
        "--use_cam_pose",
        action="store_true",
        default=True,
        help="Apply camera pose transform (cam txt preferred, COLMAP txt fallback).",
    )
    parser.add_argument(
        "--disable_cam_pose",
        dest="use_cam_pose",
        action="store_false",
        help="Disable applying camera pose transform.",
    )
    parser.add_argument(
        "--source_pair",
        type=str,
        default="12",
        help="Source pair spec, e.g. '12' means [1,2], or '1,2'.",
    )
    parser.add_argument(
        "--target_pair",
        type=str,
        default="23",
        help="Target pair spec, e.g. '23' means [2,3], or '2,3'.",
    )
    parser.add_argument("--source_index", type=int, default=12, help="(legacy, unused in pair mode)")
    parser.add_argument("--target_index", type=int, default=23, help="(legacy, unused in pair mode)")
    parser.add_argument(
        "--index_mode",
        type=str,
        choices=["frame_id", "ordinal"],
        default="frame_id",
        help="How to interpret source_index/target_index.",
    )
    parser.add_argument(
        "--strict_index",
        action="store_true",
        help="If set, do not fallback to nearest frame when index is missing/out-of-range.",
    )
    parser.add_argument(
        "--allow_same_frame",
        action="store_true",
        help="Allow source_pair and target_pair to be identical.",
    )
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--max_depth", type=float, default=1e6)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument(
        "--num_points",
        type=int,
        default=-1,
        help="Fixed sampling size after voxel downsample; <=0 means keep all points (default).",
    )
    parser.add_argument("--post_lr", type=float, default=None, help="Default from yaml post_lr if not set.")
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
        default="/home/szm/Paconv_730/checkpoints/Source_Flow_softmax_topkpoint_topmask_fuse_8192_mix/itags_mix_pair_registration",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pair_registration(args)


if __name__ == "__main__":
    main()
