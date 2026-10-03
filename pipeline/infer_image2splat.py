#!/usr/bin/env python3
"""
Image2Splat Single-Image 3D Gaussian Splatting Inference Runner
Rapid single-image 3D object generation for Splatris.
Bypasses COLMAP SfM entirely and lifts single-view RGB + Monocular Depth into
3D Gaussian Splats (.splat and .ply).
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

def log_stage(stage, pct, msg):
    print(f"STAGE:{stage}:{pct}:{msg}", flush=True)

def run_image2splat(image_path: Path, output_dir: Path):
    t0 = time.time()
    log_stage("IMAGE2SPLAT_INIT", 10, f"Loading input image: {image_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_splat = output_dir / "scene.splat"
    out_ply = output_dir / "scene.ply"

    im_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if im_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    has_alpha = (len(im_bgr.shape) == 3 and im_bgr.shape[2] == 4)
    if has_alpha:
        alpha = im_bgr[:, :, 3]
        im_bgr = im_bgr[:, :, :3]
    else:
        alpha = None

    H, W = im_bgr.shape[:2]
    log_stage("IMAGE2SPLAT_DEPTH", 30, f"Computing monocular depth map on NVIDIA GPU ({W}x{H})...")

    # Load MiDaS or Depth model on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(im_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = prediction.cpu().numpy()

    # Normalize depth map: foreground has higher disparity in MiDaS
    # Invert so metric Z is positive distance from camera (0.8m to 3.5m)
    d_min = np.percentile(depth_map, 2)
    d_max = np.percentile(depth_map, 98)
    d_norm = np.clip((depth_map - d_min) / max(d_max - d_min, 1e-5), 0.0, 1.0)

    # Object segmentation / foreground isolation
    log_stage("IMAGE2SPLAT_SEGMENT", 55, "Segmenting primary object foreground and unprojecting 3D rays...")
    if alpha is not None:
        fg_mask = alpha > 40
    else:
        # Depth-based saliency mask: keep primary central object
        y_coords, x_coords = np.indices((H, W))
        center_y, center_x = H / 2.0, W / 2.0
        dist_from_center = np.sqrt(((x_coords - center_x) / (W / 2.0))**2 + ((y_coords - center_y) / (H / 2.0))**2)
        center_weight = np.clip(1.0 - 0.5 * dist_from_center, 0.0, 1.0)
        saliency = d_norm * center_weight
        thresh = np.percentile(saliency[saliency > 0.1], 25) if np.any(saliency > 0.1) else 0.3
        fg_mask = (saliency >= thresh) & (dist_from_center < 1.2)

    # If mask is too small or empty, fall back to center crop
    if np.sum(fg_mask) < 2000:
        fg_mask = dist_from_center < 0.9

    # Unproject to 3D Camera coordinates
    # Focal length: standard 50mm FOV
    fx = fy = float(max(W, H) * 0.9)
    cx = float(W / 2.0)
    cy = float(H / 2.0)

    # Metric distance: scale between 1.0m and 2.5m
    Z = 1.0 + (1.0 - d_norm) * 1.5

    # Pixel grid step to optimize Gaussian density (stride 1 or 2 depending on resolution)
    stride = 1 if (W * H <= 512 * 512) else 2
    y_idx, x_idx = np.where(fg_mask[::stride, ::stride])
    y_idx = y_idx * stride
    x_idx = x_idx * stride

    u = x_idx.astype(np.float32)
    v = y_idx.astype(np.float32)
    z_vals = Z[y_idx, x_idx]

    # Standard computer vision pinhole unprojection
    # Flip Y so +Y is up, -Z is forward
    X = (u - cx) * z_vals / fx
    Y = -(v - cy) * z_vals / fy
    Z_3d = -z_vals

    # Center object at origin (0, 1.2, -1.8) for comfortable WebXR inspection
    X -= np.median(X)
    Y = Y - np.median(Y) + 1.2
    Z_3d = Z_3d - np.median(Z_3d) - 1.8

    xyz = np.stack([X, Y, Z_3d], axis=-1)
    colors = im_rgb[y_idx, x_idx]
    num_splats = len(xyz)

    log_stage("IMAGE2SPLAT_GAUSSIANS", 75, f"Generating {num_splats:,} 3D Gaussian ellipsoids...")

    # Local pixel spacing determines Gaussian footprint scale
    base_scale = (z_vals / fx) * stride * 1.3
    # Anisotropic splats slightly elongated horizontally/tangentially
    scales = np.stack([base_scale, base_scale, base_scale * 0.5], axis=-1)

    # Rotation: identity quaternion [1, 0, 0, 0]
    quaternions = np.zeros((num_splats, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0  # qw = 1.0

    # Opacities
    opacities = np.full(num_splats, 255, dtype=np.uint8)

    log_stage("IMAGE2SPLAT_EXPORT", 90, "Writing WebXR binary .splat and .ply model files...")

    # 1. Write binary .splat (32 bytes per Gaussian)
    with open(out_splat, "wb") as f:
        for i in range(num_splats):
            pos_bytes = struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
            scale_bytes = struct.pack("<fff", float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2]))
            color_bytes = struct.pack("<BBBB", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]), int(opacities[i]))
            # [rx, ry, rz, rw] normalized uint8: (q + 1) * 127.5
            qw, qx, qy, qz = quaternions[i]
            rx = int(np.clip((qx + 1.0) * 127.5, 0, 255))
            ry = int(np.clip((qy + 1.0) * 127.5, 0, 255))
            rz = int(np.clip((qz + 1.0) * 127.5, 0, 255))
            rw = int(np.clip((qw + 1.0) * 127.5, 0, 255))
            rot_bytes = struct.pack("<BBBB", rx, ry, rz, rw)
            f.write(pos_bytes + scale_bytes + color_bytes + rot_bytes)

    # 2. Write standard 3DGS .ply file
    from plyfile import PlyData, PlyElement
    SH_C0 = 0.28209479177387814
    f_dc_0 = ((colors[:, 0].astype(np.float32) / 255.0) - 0.5) / SH_C0
    f_dc_1 = ((colors[:, 1].astype(np.float32) / 255.0) - 0.5) / SH_C0
    f_dc_2 = ((colors[:, 2].astype(np.float32) / 255.0) - 0.5) / SH_C0
    log_scales = np.log(np.clip(scales, 1e-6, None)).astype(np.float32)
    logit_opacity = np.full(num_splats, 4.0, dtype=np.float32)

    vertex_dtypes = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    vertex_data = np.empty(num_splats, dtype=vertex_dtypes)
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['f_dc_0'] = f_dc_0
    vertex_data['f_dc_1'] = f_dc_1
    vertex_data['f_dc_2'] = f_dc_2
    vertex_data['opacity'] = logit_opacity
    vertex_data['scale_0'] = log_scales[:, 0]
    vertex_data['scale_1'] = log_scales[:, 1]
    vertex_data['scale_2'] = log_scales[:, 2]
    vertex_data['rot_0'] = quaternions[:, 0]
    vertex_data['rot_1'] = quaternions[:, 1]
    vertex_data['rot_2'] = quaternions[:, 2]
    vertex_data['rot_3'] = quaternions[:, 3]

    ply_el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_el]).write(str(out_ply))

    elapsed = time.time() - t0
    log_stage("COMPLETE", 100, f"Image2Splat completed in {elapsed:.2f}s! Generated {num_splats:,} Gaussians.")
    print(f"[Image2Splat] Exported {out_splat} ({os.path.getsize(out_splat):,} bytes) and {out_ply}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image2Splat Inference Runner")
    parser.add_argument("--image", required=True, help="Path to input image (.jpg, .png)")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    run_image2splat(Path(args.image), Path(args.output_dir))
