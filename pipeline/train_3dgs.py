#!/usr/bin/env python3
"""
Nebius GPU 3D Gaussian Splatting Training Pipeline
Executes Nerfstudio Splatfacto (real 3DGS) on NVIDIA CUDA hardware,
exporting standard binary .splat models.
"""

import argparse
import os
import shutil
import subprocess
import struct
import time
import sys
from pathlib import Path

try:
    import numpy as np
    from plyfile import PlyData
except ImportError:
    sys.stderr.write("[Error] Missing dependencies. Please install numpy and plyfile.\n")
    sys.exit(1)


def log_stage(stage_name, progress_pct, details=""):
    """Emits structured progress logs parseable by the Node.js runner."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [STAGE:{stage_name}] [PROGRESS:{progress_pct}%] {details}", flush=True)


# Fake CUDA_HOME: bin/nvcc reports 12.4 (to satisfy PyTorch check) but delegates to real CUDA 13.0
# This lets gsplat JIT-compile its CUDA kernels at runtime without version mismatch errors.
FAKE_CUDA_HOME = "/home/ubuntu/fakecuda"
REAL_CUDA_BIN = "/usr/local/cuda-13.0/bin"
VENV_BIN = str(Path(sys.executable).parent)

def run_command(cmd, stage_name, progress_pct, error_msg, env=None):
    """Executes a subprocess and streams output."""
    log_stage(stage_name, progress_pct, f"Executing: {' '.join(cmd)}")
    
    # Inject CUDA and Python virtualenv environment
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    run_env["CUDA_HOME"] = FAKE_CUDA_HOME
    run_env["CUDA_PATH"] = FAKE_CUDA_HOME
    current_path = run_env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    # Put fake nvcc first, then real cuda 13.0 bin, then venv bin for ns-process-data / nerfstudio
    run_env["PATH"] = f"{FAKE_CUDA_HOME}/bin:{REAL_CUDA_BIN}:{VENV_BIN}:{current_path}"
    run_env["TORCH_CUDA_ARCH_LIST"] = "8.9"  # L40S compute capability
    if env:
        run_env.update(env)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=run_env
    )

    output_lines = []
    for line in process.stdout:
        line_str = line.strip()
        if line_str:
            output_lines.append(line_str)
            if len(output_lines) > 50:
                output_lines.pop(0)
            print(f"  [{stage_name}] {line_str}", flush=True)

    process.wait()
    if process.returncode != 0:
        tail_output = "\n".join(output_lines[-12:]) if output_lines else "No output captured"
        raise RuntimeError(f"{error_msg} (Exit code: {process.returncode})\nDetails:\n{tail_output}")


def convert_ply_to_splat(ply_path, splat_path):
    """
    Converts a standard 3DGS .ply to the binary .splat format expected by WebXR,
    filtering out needle rays, low-opacity floaters, and extreme scale outliers.
    """
    log_stage("SPLAT_CONVERSION", 90, f"Converting {ply_path} to WebXR .splat format with needle ray pruning")
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0].data

    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # 3DGS uses SH coefficients for base color
    SH_C0 = 0.28209479177387814
    r = (v['f_dc_0'] * SH_C0 + 0.5) * 255
    g = (v['f_dc_1'] * SH_C0 + 0.5) * 255
    b = (v['f_dc_2'] * SH_C0 + 0.5) * 255
    colors = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)

    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1))
    
    # Sigmoid opacity
    opacities = (1 / (1 + np.exp(-v['opacity']))) * 255
    opacities = opacities.clip(0, 255).astype(np.uint8)

    quaternions = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)

    # --- Strict Needle & Ray Pruning Filter ---
    # 1. Cull near-invisible floaters (opacity threshold: >= 13/255, i.e. > 5%)
    # 1. Cull transparent background fog (opacity < 26)
    mask = opacities >= 26

    # 2. Cull extreme anisotropic needle splats (where max_scale / min_scale > 18.0)
    max_scale = np.max(scales, axis=-1)
    min_scale = np.min(scales, axis=-1)
    scale_ratio = max_scale / np.clip(min_scale, 1e-6, None)
    mask &= (scale_ratio <= 18.0)

    # 3. Cull gigantic billboard Gaussians (scale > 0.08 world units)
    mask &= (max_scale <= 0.08)

    # 4. Outlier radius culling relative to scene median position (capped at 4.0m)
    if len(xyz) > 0:
        med_pos = np.median(xyz, axis=0)
        dists = np.linalg.norm(xyz - med_pos, axis=-1)
        radius_cutoff = min(float(np.percentile(dists, 96.0)), 4.0)
        mask &= (dists <= radius_cutoff)

    # 5. Crop splats outside primary scene bounding box ([-2.5, -1.0, -2.5] to [2.5, 2.5, 2.5])
    # to eliminate exterior colorful floaters and unmatched background ray artifacts
    bbox_mask = (
        (xyz[:, 0] >= -2.5) & (xyz[:, 0] <= 2.5) &
        (xyz[:, 1] >= -1.0) & (xyz[:, 1] <= 2.5) &
        (xyz[:, 2] >= -2.5) & (xyz[:, 2] <= 2.5)
    )
    mask &= bbox_mask

    # Apply mask
    pruned_count = int(np.sum(~mask))
    xyz = xyz[mask]
    scales = scales[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    quaternions = quaternions[mask]
    num_splats = len(xyz)
    print(f"[Pruning] Pruned {pruned_count} needle/ray/outlier Gaussians; retaining {num_splats} clean room splats.")

    os.makedirs(os.path.dirname(splat_path), exist_ok=True)
    
    with open(splat_path, "wb") as f:
        for i in range(num_splats):
            # 1. Position (12 bytes)
            pos_bytes = struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
            # 2. Scale (12 bytes)
            scale_bytes = struct.pack("<fff", float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2]))
            # 3. Color (4 bytes)
            color_bytes = struct.pack("<BBBB", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]), int(opacities[i]))
            # 4. Rotation (4 bytes, normalized 0-255 uint8) [rw, rx, ry, rz -> rx, ry, rz, rw for webxr]
            qw, qx, qy, qz = quaternions[i]
            rx = int(np.clip((qx + 1.0) * 127.5, 0, 255))
            ry = int(np.clip((qy + 1.0) * 127.5, 0, 255))
            rz = int(np.clip((qz + 1.0) * 127.5, 0, 255))
            rw = int(np.clip((qw + 1.0) * 127.5, 0, 255))
            rot_bytes = struct.pack("<BBBB", rx, ry, rz, rw)

            f.write(pos_bytes)
            f.write(scale_bytes)
            f.write(color_bytes)
            f.write(rot_bytes)
            
    log_stage("SPLAT_EXPORT_COMPLETE", 100, f"Exported {num_splats} clean splats to {splat_path}")


def convert_triangle_checkpoint(checkpoint_path, out_ply_path, out_splat_path=None):
    """
    Converts a Triangle Splatting checkpoint (point_cloud_state_dict.pt)
    to a standard colored mesh PLY file (with faces) and a WebXR binary .splat file.
    """
    import torch
    
    log_stage("TRIANGLE_CONVERSION", 90, f"Converting checkpoint {checkpoint_path} to Mesh PLY and WebXR .splat")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pts = sd["triangles_points"]
    if hasattr(pts, "detach"):
        pts = pts.detach()
    if hasattr(pts, "cpu"):
        pts = pts.cpu()
    pts = pts.float().numpy()
    
    num_triangles = pts.shape[0]
    
    # 1. Colors from features_dc
    features_dc = sd.get("features_dc", None)
    if features_dc is not None:
        if hasattr(features_dc, "detach"):
            features_dc = features_dc.detach()
        if hasattr(features_dc, "cpu"):
            features_dc = features_dc.cpu()
        fdc = features_dc.float().squeeze().numpy()
        if fdc.ndim == 1:
            fdc = fdc.reshape(1, -1)
        SH_C0 = 0.28209479177387814
        rgb = np.clip((fdc * SH_C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        if rgb.shape[0] != num_triangles:
            rgb = np.resize(rgb, (num_triangles, 3))
    else:
        rgb = np.full((num_triangles, 3), 200, dtype=np.uint8)
        
    # Opacities
    raw_opacity = sd.get("opacity", None)
    if raw_opacity is not None:
        if hasattr(raw_opacity, "detach"):
            raw_opacity = raw_opacity.detach()
        if hasattr(raw_opacity, "cpu"):
            raw_opacity = raw_opacity.cpu()
        op_vals = raw_opacity.float().squeeze().numpy()
        opacities = np.clip(1.0 / (1.0 + np.exp(-op_vals)) * 255.0, 0, 255).astype(np.uint8)
        if opacities.shape[0] != num_triangles:
            opacities = np.resize(opacities, (num_triangles,))
    else:
        opacities = np.full((num_triangles,), 255, dtype=np.uint8)
        
    # 2. Write standard colored mesh PLY with vertex coordinates, colors, and face indices
    all_verts = pts.reshape(-1, 3)
    num_verts = all_verts.shape[0]
    vert_colors = np.repeat(rgb, 3, axis=0)
    vert_alpha = np.repeat(opacities, 3, axis=0)
    
    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    with open(out_ply_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_verts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar alpha\n"
            f"element face {num_triangles}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        
        vertex_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')
        ])
        v_data = np.empty(num_verts, dtype=vertex_dtype)
        v_data['x'] = all_verts[:, 0]
        v_data['y'] = all_verts[:, 1]
        v_data['z'] = all_verts[:, 2]
        v_data['red'] = vert_colors[:, 0]
        v_data['green'] = vert_colors[:, 1]
        v_data['blue'] = vert_colors[:, 2]
        v_data['alpha'] = vert_alpha
        f.write(v_data.tobytes())
        
        face_dtype = np.dtype([
            ('count', 'u1'),
            ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4')
        ])
        face_data = np.empty(num_triangles, dtype=face_dtype)
        face_data['count'] = 3
        idxs = np.arange(num_triangles, dtype=np.int32) * 3
        face_data['i0'] = idxs
        face_data['i1'] = idxs + 1
        face_data['i2'] = idxs + 2
        f.write(face_data.tobytes())
        
    print(f"[Export] Saved Mesh PLY with {num_verts} vertices, {num_triangles} faces to {out_ply_path}")
    
    # 3. Write WebXR binary .splat file (32 bytes per triangle facet)
    if out_splat_path:
        os.makedirs(os.path.dirname(out_splat_path), exist_ok=True)
        v0 = pts[:, 0, :]
        v1 = pts[:, 1, :]
        v2 = pts[:, 2, :]
        centroids = (v0 + v1 + v2) / 3.0
        
        e1 = v1 - v0
        e2 = v2 - v0
        N = np.cross(e1, e2)
        n_len = np.linalg.norm(N, axis=1, keepdims=True)
        L1 = np.linalg.norm(e1, axis=1, keepdims=True)
        H = n_len / np.clip(L1, 1e-8, None)
        dists = np.linalg.norm(centroids, axis=1)

        # CULLING CRITERIA:
        # 1. Cull huge multi-meter triangles (L1 > 0.25m or H > 0.25m) that create massive milky white fog blobs
        # 2. Cull degenerate zero-area triangles (L1 < 2mm or H < 2mm)
        # 3. Cull distant background/camera floaters (dist > 5.0m)
        # 4. Cull transparent/invisible volume noise (opacities < 30)
        valid_mask = (L1[:, 0] <= 0.25) & (H[:, 0] <= 0.25) & (L1[:, 0] >= 0.002) & (H[:, 0] >= 0.002) & (dists <= 5.0) & (opacities >= 30)

        c_filt = centroids[mask if (mask := valid_mask).any() else slice(None)]
        e1_filt = e1[mask]
        N_filt = N[mask]
        n_len_filt = n_len[mask]
        L1_filt = L1[mask]
        H_filt = H[mask]
        rgb_filt = rgb[mask]
        op_filt = opacities[mask]
        num_valid = len(c_filt)

        # Orthonormal Basis: u1 (along edge 1), n_unit (surface normal), u2 (bitangent in triangle plane)
        n_unit = N_filt / np.clip(n_len_filt, 1e-8, None)
        u1 = e1_filt / np.clip(L1_filt, 1e-8, None)
        u2 = np.cross(n_unit, u1)

        # R is a 3x3 rotation matrix whose columns are [u1, u2, n_unit]
        R = np.stack([u1, u2, n_unit], axis=-1)

        # Vectorized Shepperd algorithm: 3x3 Rotation Matrix -> Quaternion [w, x, y, z]
        m00 = R[:, 0, 0]; m01 = R[:, 0, 1]; m02 = R[:, 0, 2]
        m10 = R[:, 1, 0]; m11 = R[:, 1, 1]; m12 = R[:, 1, 2]
        m20 = R[:, 2, 0]; m21 = R[:, 2, 1]; m22 = R[:, 2, 2]

        tr = m00 + m11 + m22
        quats = np.zeros((num_valid, 4), dtype=np.float32)

        c1 = tr > 0
        s1_val = 2.0 * np.sqrt(np.maximum(1e-8, 1.0 + tr[c1]))
        quats[c1, 0] = 0.25 * s1_val
        quats[c1, 1] = (m21[c1] - m12[c1]) / s1_val
        quats[c1, 2] = (m02[c1] - m20[c1]) / s1_val
        quats[c1, 3] = (m10[c1] - m01[c1]) / s1_val

        c2 = (~c1) & (m00 > m11) & (m00 > m22)
        s2_val = 2.0 * np.sqrt(np.maximum(1e-8, 1.0 + m00[c2] - m11[c2] - m22[c2]))
        quats[c2, 0] = (m21[c2] - m12[c2]) / s2_val
        quats[c2, 1] = 0.25 * s2_val
        quats[c2, 2] = (m01[c2] + m10[c2]) / s2_val
        quats[c2, 3] = (m02[c2] + m20[c2]) / s2_val

        c3 = (~c1) & (~c2) & (m11 > m22)
        s3_val = 2.0 * np.sqrt(np.maximum(1e-8, 1.0 + m11[c3] - m00[c3] - m22[c3]))
        quats[c3, 0] = (m02[c3] - m20[c3]) / s3_val
        quats[c3, 1] = (m01[c3] + m10[c3]) / s3_val
        quats[c3, 2] = 0.25 * s3_val
        quats[c3, 3] = (m12[c3] + m21[c3]) / s3_val

        c4 = (~c1) & (~c2) & (~c3)
        s4_val = 2.0 * np.sqrt(np.maximum(1e-8, 1.0 + m22[c4] - m00[c4] - m11[c4]))
        quats[c4, 0] = (m10[c4] - m01[c4]) / s4_val
        quats[c4, 1] = (m02[c4] + m20[c4]) / s4_val
        quats[c4, 2] = (m12[c4] + m21[c4]) / s4_val
        quats[c4, 3] = 0.25 * s4_val

        q_norm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(q_norm, 1e-8, None)

        # Calibrated Gaussian standard deviations: 0.28x triangle dimensions keeps splats tightly inside boundaries
        scale_1 = (L1_filt[:, 0] * 0.28).astype(np.float32)
        scale_2 = (H_filt[:, 0] * 0.28).astype(np.float32)
        scale_3 = np.full(num_valid, 0.0015, dtype=np.float32)

        splat_dtype = np.dtype([
            ('pos', '<f4', 3),
            ('scale', '<f4', 3),
            ('color', 'u1', 4),
            ('rot', 'u1', 4)
        ])
        s_data = np.empty(num_valid, dtype=splat_dtype)
        s_data['pos'] = c_filt.astype(np.float32)
        s_data['scale'] = np.column_stack([scale_1, scale_2, scale_3])
        s_data['color'][:, 0] = rgb_filt[:, 0]
        s_data['color'][:, 1] = rgb_filt[:, 1]
        s_data['color'][:, 2] = rgb_filt[:, 2]
        s_data['color'][:, 3] = op_filt

        # WebXR uint8 rot format: [rx, ry, rz, rw]
        s_data['rot'][:, 0] = np.clip((quats[:, 1] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        s_data['rot'][:, 1] = np.clip((quats[:, 2] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        s_data['rot'][:, 2] = np.clip((quats[:, 3] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        s_data['rot'][:, 3] = np.clip((quats[:, 0] + 1.0) * 127.5, 0, 255).astype(np.uint8)

        with open(out_splat_path, "wb") as f:
            f.write(s_data.tobytes())
        print(f"[Export] Saved WebXR .splat with {len(s_data):,} calibrated facets ({len(s_data)*32/(1024*1024):.1f} MB) to {out_splat_path}")

    # 4. Spatially partition mesh into WebXR chunks (.chunks.bin) for Quest 3 frustum culling
    out_chunks_path = out_ply_path.replace(".ply", ".chunks.bin")
    try:
        try:
            from pipeline.mesh_chunker import chunk_mesh_ply
        except ImportError:
            from mesh_chunker import chunk_mesh_ply
        chunk_mesh_ply(out_ply_path, out_chunks_path, grid_dim=4)
    except Exception as e:
        print(f"[Export] Spatial chunking notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Nebius GPU Real 3DGS & Triangle Splatting Pipeline")
    parser.add_argument("--images_dir", required=True, help="Path to input extracted frames")
    parser.add_argument("--output_dir", required=True, help="Directory to save models")
    parser.add_argument("--training_method", default="3dgs", choices=["3dgs", "triangle"], help="Reconstruction method: '3dgs' (default) or 'triangle' (Triangle Splatting mesh)")
    parser.add_argument("--matcher", default="vocab_tree", choices=["vocab_tree", "exhaustive", "sequential"], help="COLMAP matcher (vocab_tree recommended for loop closure)")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations (capped at 10000)")
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0004, help="Gradient threshold for densifying Gaussians")
    parser.add_argument("--downscale_factor", type=int, default=2, help="Image downscale factor for dataparser (2 = standard 1080p)")
    parser.add_argument("--num_downscales", type=int, default=2, help="Number of times to downscale images (default: 2)")
    parser.add_argument("--frame_indices", default="", help="Comma-separated frame indices to retain")
    parser.add_argument("--colmap_bin", default="colmap", help="Path to COLMAP executable")
    parser.add_argument("--output_splat", default="scene.splat", help="Output .splat filename")
    parser.add_argument("--output_ply", default="scene.ply", help="Output .ply filename")
    parser.add_argument("--scene_id", default="", help="Unique scene identifier for COLMAP caching")
    parser.add_argument("--cached_scene_id", default="", help="Bypass COLMAP and load cached SfM poses from this scene identifier")
    parser.add_argument("--cache_dir", default="/tmp/splatris_colmap_cache", help="Base directory for COLMAP pose cache")

    args = parser.parse_args()

    # Step Limit Guard: strictly cap iterations to 10,000 for 5-6 min L40S runs
    args.iterations = min(int(args.iterations), 10000)

    work_dir = os.path.join(args.output_dir, "pipeline_work")
    colmap_dir = os.path.join(work_dir, "colmap")
    ns_outputs = os.path.join(work_dir, "outputs")
    os.makedirs(work_dir, exist_ok=True)

    log_stage("PREPARATION", 10, f"Initializing real 3DGS pipeline for {args.images_dir} (target={args.iterations} iters, densify_thresh={args.densify_grad_thresh}, downscale={args.downscale_factor})")

    # Matcher Selection & Loop Closure Safeguard:
    # For room tours, 360° loops, and walkthroughs, vocab_tree or exhaustive compares
    # the closing frames against the initial frames to achieve seamless loop closure without duplicate ghost room drift.
    matcher_method = args.matcher
    if matcher_method not in ["vocab_tree", "exhaustive", "sequential"]:
        matcher_method = "vocab_tree"

    # If exhaustive matching is requested on an extensive frame set, switch to vocab_tree to maintain loop closure
    if matcher_method == "exhaustive":
        try:
            frame_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(frame_files) > 200:
                print(f"[Notice] Frame count ({len(frame_files)}) is high for exhaustive matching; using vocab_tree to maintain loop closure.")
                matcher_method = "vocab_tree"
        except Exception:
            pass

    print(f"[COLMAP] Using feature matching method: {matcher_method} (ensures 360° loop closure)")

    # Check for existing cached COLMAP SfM reconstruction (Smart Bypass)
    cache_base = Path(args.cache_dir)
    cache_base.mkdir(parents=True, exist_ok=True)
    has_cached_poses = False

    lookup_scene_id = args.cached_scene_id or args.scene_id
    if lookup_scene_id:
        scene_cache_dir = cache_base / lookup_scene_id
        candidate_sparse = scene_cache_dir / "colmap" / "sparse" / "0"
        if not candidate_sparse.exists():
            candidate_sparse = scene_cache_dir / "sparse" / "0"
        if candidate_sparse.exists():
            log_stage("COLMAP_CACHE_BYPASS", 35, f"Bypassing COLMAP SfM: Using cached camera poses and features for scene {lookup_scene_id}")
            colmap_dir = str(scene_cache_dir)
            has_cached_poses = True
            print(f"[Smart Bypass] Injected cached COLMAP features directly from {scene_cache_dir}")

    if not has_cached_poses:
        # Dynamic Overlap Frame Sampling:
        # Cap processed frames to a maximum of 220 frames (calculated dynamically from video duration * 2.2).
        # This maintains the critical 70-80% visual overlap for 360-degree perimeter loops and loop closure
        # without compute bloat on the L40S GPU.
        MAX_FRAMES = 220
        image_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        all_frames = sorted([f for f in os.listdir(args.images_dir) if f.endswith(image_exts)])

        if len(all_frames) > MAX_FRAMES:
            total_frames = len(all_frames)
            # Uniformly space out exactly MAX_FRAMES indices spanning from 0 to total_frames - 1
            keep_indices = set(round(i * (total_frames - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES))
            pruned_count = 0
            for idx, frame_name in enumerate(all_frames):
                if idx not in keep_indices:
                    try:
                        os.remove(os.path.join(args.images_dir, frame_name))
                        pruned_count += 1
                    except Exception as e:
                        print(f"  [Decimation] Warning: could not remove frame {frame_name}: {e}")

            remaining_count = len([f for f in os.listdir(args.images_dir) if f.endswith(image_exts)])
            log_stage("FRAME_DECIMATION", 15, f"Decimated frames from {total_frames} down to {remaining_count} (cap: {MAX_FRAMES}) with uniform stride for wide baseline COLMAP")
        else:
            log_stage("FRAME_DECIMATION", 15, f"Frame count ({len(all_frames)}) is within dynamic budget limit of {MAX_FRAMES} frames")

        # 1. Process Data (COLMAP SfM) using Nerfstudio
        # Camera model is explicitly set to 'perspective' (COLMAP OPENCV pinhole with distortion, avoid fisheye)
        run_command(
            [
                "ns-process-data", "images",
                "--data", args.images_dir,
                "--output-dir", colmap_dir,
                "--matching-method", matcher_method,
                "--camera-type", "perspective",
                "--num-downscales", str(args.num_downscales),
                "--no-gpu"
            ],
            stage_name="COLMAP_RECONSTRUCTION",
            progress_pct=25,
            error_msg="COLMAP via ns-process-data failed"
        )

        # Cache COLMAP output (sparse/0/ and images/) linked to scene_id
        target_cache_id = args.scene_id or args.cached_scene_id
        if target_cache_id:
            try:
                target_cache_dir = cache_base / target_cache_id
                target_cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Cache sparse/0
                src_sparse = Path(colmap_dir) / "colmap" / "sparse" / "0"
                if not src_sparse.exists():
                    src_sparse = Path(colmap_dir) / "sparse" / "0"
                if src_sparse.exists():
                    dst_colmap_sparse = target_cache_dir / "colmap" / "sparse" / "0"
                    dst_colmap_sparse.mkdir(parents=True, exist_ok=True)
                    dst_sparse_direct = target_cache_dir / "sparse" / "0"
                    dst_sparse_direct.mkdir(parents=True, exist_ok=True)
                    for item in src_sparse.iterdir():
                        if item.is_file():
                            shutil.copy2(item, dst_colmap_sparse / item.name)
                            shutil.copy2(item, dst_sparse_direct / item.name)

                # 2. Cache images
                src_images = Path(colmap_dir) / "images"
                if not src_images.exists():
                    src_images = Path(args.images_dir)
                if src_images.exists():
                    dst_images = target_cache_dir / "images"
                    dst_images.mkdir(parents=True, exist_ok=True)
                    for item in src_images.iterdir():
                        if item.is_file() and not (dst_images / item.name).exists():
                            shutil.copy2(item, dst_images / item.name)

                # 3. Cache transforms.json
                for t_name in ["transforms.json", "transforms_train.json"]:
                    src_t = Path(colmap_dir) / t_name
                    if src_t.exists():
                        shutil.copy2(src_t, target_cache_dir / t_name)

                log_stage("COLMAP_CACHED", 35, f"Successfully cached COLMAP output (sparse poses & images) for scene {target_cache_id}")
            except Exception as cache_err:
                print(f"[Warning] Failed to cache COLMAP output: {cache_err}")

    # --- CONDITIONAL ROUTING: 3DGS vs. Triangle Splatting ---
    if args.training_method == "triangle":
        log_stage("TRIANGLE_SPLATTING_PREP", 35, "Preparing shared COLMAP features for Triangle Splatting")
        
        # Nerfstudio outputs colmap/sparse/0 and colmap/images (or colmap_dir/images)
        colmap_sparse_0 = os.path.join(colmap_dir, "colmap", "sparse", "0")
        if not os.path.exists(colmap_sparse_0):
            if os.path.exists(os.path.join(colmap_dir, "sparse", "0")):
                colmap_sparse_0 = os.path.join(colmap_dir, "sparse", "0")
            else:
                raise RuntimeError(f"COLMAP sparse/0 directory not found in {colmap_dir}")
                
        # Source root for triangle-splatting dataset reader (parent of sparse/0)
        source_root = os.path.dirname(os.path.dirname(colmap_sparse_0))
        
        # Ensure 'images' directory exists inside source_root
        target_images_link = os.path.join(source_root, "images")
        if not os.path.exists(target_images_link):
            candidate_images = os.path.join(colmap_dir, "images")
            if not os.path.exists(candidate_images):
                candidate_images = args.images_dir
            try:
                os.symlink(os.path.abspath(candidate_images), target_images_link)
                print(f"[Triangle Splatting] Symlinked images from {candidate_images} to {target_images_link}")
            except Exception as e:
                try:
                    shutil.copytree(candidate_images, target_images_link)
                except Exception as ce:
                    print(f"[Triangle Splatting] Images link note: {ce}")
                    
        # Triangle Splatting Training script
        triangle_repo_dir = "/home/ubuntu/triangle-splatting"
        triangle_train_script = os.path.join(triangle_repo_dir, "train.py")
        if not os.path.exists(triangle_train_script):
            local_cand = Path(__file__).resolve().parent.parent / "triangle-splatting" / "train.py"
            if local_cand.exists():
                triangle_train_script = str(local_cand)
                triangle_repo_dir = str(local_cand.parent)

        triangle_model_dir = os.path.join(work_dir, "triangle_model")
        os.makedirs(triangle_model_dir, exist_ok=True)
        
        # Triangle splatting target iterations (optimal convergence at 7,000 steps)
        triangle_iters = min(args.iterations, 7000)
        
        log_stage("TRIANGLE_SPLATTING_OPTIMIZATION", 45, f"Optimizing Triangle Splatting model on CUDA ({triangle_iters} steps)")
        
        triangle_env = {
            "PYTHONPATH": f"{triangle_repo_dir}:{os.environ.get('PYTHONPATH', '')}"
        }
        
        run_command(
            [
                sys.executable, triangle_train_script,
                "-s", source_root,
                "-m", triangle_model_dir,
                "--iterations", str(triangle_iters),
                "--save_iterations", str(triangle_iters),
                "--quiet"
            ],
            stage_name="TRIANGLE_SPLATTING_OPTIMIZATION",
            progress_pct=75,
            error_msg="Triangle Splatting training failed",
            env=triangle_env
        )
        
        # Locate the saved checkpoint
        log_stage("TRIANGLE_EXPORT_PREPARATION", 85, "Locating Triangle Splatting checkpoint")
        checkpoint_path = None
        pc_dir = os.path.join(triangle_model_dir, "point_cloud")
        if os.path.exists(pc_dir):
            for entry in sorted(os.listdir(pc_dir), reverse=True):
                cand = os.path.join(pc_dir, entry, "point_cloud_state_dict.pt")
                if os.path.exists(cand):
                    checkpoint_path = cand
                    break
                    
        if not checkpoint_path:
            raise RuntimeError(f"Could not find point_cloud_state_dict.pt in {triangle_model_dir}")
            
        # Export to both scene.ply (mesh) and scene.splat (WebXR facet splats)
        target_ply = os.path.join(args.output_dir, args.output_ply if args.output_ply.endswith(".ply") else "scene.ply")
        target_splat = os.path.join(args.output_dir, args.output_splat if args.output_splat.endswith(".splat") else "scene.splat")
        
        convert_triangle_checkpoint(checkpoint_path, target_ply, target_splat)
        
        log_stage("TRIANGLE_EXPORT_COMPLETE", 100, f"Triangle Splatting complete. Output saved to {target_ply} and {target_splat}")
        return

    # --- STANDARD 3DGS PIPELINE (DEFAULT FALLBACK, 100% INTACT) ---
    # Stop split at 8,000 steps so the remaining 2,000 iterations refine positions & colors without Gaussian explosion
    stop_split_step = min(8000, args.iterations)

    # 2. Train Splatfacto (3D Gaussian Splatting) with scale regularization & needle culling
    run_command(
        [
            sys.executable, "-m", "nerfstudio.scripts.train", "splatfacto",
            "--data", colmap_dir,
            "--output-dir", ns_outputs,
            "--machine.num-devices", "1",
            "--max-num-iterations", str(args.iterations),
            "--pipeline.model.camera-optimizer.mode", "SO3xR3",
            "--pipeline.model.random-init", "False",
            "--pipeline.model.densify-grad-thresh", str(args.densify_grad_thresh),
            "--pipeline.model.stop-split-at", str(stop_split_step),
            "--pipeline.model.cull-alpha-thresh", "0.05",
            "--pipeline.model.cull-scale-thresh", "0.5",
            "--pipeline.model.use-scale-regularization", "True",
            "--pipeline.model.max-gauss-ratio", "10.0",
            "--vis", "tensorboard",
            "nerfstudio-data",
            "--downscale-factor", str(args.downscale_factor)
        ],
        stage_name="3DGS_OPTIMIZATION",
        progress_pct=50,
        error_msg="Nerfstudio Splatfacto training failed"
    )

    # Find the generated config.yml
    log_stage("EXPORT_PREPARATION", 80, "Locating generated Nerfstudio config")
    config_path = None
    for root, dirs, files in os.walk(ns_outputs):
        if "config.yml" in files:
            config_path = os.path.join(root, "config.yml")
            break
            
    if not config_path:
        raise RuntimeError("Could not find config.yml in Nerfstudio outputs directory.")

    # 3. Export to PLY
    export_dir = os.path.join(work_dir, "export")
    out_ply_path = os.path.join(export_dir, "splat.ply")
    os.makedirs(export_dir, exist_ok=True)

    run_command(
        [
            sys.executable, "-m", "nerfstudio.scripts.exporter", "gaussian-splat",
            "--load-config", config_path,
            "--output-dir", export_dir
        ],
        stage_name="PLY_EXPORT",
        progress_pct=85,
        error_msg="Nerfstudio PLY export failed"
    )

    if not os.path.exists(out_ply_path):
        raise RuntimeError(f"Expected PLY file not found at {out_ply_path}")

    # 4. Convert PLY to SPLAT
    splat_path = os.path.join(args.output_dir, args.output_splat)
    convert_ply_to_splat(out_ply_path, splat_path)
    
    # Also save the raw PLY for reference if requested
    target_ply = os.path.join(args.output_dir, args.output_ply)
    shutil.copy2(out_ply_path, target_ply)


if __name__ == "__main__":
    main()
