#!/usr/bin/env python3
"""Convert Nerfstudio-exported 3DGS PLY to binary .splat format for WebXR."""
import struct
import os
import numpy as np
from plyfile import PlyData

PLY_PATH = "/tmp/splatris_jobs/b315ec55-36b6-4583-aa14-8dd726dbad8d/output_real3/pipeline_work/export/splat.ply"
SPLAT_PATH = "/tmp/splatris_jobs/b315ec55-36b6-4583-aa14-8dd726dbad8d/output_real3/scene.splat"

plydata = PlyData.read(PLY_PATH)
v = plydata.elements[0].data
print(f"Loaded {len(v)} Gaussians from PLY", flush=True)

xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

SH_C0 = 0.28209479177387814
r = np.clip((v["f_dc_0"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
g = np.clip((v["f_dc_1"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
b = np.clip((v["f_dc_2"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)

scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).astype(np.float32)
opacities = np.clip((1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))) * 255, 0, 255).astype(np.uint8)
quaternions = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

num = len(xyz)
print(f"Writing {num} splats to {SPLAT_PATH}...", flush=True)

with open(SPLAT_PATH, "wb") as f:
    for i in range(num):
        # Position (12 bytes)
        f.write(struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
        # Scale (12 bytes)
        f.write(struct.pack("<fff", float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2])))
        # Color RGBA (4 bytes)
        f.write(struct.pack("<BBBB", int(r[i]), int(g[i]), int(b[i]), int(opacities[i])))
        # Rotation quaternion [rx,ry,rz,rw] normalized 0-255 (4 bytes)
        qw, qx, qy, qz = quaternions[i]
        rx = int(np.clip((qx + 1.0) * 127.5, 0, 255))
        ry = int(np.clip((qy + 1.0) * 127.5, 0, 255))
        rz = int(np.clip((qz + 1.0) * 127.5, 0, 255))
        rw = int(np.clip((qw + 1.0) * 127.5, 0, 255))
        f.write(struct.pack("<BBBB", rx, ry, rz, rw))

size = os.path.getsize(SPLAT_PATH)
print(f"Done! {num} splats, {size / 1024 / 1024:.1f} MB -> {SPLAT_PATH}", flush=True)
