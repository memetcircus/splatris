# ☁️ Nebius AI Cloud GPU Worker Setup Guide

This guide provides end-to-end instructions for provisioning and configuring an **NVIDIA® L40S GPU** instance on **Nebius AI Cloud** to run the **Splatris 3DGS & Triangle Splatting Worker Service** (`splatris-worker.service`).

---

## 1. Nebius AI Cloud VM Provisioning

In the [Nebius AI Cloud Console](https://console.nebius.com/), create a new Compute Instance with the following specifications:

| Parameter | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Name** | `bronze-bug-instance-6` | Or your choice of instance name |
| **Platform** | `gpu-l40s-a` | **NVIDIA® L40S Intel** (Ada Lovelace architecture) |
| **GPU Preset** | **1 GPU, 8 vCPUs, 32 GiB RAM** | Dedicated 48 GB GDDR6 GPU memory |
| **VM Capacity** | **Preemptible** | Cost-effective (~80% discount for training workloads) |
| **Operating System** | **Ubuntu 24.04 LTS for NVIDIA® GPUs** | Pre-configured with NVIDIA drivers and CUDA |
| **Boot Disk** | **80 GiB SSD** | Ample storage for COLMAP databases & model checkpoints |
| **Security Group** | Allow TCP Port `22` (SSH) & `8000` (FastAPI) | Restrict port 8000 to your backend IP or VPC |

---

## 2. System Prerequisites & Dependencies

Connect to your instance via SSH:
```bash
ssh ubuntu@<NEBIUS_VM_PUBLIC_IP>
```

Update package lists and install essential system libraries, compilers, COLMAP (Structure-from-Motion), and FFmpeg:
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    colmap \
    ffmpeg \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    python3-venv \
    unzip
```

Verify NVIDIA GPU driver and CUDA recognition:
```bash
nvidia-smi
```
Expected output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|   0  NVIDIA L40S                    Off |   00000000:00:05.0 Off |                    0 |
| N/A   32C    P0             34W /  350W |    1420MiB /  46068MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Python Virtual Environment & CUDA Toolchain

Set up an isolated Python environment for Splatris:
```bash
python3 -m venv /home/ubuntu/splatris_env
source /home/ubuntu/splatris_env/bin/activate
pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA 12.4 support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install Gaussian Rasterization & Geometry Kernels
For Ada Lovelace architecture (L40S / RTX 4090), set `TORCH_CUDA_ARCH_LIST="8.9"` during compilation:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"

# Clone and compile custom differentiable rasterizer
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff-gaussian-rasterization
pip install /tmp/diff-gaussian-rasterization

# Install fast spatial k-nearest-neighbors
git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn
pip install /tmp/simple-knn
```

### Install Pipeline & Server Packages
```bash
pip install \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    opencv-python-headless \
    pillow \
    numpy \
    scipy \
    timm \
    rembg \
    requests \
    tqdm \
    plyfile
```

---

## 4. Deploy Splatris Pipeline Scripts

Clone the repository to `/home/ubuntu/splatris`:
```bash
git clone https://github.com/memetcircus/splatris.git /home/ubuntu/splatris
```

Verify key pipeline worker files are present:
- `/home/ubuntu/splatris/pipeline/gpu_worker_server.py` — High-throughput FastAPI service
- `/home/ubuntu/splatris/pipeline/train_3dgs.py` — 3D Gaussian Splatting & Triangle trainer

Create temporary job staging directory:
```bash
mkdir -p /tmp/splatris_jobs
chmod 777 /tmp/splatris_jobs
```

---

## 5. Configure Systemd Production Service

Create a persistent service unit file at `/etc/systemd/system/splatris-worker.service`:
```bash
sudo bash -c 'cat > /etc/systemd/system/splatris-worker.service << "EOF"
[Unit]
Description=Nebius GPU 3DGS Training Worker Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/splatris/pipeline
Environment="PORT=8000"
Environment="TORCH_CUDA_ARCH_LIST=8.9"
Environment="PATH=/home/ubuntu/splatris_env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/splatris_env/bin/python3 /home/ubuntu/splatris/pipeline/gpu_worker_server.py
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF'
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable splatris-worker.service
sudo systemctl start splatris-worker.service
```

Verify service status:
```bash
sudo systemctl status splatris-worker.service
```

---

## 6. Worker Verification & Health Check

Test the local worker health endpoint:
```bash
curl -s http://localhost:8000/health | jq .
```

Expected JSON response:
```json
{
  "status": "online",
  "service": "Nebius GPU 3DGS Worker",
  "cuda_available": true,
  "gpu_info": "NVIDIA L40S (48.0 GB VRAM)",
  "active_jobs": 0
}
```

### Worker API Endpoints Reference
| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check reporting CUDA availability, GPU model, and active job count. |
| `/train` | `POST` | Dispatches COLMAP + 3DGS/Triangle training from streamed frame ZIP archive. |
| `/status/{jobId}` | `GET` | Real-time progress percentage, stage, and completion status. |
| `/logs/{jobId}` | `GET` | Live training logs and iteration counters. |
| `/download/{jobId}/scene.splat` | `GET` | Binary 32-byte `.splat` scene stream. |
| `/abort/{jobId}` | `POST` | Immediate cancellation and process termination kill switch. |
| `/cleanup/{jobId}` | `POST` | Purges intermediate frames and temporary files to free disk. |

---

## 7. Connecting Splatris Backend to the Nebius GPU Worker

In your local or hosted Splatris backend instance, configure the `.env` file:
```env
# Remote Nebius GPU Worker Endpoint
NEBIUS_GPU_ENDPOINT=http://<NEBIUS_VM_PUBLIC_IP>:8000
```

When users upload videos or photos, Splatris will automatically stream frame archives to this Nebius L40S instance, train the 3D Gaussian Splat scene, and download the finished `.splat` model for instant WebXR viewing.
