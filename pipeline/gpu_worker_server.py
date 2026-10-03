#!/usr/bin/env python3
"""
Nebius GPU 3DGS Worker Server
A high-throughput FastAPI microservice running on Nebius NVIDIA GPU instances.
Receives room video frames, runs COLMAP + 3DGS training, and serves binary .splat models.
"""

import json
import os
import shutil
import subprocess
import signal
import sys
import threading
import uuid
import zipfile
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="Nebius GPU 3DGS Training Worker",
    description="Dedicated NVIDIA CUDA worker for COLMAP Structure-from-Motion and 3D Gaussian Splatting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/splatris_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job state tracker & active process registry
jobs_state = {}
active_processes = {}


@app.get("/health")
def health_check():
    gpu_info = "No GPU / CPU Fallback"
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)"
    except ImportError:
        pass

    return {
        "status": "online",
        "service": "Nebius GPU 3DGS Worker",
        "cuda_available": cuda_available,
        "gpu": gpu_info,
        "active_jobs": sum(1 for j in jobs_state.values() if j.get("status") == "processing")
    }


def run_training_task(job_id: str, images_dir: Path, output_dir: Path, matcher: str, iterations: int, frame_indices: str, densify_grad_thresh: float = 0.0004, downscale_factor: int = 2, training_method: str = "3dgs", scene_id: str = None, cached_scene_id: str = None):
    job = jobs_state.get(job_id)
    if not job:
        return

    # Target 10,000 steps (optimal sweet spot for ~6 min training on L40S)
    effective_iterations = min(int(iterations), 10000)

    job_dir = output_dir.parent
    script_path = Path(__file__).parent / "train_3dgs.py"
    output_splat = output_dir / "scene.splat"
    output_ply = output_dir / "scene.ply"
    log_file_path = job_dir / "train.log"

    try:
        if training_method == "image2splat":
            # Bypass COLMAP SfM entirely since this method operates on a single image
            all_imgs = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if not all_imgs:
                job["status"] = "failed"
                job["error"] = f"No input image found in {images_dir} for Image2Splat reconstruction"
                return

            input_image = all_imgs[0]
            script_path = Path("/home/ubuntu/Image2Splat/infer_image2splat.py")
            if not script_path.exists():
                script_path = Path(__file__).parent / "infer_image2splat.py"

            isolated_python = Path("/home/ubuntu/image2splat_env/bin/python3")
            python_bin = str(isolated_python if isolated_python.exists() else sys.executable)

            cmd = [
                python_bin, str(script_path),
                "--image", str(input_image),
                "--output_dir", str(output_dir)
            ]
            job["stage"] = "Running Image2Splat Instant Inference"
        else:
            cmd = [
                sys.executable, str(script_path),
                "--images_dir", str(images_dir),
                "--output_dir", str(output_dir),
                "--training_method", training_method,
                "--matcher", matcher,
                "--iterations", str(effective_iterations),
                "--densify_grad_thresh", str(densify_grad_thresh),
                "--downscale_factor", str(downscale_factor),
                "--num_downscales", "2",
                "--output_splat", "scene.splat",
                "--output_ply", "scene.ply"
            ]

            if scene_id:
                cmd.extend(["--scene_id", str(scene_id)])
            if cached_scene_id:
                cmd.extend(["--cached_scene_id", str(cached_scene_id)])
            if frame_indices:
                cmd.extend(["--frame_indices", frame_indices])

        # Inject complete environment for CUDA + fakecuda + venv
        worker_env = os.environ.copy()
        worker_env["PYTHONUNBUFFERED"] = "1"
        worker_env["CUDA_HOME"] = "/home/ubuntu/fakecuda"
        worker_env["CUDA_PATH"] = "/home/ubuntu/fakecuda"
        venv_bin = str(Path(sys.executable).parent)
        current_path = worker_env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        worker_env["PATH"] = f"/home/ubuntu/fakecuda/bin:/usr/local/cuda-13.0/bin:{venv_bin}:{current_path}"
        worker_env["TORCH_CUDA_ARCH_LIST"] = "8.9"

        output_tail = []

        job["status"] = "processing"
        if training_method != "image2splat":
            job["stage"] = "Initializing COLMAP SfM & GPU Pipeline"
        job["progress"] = 10

        with open(log_file_path, "w", buffering=1) as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=worker_env,
                preexec_fn=os.setsid
            )
            active_processes[job_id] = process

            for line in process.stdout:
                if job.get("status") == "aborted":
                    break
                line_str = line.strip()
                if line_str:
                    print(f"[{job_id[:8]}] {line_str}", flush=True)
                    log_f.write(line_str + "\n")
                    output_tail.append(line_str)
                    if len(output_tail) > 40:
                        output_tail.pop(0)

                if "[STAGE:" in line_str:
                    # Extract stage and progress
                    try:
                        stage_part = line_str.split("[STAGE:")[1].split("]")[0]
                        prog_part = line_str.split("[PROGRESS:")[1].split("%]")[0]
                        job["stage"] = stage_part.replace("_", " ").title()
                        job["progress"] = int(prog_part)
                    except Exception:
                        pass
                elif line_str.startswith("STAGE:"):
                    try:
                        parts = line_str.split(":", 3)
                        job["stage"] = parts[3] if len(parts) > 3 else parts[1].replace("_", " ").title()
                        job["progress"] = int(parts[2])
                    except Exception:
                        pass

            process.wait()

        if job.get("status") == "aborted":
            print(f"[{job_id[:8]}] Training aborted by user. Skipping finalization.", flush=True)
            return

        is_triangle = (training_method == "triangle")
        success = (process.returncode == 0) and (output_splat.exists() or output_ply.exists())
        if success:
            job["status"] = "completed"
            job["stage"] = "Completed"
            job["progress"] = 100
            job["training_method"] = training_method
            primary_file = output_ply if (is_triangle and output_ply.exists()) else (output_splat if output_splat.exists() else output_ply)
            job["output_filename"] = primary_file.name
            job["model_path"] = str(primary_file)
            job["splat_path"] = str(output_splat if output_splat.exists() else primary_file)
            job["splat_size_bytes"] = primary_file.stat().st_size
        else:
            if job.get("status") == "aborted":
                return
            tail_msg = "\n".join(output_tail[-15:]) if output_tail else "No output captured"
            job["status"] = "failed"
            job["stage"] = "Failed"
            job["error"] = f"Training failed with exit code {process.returncode}:\n{tail_msg}"
            print(f"[{job_id[:8]}] ERROR: {job['error']}", flush=True)
    except Exception as e:
        if job.get("status") == "aborted":
            return
        job["status"] = "failed"
        job["stage"] = "Failed"
        job["error"] = str(e)
        print(f"[{job_id[:8]}] EXCEPTION: {e}", flush=True)
    finally:
        active_processes.pop(job_id, None)


@app.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    frames_zip: UploadFile = File(None),
    jobId: str = Form(None),
    sceneId: str = Form(None),
    cachedSceneId: str = Form(None),
    colmapMatcher: str = Form("vocab_tree"),
    iterations: int = Form(10000),
    filteredFrameIndices: str = Form(""),
    densify_grad_thresh: float = Form(0.0004),
    downscale_factor: int = Form(2),
    training_method: str = Form("3dgs")
):
    job_id = jobId or str(uuid.uuid4())
    effective_scene_id = sceneId or cachedSceneId or job_id
    job_dir = JOBS_DIR / job_id
    images_dir = job_dir / "images"
    output_dir = job_dir / "output"

    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir_base = Path("/tmp/splatris_colmap_cache")
    is_bypass = False

    if cachedSceneId:
        cached_path = cache_dir_base / cachedSceneId
        sparse_check = cached_path / "colmap" / "sparse" / "0"
        if not sparse_check.exists():
            sparse_check = cached_path / "sparse" / "0"
        if sparse_check.exists():
            is_bypass = True
            print(f"[{job_id[:8]}] SMART BYPASS ACTIVE: Reusing cached COLMAP poses from {cached_path}")

    if not is_bypass:
        if not frames_zip:
            raise HTTPException(status_code=400, detail="Missing required frames_zip for fresh training job without cache.")
        # Save and extract zip
        zip_path = job_dir / "frames.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(frames_zip.file, f)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)

        # Enforce dynamic frame sampling cap on worker: allow up to 220 frames with uniform stride
        MAX_WORKER_FRAMES = 220
        extracted_imgs = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if len(extracted_imgs) > MAX_WORKER_FRAMES:
            total_imgs = len(extracted_imgs)
            keep_idxs = set(round(i * (total_imgs - 1) / (MAX_WORKER_FRAMES - 1)) for i in range(MAX_WORKER_FRAMES))
            for idx, img_path in enumerate(extracted_imgs):
                if idx not in keep_idxs:
                    try:
                        img_path.unlink()
                    except Exception:
                        pass
            print(f"[{job_id[:8]}] Decimated frames from {total_imgs} to {len(list(images_dir.glob('*')))} (cap: {MAX_WORKER_FRAMES})")

    jobs_state[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "training_method": training_method,
        "stage": f"Bypassing upload & COLMAP: loading cached poses for {cachedSceneId}" if is_bypass else f"Received frames; queued for GPU training ({training_method})",
        "progress": 35 if is_bypass else 5,
        "error": None,
        "splat_path": None
    }

    # Step limit guard: strictly cap iterations to 10000
    iterations = min(int(iterations), 10000)

    background_tasks.add_task(
        run_training_task,
        job_id,
        images_dir,
        output_dir,
        colmapMatcher,
        iterations,
        filteredFrameIndices,
        densify_grad_thresh,
        downscale_factor,
        training_method,
        scene_id=effective_scene_id,
        cached_scene_id=cachedSceneId if is_bypass else None
    )

    return {
        "success": True,
        "jobId": job_id,
        "status": "processing",
        "training_method": training_method,
        "is_bypass": is_bypass,
        "iterations": iterations,
        "message": f"Training job {job_id} scheduled on Nebius GPU ({training_method}, {iterations} steps, bypass={is_bypass})."
    }


@app.get("/cache/status/{scene_id}")
def get_cache_status(scene_id: str):
    cache_path = Path("/tmp/splatris_colmap_cache") / scene_id
    sparse_0 = cache_path / "colmap" / "sparse" / "0"
    if not sparse_0.exists():
        sparse_0 = cache_path / "sparse" / "0"
    is_cached = sparse_0.exists()
    return {
        "success": True,
        "sceneId": scene_id,
        "hasCache": is_cached,
        "cacheDir": str(cache_path) if is_cached else None
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs_state.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "success": True,
        "jobId": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "training_method": job.get("training_method", "3dgs"),
        "output_filename": job.get("output_filename"),
        "error": job.get("error"),
        "splat_size_bytes": job.get("splat_size_bytes")
    }


@app.get("/logs/{job_id}")
def get_logs(job_id: str):
    log_file = JOBS_DIR / job_id / "train.log"
    if not log_file.exists():
        return {"jobId": job_id, "logs": "No log file found"}
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        return {"jobId": job_id, "logs": content[-8000:]}
    except Exception as e:
        return {"jobId": job_id, "logs": f"Error reading logs: {e}"}


@app.get("/download/{job_id}/scene.splat")
def download_splat(job_id: str):
    job = jobs_state.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed" or not job.get("splat_path"):
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    splat_file = Path(job["splat_path"])
    if not splat_file.exists():
        raise HTTPException(status_code=404, detail="Output .splat file not found on disk")

    return FileResponse(
        str(splat_file),
        media_type="application/octet-stream",
        filename=f"{job_id}.splat"
    )


@app.get("/download/{job_id}/{filename}")
def download_model_file(job_id: str, filename: str):
    job = jobs_state.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    safe_name = Path(filename).name
    file_path = JOBS_DIR / job_id / "output" / safe_name
    if not file_path.exists():
        if job.get("model_path") and Path(job["model_path"]).name == safe_name:
            file_path = Path(job["model_path"])
        elif job.get("splat_path") and Path(job["splat_path"]).exists():
            file_path = Path(job["splat_path"])
        else:
            raise HTTPException(status_code=404, detail=f"Output file {safe_name} not found on disk")

    media_type = "application/x-ply" if safe_name.endswith(".ply") else "application/octet-stream"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=f"{job_id}_{safe_name}"
    )


@app.post("/abort/{job_id}")
def abort_job(job_id: str):
    """
    Terminates the active training process (COLMAP / 3DGS) immediately,
    cleans up GPU resources and disk files, and marks job as aborted.
    """
    proc = active_processes.get(job_id)
    killed = False
    if proc:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            killed = True
            print(f"[{job_id[:8]}] Terminated active process group {pgid} via SIGKILL", flush=True)
        except Exception as e:
            try:
                proc.kill()
                killed = True
                print(f"[{job_id[:8]}] Fallback proc.kill() succeeded: {e}", flush=True)
            except Exception as e2:
                print(f"[{job_id[:8]}] Failed to kill process: {e2}", flush=True)
        active_processes.pop(job_id, None)

    # Also kill any orphan processes matching this job if any
    try:
        subprocess.run(["pkill", "-9", "-f", job_id], check=False)
    except Exception:
        pass

    if job_id in jobs_state:
        jobs_state[job_id]["status"] = "aborted"
        jobs_state[job_id]["stage"] = "Aborted by User"
        jobs_state[job_id]["progress"] = 0
        jobs_state[job_id]["error"] = "Job aborted by user."
    else:
        jobs_state[job_id] = {
            "job_id": job_id,
            "status": "aborted",
            "stage": "Aborted by User",
            "progress": 0,
            "error": "Job aborted by user.",
            "splat_path": None
        }

    # Clean up intermediate directory
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f"[{job_id[:8]}] Purged intermediate job directory: {job_dir}", flush=True)
        except Exception as e:
            print(f"[{job_id[:8]}] Error purging job directory: {e}", flush=True)

    return {
        "success": True,
        "jobId": job_id,
        "status": "aborted",
        "processKilled": killed,
        "message": f"Job {job_id} successfully aborted on Nebius GPU worker."
    }


@app.post("/cleanup/{job_id}")
def cleanup_job(job_id: str):
    """
    Purges the /tmp/splatris_jobs/<jobId>/ intermediate folder and frees disk space.
    """
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f"[{job_id[:8]}] Cleaned up intermediate job directory: {job_dir}", flush=True)
            if job_id in jobs_state:
                jobs_state[job_id]["cleaned_up"] = True
            return {"success": True, "message": f"Purged job {job_id} on worker"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": True, "message": "Directory already removed or does not exist"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("gpu_worker_server:app", host="0.0.0.0", port=port, reload=False)
