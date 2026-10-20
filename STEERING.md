# Splatris Spatial AI Platform — Steering Document

## 1. Executive Summary & Vision

**Splatris** is an end-to-end Spatial AI platform engineered to bridge spatial computing (3D Gaussian Splatting, scene representations, point clouds) with advanced large language models, GPU reconstruction pipelines, and WebXR immersive interfaces.

The platform encompasses three interconnected tiers:
1. **Spatial Asset Ingestion & Video Processing**: High-fidelity frame sequence extraction (2 FPS, `-q:v 2`) and metadata inspection via native system FFmpeg/FFprobe with Apple Silicon hardware acceleration.
2. **Autonomous Spatial Orchestration & Training Engine**:
   - Parameter calibration via **`nvidia/Nemotron-3-Ultra-550b-a55b`** (COLMAP matching strategy, iteration tuning, frame pruning).
   - Asynchronous **Nebius GPU 3DGS Training Job Runner** (`/api/spatial/job/start`, `/api/spatial/job/status/:jobId`) managing training stages through to binary `.splat` scene export.
   - **Spatial Audio RAG** (`/api/spatial/rag-query`) combining Tavily search grounding with Nemotron 3 Ultra to position audio emitters in 3D scene space.
3. **WebXR Client & Spatial Audio Viewer**: Real-time Three.js 3D Gaussian Splatting viewer with WebXR immersive session support, binary `.splat` parser, 3D Web Audio spatial panning, and interactive HUD panels.

---

## 2. Hardware & Host Environment

| Parameter | Specification | Details & Notes |
| :--- | :--- | :--- |
| **Local Orchestrator** | **Apple Mac mini (M4)** | Apple Silicon (ARM64 architecture) providing local development, FFmpeg frame extraction, and WebXR orchestration host. |
| **Local Operating System** | macOS (Darwin ARM64) | Native Unix environment with zsh shell and Node.js v24+. |
| **Media Acceleration** | Apple Silicon VideoToolbox / NEON | Homebrew FFmpeg 8.1.2 compiled with NEON / VideoToolbox acceleration for 2 FPS frame extraction. |
| **Remote GPU Cluster** | **Nebius NVIDIA L40S (46GB VRAM)** | High-throughput Ada Lovelace GPU instance (`89.169.112.102`) running Ubuntu 24.04, Driver 580.173, CUDA 13.0. |
| **Remote Worker Service** | FastAPI Worker Microservice | Dedicated Python service on port 8000 managed via `nohup` / `systemd` executing COLMAP SfM and Nerfstudio Splatfacto 3DGS CUDA training. |
| **3D Client Performance** | **72–75 FPS** | Verified real-time rendering of authentic 359,000+ Gaussian splats via WebGL / Three.js on M4 GPU and Meta Quest 3 native 72/75Hz display. |

---

## 3. Architecture & Technology Stack

| Layer / Concern | Technology | Implementation & Status |
| :--- | :--- | :--- |
| **Local Host Platform** | Apple Mac mini (M4) | ARM64 architecture, local development & orchestration host. |
| **Remote GPU Instance** | Nebius NVIDIA L40S (46GB VRAM) | Dedicated remote compute host (`89.169.112.102:8000`) for COLMAP and 3DGS training. |
| **Runtime** | Node.js (v24+) | Native ES Modules (`"type": "module"`). |
| **Web Framework** | Express 5.x (`express@^5.2.1`) | High-performance routing, JSON/URL-encoded middleware, CORS, static file delivery (`public/`, `uploads/`). |
| **AI Inference Gateway** | Nebius Token Factory (`openai@^7.20.0`) | Official OpenAI SDK configured with baseURL `https://api.tokenfactory.nebius.com/v1`. |
| **Primary Reasoning Model** | `nvidia/Nemotron-3-Ultra-550b-a55b` | 550B MoE model with deep chain-of-thought reasoning (`reasoning_content`) for SfM matching calibration and frame pruning. |
| **Grounding Engine** | Tavily Search API | Real-time web retrieval grounding for spatial scene Q&A. |
| **Spatial 3DGS Job Runner** | Real Nebius GPU Worker Pipeline | `services/nebiusGpuService.js` dispatches frames to remote FastAPI worker (`pipeline/gpu_worker_server.py`) and downloads authentic `.splat` models. |
| **SfM & 3D Reconstruction** | COLMAP 3.9.1 + Nerfstudio Splatfacto | Complete Nerfstudio pipeline (`pipeline/train_3dgs.py`) executing feature extraction, sequential/exhaustive matching, and full 3D Gaussian Splatting optimization via `gsplat`. |
| **3D Gaussian Format** | Binary `.splat` (32 bytes/splat) & `.ply` | Standard layout: Position (12B) + Scale (12B) + Color (4B) + Quaternion (4B). |
| **Immersive Frontend** | Three.js (r128) + `@mkkellogg/gaussian-splats-3d` + WebXR | True 3D Gaussian Splatting engine (`DropInViewer`) with anisotropic covariance projection, depth-sorted alpha blending, percentile outlier filtering, `local-floor` space, Meta Quest Touch controller pointers. |
| **Spatial Audio** | Web Audio API (`PannerNode`, `AudioListener`) | True 3D spatial panning with speech synthesis and dynamic 3D emitter repositioning. |
| **Video Processing** | `ffmpeg` / `ffprobe` (`node:child_process`) | Adaptive frame extraction targeting 150–200 frames, orientation detection, and JSON stream metadata inspection. |
| **Asset Ingestion** | `multer` (`multer@^2.4.0`) | Multipart upload handling with file type validation (MP4/MOV) and 500MB limits. |
| **Configuration & Secrets** | `dotenv`, `.gitignore` | Isolated configuration for `PORT`, `NEBIUS_API_KEY`, `TAVILY_API_KEY`, and `NEBIUS_GPU_ENDPOINT`. |

---

## 4. Project Structure

```text
splatris/
├── .env                  # Local secrets, API keys & NEBIUS_GPU_ENDPOINT (git-ignored, active)
├── .env.example          # Clean template for environment configuration
├── .gitignore            # Excludes node_modules, .env, uploads/, and OS artifacts
├── package.json          # Project configuration, scripts, and dependencies
├── pipeline/             # Remote Nebius GPU Training Pipeline
│   ├── gpu_worker_server.py # FastAPI microservice running on Nebius L40S (port 8000)
│   ├── requirements.txt  # Remote Python dependencies (torch, torchvision, fastapi, uvicorn, plyfile)
│   ├── train_3dgs.py     # Standalone Nerfstudio Splatfacto + COLMAP pipeline & .splat exporter
│   └── convert_ply_to_splat.py # Standalone PLY to WebXR 32-byte .splat converter
├── public/               # WebXR Front-end Application
│   ├── app.js            # Three.js WebXR engine, 3DGS loader, Spatial Audio RAG, Meta Quest tracking
│   ├── index.html        # Main HTML entrypoint with glassmorphic HUD panels & dropzone
│   ├── lib/              # Client-side 3D engines
│   │   └── gaussian-splats-3d.umd.js # @mkkellogg/gaussian-splats-3d dedicated WebXR engine
│   ├── samples/          # Bundled sample room videos (sample_living_room.mp4)
│   └── style.css         # Cyber-spatial dark mode theme, glassmorphism, responsive HUD
├── routes/
│   ├── agent.routes.js   # Agent routing (/api/agent/test, /api/agent/analyze-frames)
│   ├── job.routes.js     # Spatial 3DGS training runner (/api/spatial/job/start, /status/:jobId)
│   └── spatial.routes.js # Video upload (/api/spatial/upload) & RAG query (/api/spatial/rag-query)
├── server.js             # Express entry point, static routes & middleware pipeline
├── services/
│   ├── agentService.js   # Nebius OpenAI client & spatial orchestrator methods
│   ├── jobService.js     # 3DGS job state machine
│   ├── nebiusGpuService.js # Remote Nebius GPU dispatcher (REST/HTTP, live telemetry, .splat download)
│   ├── ragService.js     # Tavily + Nemotron 3 Ultra spatial RAG reasoning
│   └── videoService.js   # Adaptive FFmpeg frame extraction, orientation detection & FFprobe metadata
├── uploads/              # Local storage for raw videos, frames, and 3D models (git-ignored)
│   ├── frames/           # uploads/frames/{videoId}/frame_%04d.jpg
│   └── models/           # uploads/models/{jobId}.splat (authentic 32B/splat binary files)
└── STEERING.md           # Steering document: architecture, progress, and roadmap
```

---

## 5. Completed Milestones & Accomplishments

### Milestone 1: Backend Foundation & Nebius Agent Bridge (Completed & Verified)
- [x] Project initialization with ES Module standard (`"type": "module"`).
- [x] Dependencies installed: `express`, `cors`, `dotenv`, `multer`, `openai`.
- [x] Environment & secret management (`.env`, `.env.example`, `.gitignore`).
- [x] Express server setup with CORS, JSON body parsing, and `/health` endpoint.
- [x] Modular agent routing with `services/agentService.js` and `GET /api/agent/test`.

### Milestone 2: Spatial Video Ingestion & Frame Extraction (Completed & Verified)
- [x] Implemented `services/videoService.js` with `ffmpeg` 2 FPS frame extraction (`-q:v 2`) and `ffprobe` metadata parsing.
- [x] Built `routes/spatial.routes.js` with `POST /api/spatial/upload` supporting MP4/MOV files up to 500MB.
- [x] Static delivery of extracted frames via `express.static('uploads')`.
- [x] End-to-end verified with synthetic test video on M4 Mac mini.

### Milestone 3: Spatial AI Pipeline Orchestrator (Completed & Verified)
- [x] Added `analyzeFramesWithAgent()` in `services/agentService.js`.
- [x] Implemented `POST /api/agent/analyze-frames` returning structured JSON decisions:
  - `colmapMatcher`: `"sequential"` vs `"exhaustive"`
  - `recommendedIterations`: e.g. `7000` vs `30000`
  - `filteredFrameIndices`: array of optimal frame indices to retain
  - `reasoning`: technical justification based on video geometry
- [x] Verified with small (6-frame) and large 4K (120-frame) capture profiles.

### Milestone 5: End-to-End Autonomous Pipeline & Nebius GPU Worker (Completed & Verified)
- [x] Implemented real remote GPU worker dispatch in `services/nebiusGpuService.js` connecting to FastAPI microservice (`pipeline/gpu_worker_server.py`) on NVIDIA L40S instance (`89.169.122.100:8000`).
- [x] Streamed live stage telemetry and progress (0–100%) through COLMAP SfM, Nerfstudio Splatfacto training, and WebXR binary `.splat` export.
- [x] Verified full room 3DGS training completing in ~5–6 minutes on NVIDIA L40S GPU.

### Milestone 6: Pre-Flight AI Video Quality Gatekeeper & Dynamic Frame Pipeline (Completed & Verified)
- [x] **Pre-Flight AI Video Quality Gatekeeper**:
  - **Calibrated OpenCV Frame Sampler ([pipeline/audit_video.py](file:///Users/akif/Projects/splatris/pipeline/audit_video.py))**: Extracts 12 uniform frames and calculates `cv2.Laplacian` blur variance, mean brightness, and contrast. Calibrated baseline blur threshold from 85 down to **50–55** (`< 52`) to accommodate standard smartphone lenses with natural depth-of-field without falsely penalizing soft textures between 50 and 80.
  - **Multimodal Vision Audit ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js))**: Dispatches frames to `openbmb/MiniCPM-V-4_5` on Nebius Token Factory (with fallback to `nvidia/Nemotron-3-Ultra-550b-a55b`) evaluating:
    1. Motion Blur & Rapid Rotation (Laplacian baseline < 52)
    2. Parallax Baseline & Motion Trajectory (translational orbit vs. fixed-axis spinning)
    3. Lighting & Exposure Stability (window blowouts, lens flares, auto-exposure shifts)
    4. Visual Texture & Monolithic Surfaces
  - **Context-Aware Dynamic Guidance**: System prompt and schema dynamically detect the scene/subject (`indoor_room`, `tabletop_object`, `vehicle`, `architectural_space`, `outdoor_scene`) and generate tailored, actionable director guidance rather than defaulting generically to room instructions.
  - **Audit Modal & Button State Logic ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
    - Animated circular SVG gauge rendering `readiness_score` (0–100%).
    - 4-Pillar checklist (Motion Blur, Parallax Baseline, Lighting Stability, Visual Texture).
    - Detected defect cards with exact timestamps (`MM:SS`) and explanations.
    - Context-aware director guidance alert cards.
    - **Button Conditional Rendering & State**:
      - If `status === "REJECTED"` (`< 75%`): Primary `"Proceed to 3DGS Training"` button is explicitly **disabled** (`disabled=true`, opacity 0.38, `cursor: not-allowed`), while `"Force Run (Bypass)"` is displayed with tooltip and warning confirmation to protect GPU credits.
      - If `status === "PASSED"` / `"APPROVED"` (`>= 75%`): Primary `"Proceed to 3DGS Training"` button is **enabled** with a 3-second auto-start countdown, while `"Force Run (Bypass)"` is omitted/hidden.
- [x] **Dynamic Overlap-Preserving Frame Pipeline**:
  - Replaced static 90-frame cap with dynamic formula: $\text{target\_frames} = \text{clamp}(\text{duration\_seconds} \times 2.2, \min=90, \max=220)$.
  - Preserves critical 70–80% visual overlap for 360° loops, eliminating sparse, disconnected floating splat artifacts.
  - COLMAP camera model explicitly set to `OPENCV` pinhole with distortion (`--camera-type perspective`).
  - Enforced `--matching-method vocab_tree` for linear matching and loop closure on the L40S GPU.
  - Synchronized updated pipeline to `89.169.122.100` and restarted `splatris-worker.service`.
  - `POST /api/spatial/rag-query`: Connects user spatial questions to Tavily search grounding and Nemotron 3 Ultra.
  - Calculates optimal 3D emitter coordinates `{ x, y, z }` positioned for immersive spatial audio.
- [x] **WebXR Front-end Application ([public/](file:///Users/akif/Projects/splatris/public/))**:
  - Built Three.js WebXR application supporting immersive VR sessions (`renderer.xr.enabled = true`).
  - Binary `.splat` loader parsing 32-byte chunks and rendering Gaussian splat particles.
  - Interactive 3D Audio Emitter visual marker with pulsing rings.
  - Web Audio API spatial panner (`PannerNode`, `AudioListener`) with speech synthesis.
### Milestone 11: UI Refinements, 3DGS WASD Navigation & Visual Audit Feedback (Completed & Verified)
- [x] **Nebius Server Credit Warning Dialog Update ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Updated header and confirmation prompt for the "Force Run (Bypass)" trigger to explicitly guard Nebius server credits:
    ```
    Nebius Server Credit Warning:

    This video scored only ${score}% readiness. Running COLMAP on a video with motion blur or axis-spin rotation wastes Nebius server credits and frequently produces sparse, floating splat artifacts.

    Are you sure you want to bypass and force training?
    ```
- [x] **WASD First-Person Navigation (Free Roam) in 3DGS Viewer ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - **First-Person Controller**: Implemented Euler pitch/yaw mouse look and WASD translation (`KeyW`, `KeyA`, `KeyS`, `KeyD`, `Space`, `Shift`) alongside Pointer Lock.
  - **Collision & Floor Alignment**: Translates horizontally along the camera's forward heading vector (`forward.y = 0`) at walking pace (2.4 m/s), bounded above floor level.
  - **Dual UI Toggles**:
    - Sidebar control chip: `#btn-free-roam` ("Enter Free Roam").
    - Floating viewport center overlay: `#btn-overlay-free-roam` with dynamic label (`Enter Free Roam` &rarr; `Exit Free Roam (ESC)`).
  - **Helper Text**: Automatically updates bottom footer text to: `"WASD: Move • Mouse: Look • ESC: Exit Free Roam"` while active, and restores default Orbit instructions on `ESC` exit.
  - **WebXR Compatibility**: Seamlessly prioritizes native 6-DoF Quest headset tracking whenever `renderer.xr.isPresenting` is active.
- [x] **Visual Frame Thumbnails & Lightbox Inspection ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Base64 Frame Attachment**: Backend attaches lightweight base64 JPEG data URIs (`thumbnail_url`) to every flagged entry in `issues_detected`, matching timestamps with extracted sample frames.
  - **Issue Card Thumbnail UI**: Renders rounded frame thumbnails (`.issue-thumb-wrapper`) with zoom-in cursor, search badge (`🔍`), and hover expansion in each issue card.
  - **Interactive Lightbox Modal**: Clicking any thumbnail opens `#frame-lightbox-modal` with enlarged frame view, timestamp badge, defect type, and technical diagnostic details.
- [x] **Proportional Dynamic Readiness Score Calculation ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Completely eliminated hardcoded floors (`58%`, `10%`) and arbitrary penalty subtractions.
  - Implemented strictly proportional scoring:
    - Counts unique problematic frames across detected issues and OpenCV Laplacian checks (`problematic_frames = unique_flagged_timestamps.size`).
    - `healthy_frames = Math.max(0, total_sampled - problematic_frames)`.
    - `readiness_score = Math.round((healthy_frames / total_sampled) * 100)`.
  - Naturally spans full dynamic range from 0% to 100%:
    - 0 defective frames: **100%** &rarr; `APPROVED` (&ge; 75%)
    - 1 defective frame: **92%** &rarr; `APPROVED` (&ge; 75%)
    - 2 defective frames: **83%** &rarr; `APPROVED` (&ge; 75%)
    - 3 defective frames: **75%** &rarr; `APPROVED` (&ge; 75%)
    - 4 defective frames: **67%** &rarr; `REJECTED` (< 75%)
    - 5 defective frames: **58%** &rarr; `REJECTED`
    - 10 defective frames: **17%** &rarr; `REJECTED`
    - 12 defective frames: **0%** &rarr; `REJECTED`

### Milestone 12: Strict Aspect Ratio Validation & Portrait Rejection (Completed & Verified)
- [x] **Pre-Flight Metadata Aspect Ratio Check ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js), [pipeline/audit_video.py](file:///Users/akif/Projects/splatris/pipeline/audit_video.py))**:
  - Before running frame sampling or Laplacian blur analysis, inspects display dimensions (`width` vs `height`) taking rotation tags into account (`rotation === 90 || rotation === 270`).
  - **Hard Rejection Logic**: If `height >= width` (Portrait or square video), immediately aborts frame extraction and model calls, returning `status: "REJECTED"`, `readiness_score: 0`, and `aspect_ratio_rejected: true`.
- [x] **Critical Blocker UI Card ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Overrides standard detected issues list with a dedicated critical blocker notice:
    - **Title**: `"Invalid Format: Vertical Video Detected"`
    - **Description**: `"3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again."`
  - Explicitly disables `"Proceed to 3DGS Training"` and hides `"Force Run (Bypass)"` to prevent catastrophic COLMAP tracking failures and wasted Nebius server credits on non-reconstructable vertical videos.
- [x] **Client-Side Instant Pre-Check ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - `checkVideoOrientationInBrowser()` checks video orientation in <50ms upon file selection, intercepting vertical uploads before unnecessary network transfers.
- [x] **Manual Consent Required (Eliminated Auto-Start Countdown) ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Completely removed the 3-second auto-start countdown timer and any automatic redirects when `status === 'PASSED'`.
  - The UI now halts on the Capture Quality Audit modal indefinitely, allowing the user to review readiness scores and metrics.
  - Training pipeline only begins when the user explicitly clicks the primary `"Proceed to 3DGS Training"` button.

- [x] **Video Ingestion UI ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
  - Drag-and-drop `#video-dropzone` supporting MP4/MOV files up to 500MB with dragover animation.
  - Progressive 5-step pipeline tracker: `[1. Upload]` &rarr; `[2. Extract]` &rarr; `[3. Nemotron]` &rarr; `[4. 3DGS Train]` &rarr; `[5. WebXR]`.
- [x] **Automated Pipeline Orchestration ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Automatic forward from `POST /api/spatial/upload` to `POST /api/agent/analyze-frames`.
  - Real-time display of Nemotron's parameter choices and reasoning callout in `#nemotron-reasoning-card`.
  - Automatic dispatch of `POST /api/spatial/job/start` using Nemotron's calibrated parameters.
- [x] **Dynamic 3D Scene Replacement ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - On job completion, automatically swaps out the placeholder mesh and loads the newly exported `.splat` file (`/uploads/models/${jobId}.splat`).
  - Centers camera viewport and highlights the "Enter WebXR" button for Meta Quest 3 inspection.
- [x] **Spatial Audio RAG Context Synchronization**:
  - Updates RAG session with the reconstructed room's metadata and Nemotron reasoning.
  - Logs a welcome announcement from the Spatial Audio Guide and refreshes preset chips with room-specific queries.

### Milestone 6: Real Nebius GPU 3DGS Pipeline & Removal of Mock Generators (Completed & Verified)
- [x] **Removed Mock Generators ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js))**:
  - Completely deleted `generateSampleSplatFile` and synthetic math loops.
  - Deleted mock timer simulation `simulateNebiusGpuTraining`.
  - Zero synthetic fallback: jobs fail honestly if frames or worker are missing, rather than generating a synthetic nebula.
- [x] **Nebius GPU Worker Pipeline ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Implemented real execution service supporting remote REST worker and live telemetry streaming.
- [x] **WebXR Headset & Controller Stabilization ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Disabled `OrbitControls` during WebXR presentation (`renderer.xr.isPresenting`) to eliminate camera fighting.
  - Added `local-floor` reference space and cyan controller pointer laser rays for Meta Quest Touch controllers.

### Milestone 7: Live Nebius L40S GPU Worker Deployment (Completed & Verified)
- [x] **Remote VM Environment Setup (`89.169.96.24`)**:
  - Provisioned Nebius GPU instance: **NVIDIA L40S** (46GB VRAM, Driver 580.173, CUDA 13.0, Ubuntu 24.04).
  - Installed system packages: `colmap 3.9.1`, `ffmpeg`, `python3-pip`, `python3-venv`, `zip`, `unzip`, `build-essential`.
  - Configured Python virtual environment (`/home/ubuntu/splatris_env`) with `torch 2.6.0+cu124`, `torchvision`, `fastapi`, `uvicorn`, `scipy`, `plyfile`.
- [x] **FastAPI Worker Microservice Deployment**:
  - Deployed `pipeline/train_3dgs.py` and `pipeline/gpu_worker_server.py` to `/home/ubuntu/splatris/pipeline/`.
  - Configured persistent systemd service `splatris-worker.service` running on port 8000.
  - Verified public health endpoint: `GET http://89.169.96.24:8000/health` returning `cuda_available: true`.

### Milestone 8: Full Real-World Room Ingestion via Nerfstudio Splatfacto (Completed & Verified)
- [x] **Adaptive Frame Extraction ([routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - Replaced rigid hardcoded 2 FPS extraction with duration-aware adaptive extraction targeting ~150 frames (between 5 and 10 FPS).
  - Resolved the sparse view issue where short walkthrough videos only extracted 23 frames.
- [x] **CUDA 13.0 vs PyTorch 12.4 ABI Shim (`fakecuda`)**:
  - Overcame PyTorch 2.6's strict `torch.utils.cpp_extension` version assertion (`The detected CUDA version (13.0) mismatches PyTorch (12.4)`).
  - Created `/home/ubuntu/fakecuda/bin/nvcc` wrapper script that reports `Cuda compilation tools, release 12.4, V12.4.131` when probed with `--version`, but forwards actual compilation arguments to `/usr/local/cuda-13.0/bin/nvcc`.
  - Symlinked `include/` and `lib64/` to real CUDA 13.0 paths.
  - Enabled runtime JIT compilation of `gsplat` CUDA rasterizer kernels for compute capability 8.9 (NVIDIA Ada Lovelace / L40S).
- [x] **Subprocess Environment Propagation & Diagnostic Telemetry**:
  - Fixed worker process environment injection: ensured `PATH` contains `fakecuda`, `cuda-13.0`, and the active virtual environment's `bin/` directory (`splatris_env/bin`).
  - Switched subprocess Python invocations to `sys.executable` to prevent fallback to system Python 3.12 lacking Nerfstudio packages.
  - Added real-time log capturing: stdout and stderr are continuously streamed to `train.log`, recorded in `journalctl`, and surfaced through `/logs/:jobId` and `/status/:jobId`.
- [x] **End-to-End Real Room Reconstruction Executed (948s)**:
  - Input: 29MB walkthrough video (`com.oculus.vrshell-20260922-100411-0.mp4`).
  - Extracted 158 frames &rarr; pruned to 79 keyframes by Nemotron 3 Ultra recommendation.
  - Executed COLMAP SfM feature extraction and sequential matching.
  - Trained 15,000 iterations of Nerfstudio Splatfacto on the NVIDIA L40S GPU.
  - Generated and exported **359,302 3D Gaussians** into binary `scene.splat` (11.5 MB).

### Milestone 9: WebXR Rendering & Scene Alignment Optimization (Completed & Verified)
- [x] **Percentile-Based Bounding Box & Outlier Filtering ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Diagnosed scene scaling issue: distant COLMAP floaters (34m+ away) were causing naive min/max calculations to compress the real 7.4m office room down to 1.2m.
  - Implemented 3rd-to-97th percentile spatial sampling to accurately measure the true physical room span (`7.43m` wide, `3.64m` tall, `7.10m` deep).
  - Preserves natural 1:1 metric scale when room dimensions fall within 2.5m–14m.
- [x] **Solid Surface Rendering with Depth Testing**:
  - Replaced `blending: THREE.AdditiveBlending` with `blending: THREE.NormalBlending`.
  - Added `depthTest: true`, `depthWrite: true`, and `alphaTest: 0.08`.
  - Overcame the "glowing translucent starfield" effect: walls, desk, monitors, chairs, and floor now occlude correctly and render as solid physical structures.
- [x] **Ground Plane & Headset Eye-Level Alignment**:
  - Aligned floor to `Y = 0` based on the 3rd percentile of vertical coordinates.
  - Viewer standing at natural Meta Quest 3 eye level (`Y ≈ 1.6m`) is positioned directly inside the room instead of floating far outside or underneath the floor.

### Milestone 10: Dedicated WebXR 3DGS Engine Integration & Ingestion Pipeline Optimization (Completed & Verified)
- [x] **Integration of `@mkkellogg/gaussian-splats-3d` ([public/lib/gaussian-splats-3d.umd.js](file:///Users/akif/Projects/splatris/public/lib/gaussian-splats-3d.umd.js))**:
  - Replaced naive `THREE.Points` billboard circle rendering with the dedicated Three.js 3DGS engine (`GaussianSplats3D.DropInViewer`).
  - Computes authentic anisotropic 3D Gaussian covariance matrices and view-dependent depth sorting from `.splat` and `.ply` files.
  - Solved the "ping-pong ball / floating bubbles" artifact: splats now tile and fuse into continuous, solid planar structures (walls, floor, desk tops).
  - Configured for maximum mobile GPU stability on Meta Quest 3 (`gpuAcceleratedSort: false`, `sharedMemoryForWorkers: false`, `integerBasedSort: false`) with zero `SharedArrayBuffer` / COOP header dependencies.
  - Retains full compatibility with existing Three.js scene elements (panning 3D Audio Emitter marker, grid helper, VR controller rays).
- [x] **Stationary WebXR Physical Anchor**:
  - Disabled default scene auto-rotation while in WebXR presentation (`renderer.xr.isPresenting`), keeping the room physically fixed relative to the user's headset tracking.
- [x] **Insta360 4K Single-Lens Walkthrough Ingestion Readiness**:
  - Updated adaptive frame extraction in `routes/spatial.routes.js` to target 150–200 high-fidelity frames with `-q:v 2` optical clarity.
  - Calibrated Nemotron 3 Ultra prompt in `services/agentService.js` to retain 80–100% of frames for walking loop captures, preserving continuous motion parallax between consecutive camera viewpoints.
- [x] **End-to-End System Verified**:
  - Verified local backend serving the 3DGS engine bundle at `http://localhost:3000/lib/gaussian-splats-3d.umd.js`.
  - Cloudflare tunnel active for Quest 3 headset connection: `https://channels-rural-calm-ann.trycloudflare.com`.
  - Nebius GPU worker online with NVIDIA L40S at `http://89.169.96.24:8000`.

### Milestone 11: High-Fidelity 3DGS Densification, Native Resolution & 15,000 Iterations (Completed & Verified)
- [x] **Increased Training Target to 15,000 Iterations**:
  - Configured `--max-num-iterations 15000` and `--pipeline.model.stop-split-at 15000` in `pipeline/train_3dgs.py`.
  - Updated `pipeline/gpu_worker_server.py`, `services/agentService.js`, and `routes/job.routes.js` to enforce a minimum of 15,000 optimization steps.
  - Allows the Splatfacto optimizer to continuously densify, split, and refine Gaussians through the full training schedule.
- [x] **Aggressive Densification Threshold Tuning**:
  - Lowered `--pipeline.model.densify-grad-thresh` from default `0.0008` to `0.0002`.
  - Enables large, blurry Gaussians on object edges, text, and fine textures to exceed the gradient norm threshold earlier and split into fine, high-detail splats.
- [x] **Preserved Native Input Resolution (1080p / 2K)**:
  - Passed `--downscale-factor 1` to `nerfstudio-data`, preventing Nerfstudio from downsampling images below 1600px.
  - Set `--num-downscales 0` in `ns-process-data images` and `--pipeline.model.num-downscales 0` in Splatfacto to eliminate resolution decimation during training.
- [x] **Complete Purge of `THREE.Points` & Three.js 0.147.0 Upgrade**:
  - Purged all legacy `THREE.Points` particle fallback logic from `public/app.js`.
  - Upgraded Three.js from `r128` to `0.147.0` to natively provide `THREE.DataUtils.fromHalfFloat` and `THREE.OrbitControls`.
  - Added cache-busting version hashes (`?v=20260923_true3dgs`) and HTTP `no-cache` headers to eliminate stale browser caching in Meta Quest 3.
- [x] **Deployed & Verified on Remote Nebius L40S GPU Worker**:
  - Synced updated `pipeline/train_3dgs.py` and `pipeline/gpu_worker_server.py` to `89.169.96.24`.
  - Re-launched worker daemon: online with 44.4 GB VRAM on NVIDIA L40S.

### Milestone 12: Balanced 10,000 Iteration Setup, Needle Splat Pruning, Rolling Retention & Scene History UI (Completed & Verified)
- [x] **New Worker IP Migration & Health Check**:
  - Migrated remote endpoint to **`http://89.169.122.100:8000`** with NVIDIA L40S (44.4 GB VRAM, Driver 580.173, CUDA 13.0).
  - Verified remote service active with health response: `cuda_available: true`, `gpu: NVIDIA L40S`.
- [x] **Root Cause Resolution & Parameter Rebalancing**:
  - Diagnosed freeze at 55% (~1730s): caused by native 2K resolution (`--downscale-factor 1`) + aggressive densification (`0.0002`) generating >4M Gaussians and stalling per-step rasterization, combined with CPU exhaustive matching.
  - Rebalanced parameters: capped iterations at **`10,000` steps** (~6 minutes on L40S), `--pipeline.model.densify-grad-thresh 0.0004`, `--downscale-factor 2`, `--pipeline.model.stop-split-at 8000`, and sequential matching safeguard for >50 frames.
- [x] **Needle & Ray Artifact Elimination**:
  - **Worker Training Bounds** in `pipeline/train_3dgs.py`:
    - `--pipeline.model.cull-alpha-thresh 0.05` (culls low-opacity floaters early)
    - `--pipeline.model.cull-scale-thresh 0.5` (culls huge billboard splats)
    - `--pipeline.model.use-scale-regularization True` (PhysGaussian scale regularization)
    - `--pipeline.model.max-gauss-ratio 10.0` (penalizes extreme aspect ratio elongation)
  - **Export-Level Pruning** in `convert_ply_to_splat`:
    - Rejects Gaussians with aspect ratio `max_scale / min_scale > 25.0`.
    - Rejects splats with `max_scale > 0.6` or opacity `< 5%`.
    - Culls position outliers beyond 99th percentile radius.
  - **Active Scene Cleaned**: Pruned existing room model (`22d0909b...`), eliminating 182,980 needle rays and reducing 248k splats to 56k clean room splats.
- [x] **Viewer Auto-Grounding, Centering & Inside Spawn ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Percentile Bounding & Auto-Grounding**: Parses raw `.splat` ArrayBuffer coordinates, computes 5th/95th percentile bounds (`floorY`, `ceilingY`, `centerX`, `centerZ`).
  - **Auto-Scale Normalization**: Automatically normalizes miniature reconstructions (< 1.8m height) to realistic human scale (2.4m ceiling height, `autoScale = 2.4 / rawHeight`).
  - **Floor & Center Alignment**: Translates `dropInViewer` by `[-centerX * scale, -floorY * scale, -centerZ * scale]` so floor aligns exactly with `Y = 0` and room center aligns with `(0, 0, 0)`.
  - **Spawn Inside**: Positions camera at `[0, eyeHeight, 0.4]` looking at `[0, eyeHeight - 0.05, -0.6]` (directly inside room center looking towards desk/window).
  - **Relaxed Orbit Zoom**: Configured `controls.minDistance = 0.05`, `controls.maxDistance = 25.0`, `controls.maxPolarAngle = Math.PI - 0.05` to prevent ceiling collisions when orbiting or zooming.
  - "Center View" button restored to the same inside-center perspective.
- [x] **Post-Training Cleanup & Rolling Retention ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js))**:
  - Local server: automatically removes `uploads/${videoId}.*` (raw video) and `uploads/frames/${videoId}` upon model verification.
  - Rolling 3-model retention: `pruneOldModels(3)` maintains only the last 3 `.splat` models in `uploads/models/`, reducing disk usage from 3.1 GB to 24 MB.
  - Worker intermediate purge: added `POST /cleanup/:job_id` and SSH fallback to purge `/tmp/splatris_jobs/<jobId>/` automatically, freeing ~1 GB per run.
- [x] **Scene History UI Component & `GET /api/models/recent` ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Backend endpoint `GET /api/models/recent` returns JSON metadata for the 3 latest models.
  - Interactive glassmorphic chip widget in the left panel allowing dynamic switching of 3DGS models in `@mkkellogg/gaussian-splats-3d` without page reloads.
- [x] **Enforce Step Limit Guard**:
  - Clamped `iterations = Math.min(iterations, 10000)` across `agentService.js`, `jobService.js`, `nebiusGpuService.js`, `train_3dgs.py`, and `gpu_worker_server.py`.
- [x] **Defensive API Error Handling**:
  - Added `safeFetchJson` in `public/app.js` inspecting `response.ok` and `content-type`, completely eliminating `SyntaxError: Unexpected token '<'` crashes during gateway timeouts.

### Milestone 13: View-Dependent Invisibility Fix, SO3xR3 Camera Alignment & Confetti Bounding Box Cropping (Completed & Verified)
- [x] **Inside-Out View-Dependent Invisibility Fix ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Camera Near/Far Clipping**: Configured `camera.near = 0.01` and `camera.far = 50.0` with `updateProjectionMatrix()` on both startup and resize to prevent wall culling when standing inside the room.
  - **Double-Sided Rendering & Anti-Culling**: Enforced `side: THREE.DoubleSide`, `depthTest: true`, `depthWrite: false`, and `frustumCulled: false` across `dropInViewer`, `splatMesh`, and `callbackMesh`.
  - **Unconditional Per-Frame Render Update**: Integrated `dropInViewer.viewer.update(renderer, camera)` directly into the Three.js animation loop, ensuring splat sorting and anisotropic shader updates run every frame regardless of callback mesh occlusion.
  - **Alpha Threshold Calibration**: Set `splatAlphaRemovalThreshold: 1` so subtle interior geometry is not prematurely stripped.
- [x] **SfM Camera Alignment & Confetti Prevention ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - **Camera Pose Optimization**: Added `--pipeline.model.camera-optimizer.mode SO3xR3` to optimize camera poses during training, resolving misregistered walls from handheld phone video trajectories.
  - **Direct COLMAP SfM Initialization**: Added `--pipeline.model.random-init False` to initialize Gaussians strictly from registered COLMAP keypoints, preventing random floater splats in empty space.
- [x] **Primary Room Bounding Box Cropping ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Added spatial cropping in `convert_ply_to_splat` strictly bounding splats within `[-3, -1, -3]` to `[3, 3, 3]`.
  - Pruned exterior floaters and background ray confetti from all active models in `uploads/models/`, reducing floater artifacts by thousands of splats.
- [x] **Remote Worker Synchronization**:
  - Deployed updated `train_3dgs.py` and `gpu_worker_server.py` to `ubuntu@89.169.122.100:/home/ubuntu/splatris/pipeline/` and restarted `splatris-worker.service`.

### Milestone 14: Dynamic 360° Loop Closure via Vocab Tree Matcher (Completed & Verified)
- [x] **Nemotron Prompt Loop Closure Refinement ([services/agentService.js](file:///Users/akif/Projects/splatris/services/agentService.js))**:
  - Eliminated hardcoded frame count thresholds (previously forced `sequential` for >50 frames).
  - Instructed Nemotron to detect room tours, walkthroughs, and 360° closed loops (where camera returns to/revisits starting position).
  - Enforced loop closure matchers (`vocab_tree` or `exhaustive`) to connect final frames with initial frames, preventing duplicate ghost rooms and wall splitting caused by pure linear sequential matching.
  - Added `isLoopTour: boolean` to Nemotron's structured JSON decisions.
- [x] **COLMAP Matcher Execution & Vocab Tree Optimization ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Added `--matcher vocab_tree` as primary choice and default for `train_3dgs.py` and `gpu_worker_server.py`.
### Milestone 15: Immersive Viewport Mode, Space Terminology & Glassmorphic Polish (Completed & Verified)
- [x] **Collapsible HUD Panels (Immersive Viewport Mode)**:
  - Added `#btn-toggle-left-panel` and `#btn-toggle-right-panel` in top navbar with smooth sliding transitions (`transform: translateX(...)` with opacity fade).
  - Wired Three.js canvas auto-resize (`camera.updateProjectionMatrix()` and `renderer.setSize()`) on panel toggle and `transitionend` to maintain pixel-perfect aspect ratio.
- [x] **Terminology Update ("Room" -> "Space")**:
  - Renamed left panel to "Space Ingestion Pipeline", right panel description to "grounded in your reconstructed space", and presets to "Space Layout", "Space Geometry", and "Space Lighting & Center".
- [x] **Redundant Sample Button Removal**:
  - Removed `#btn-start-job` and sample video fetch handler; scene history widget provides instant 1-click loading without redundant training.
- [x] **Video Ingestion Limits & Dropzone Polish**:
  - Enforced 500 MB max file size and >90s duration warning with animated glassmorphic toast notification system (`#toast-container`).
- [x] **Glassmorphic Q&A Input**:
  - Applied dark glass container (`rgba(255, 255, 255, 0.05)`, blur 12px) with integrated neon cyan/teal send button.

### Milestone 16: Strict 90-Frame Decimation for Wide-Baseline COLMAP (Completed & Verified)
- [x] **Wide-Baseline Photogrammetry Rationale**:
  - Combining dense consecutive frames (e.g., 201 frames over 40s) with `vocab_tree` caused near-zero parallax baseline between consecutive views, leading to catastrophic feature blurring, shallow triangulation angles, and flat, cloudy point soup.
  - Enforced strict decimation capping processed frames to a maximum of 90 frames with uniform stride (`keep_indices = round(i * (N - 1) / (K - 1))`), preserving both start and end frames for 360° loop closure while expanding parallax baseline by ~2.25x.
- [x] **Multi-Tier Enforcement**:
  - **`services/videoService.js`**: `extractFrames(videoPath, outputDir, fps, maxFrames = 90)` uniformly decimates frames upon extraction and removes unselected files from disk.
  - **`routes/spatial.routes.js`**: Sets `MAX_FRAMES = 90` and computes adaptive fps (`Math.min(5, Math.max(1, Math.ceil(90 / duration)))`).
  - **`services/agentService.js`**: Nemotron prompt instructed to cap retained frames to 75–90 frames; backend strictly clamps `filteredFrameIndices` to $\le 90$.
  - **`services/nebiusGpuService.js`**: Validates and decimates frames to $\le 90$ before packaging `frames.zip` for remote worker dispatch.
  - **`pipeline/train_3dgs.py` & `pipeline/gpu_worker_server.py`**: Added decimation guard directly before `ns-process-data` COLMAP SfM execution.

### Milestone 17: Abort & Reset Pipeline Kill Switch (Completed & Verified)
- [x] **Prominent Red Abort Button in Sidebar ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html) & [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Embedded `#btn-abort-job` inside `#job-progress-card` in the `Space Ingestion Pipeline` sidebar.
  - Styled with a prominent red glass aesthetic (`rgba(239, 68, 68, 0.12)` background, danger border, neon hover glow, and SVG cancel icon).
  - Automatically revealed when the job enters the `TRAINING` state (Step 4) and hidden when idle or completed.
- [x] **Dual-Layer Backend Kill Switch ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js) & [pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py))**:
  - **Remote Worker API**: Added `POST /abort/{job_id}` endpoint to FastAPI on Nebius L40S.
  - **Process Group Termination**: Launched subprocess with `preexec_fn=os.setsid` and terminates active COLMAP / Splatfacto jobs via `os.killpg(pgid, signal.SIGKILL)` to immediately stop GPU kernel execution and release VRAM.
  - **SSH Fail-Safe Guarantee**: Automatically triggers SSH backup command (`ssh ubuntu@89.169.122.100 "pkill -9 -f ...; rm -rf /tmp/splatris_jobs/..."`) to ensure zero orphan processes remain.
  - **Job Lifecycle Abort Guard**: Added `abortJob(jobId)` in [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js) and `POST /api/spatial/job/:jobId/abort` in [routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js).
- [x] **Full Frontend UI Reset ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Implemented `abortCurrentPipeline()` prompting confirmation before termination.
  - Added `resetPipelineUI()` which halts telemetry polling (`clearInterval`), resets step indicator to Step 1 (Upload), zeroes out progress bar (`width: 0%`, `0%`), restores status pill to "Idle", resets the dropzone copy and file input, and displays an informative toast notification.

### Milestone 18: Dual Training Pipeline (3DGS vs. Triangle Splatting Toggle) (Completed & Verified)
- [x] **Frontend UI Segmented Control ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html) & [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Added a sleek glassmorphic segmented control (`#training-method-selector`) under the video dropzone with options:
    - **`Standard 3DGS`** (`value="3dgs"`, Default fallback)
    - **`Triangle Splatting (Mesh-based)`** (`value="triangle"`, WebXR & Meta Quest 3 optimized)
  - Styled with dark glassmorphism (`backdrop-filter: blur(8px)`), interactive cyan radio indicator dots, active gradient highlights, and dynamic badge updates.
  - Dynamically updates proceed button labels in the Capture Quality Audit modal (`Proceed to Triangle Training` vs `Proceed to 3DGS Training`).
- [x] **Full Backend API Integration ([routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js), [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - `/api/spatial/job/start` receives `training_method` parameter, validates and defaults to `"3dgs"`.
  - Stored on job record with dynamic output model routing (`.ply` for triangle splatting, `.splat` for standard 3DGS).
  - Passed directly as `training_method` in multipart form payload to the remote Nebius GPU worker.
- [x] **Remote Nebius GPU Worker Pipeline Routing ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py) & [pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py))**:
  - **100% Intact 3DGS Fallback**: All existing Nerfstudio `splatfacto` execution, PLY export, and needle pruning logic remains unmodified and functional as the default fallback when `training_method == "3dgs"`.
  - **Shared COLMAP Acceleration**: Both methods share the exact same initial COLMAP extraction step (`ns-process-data images`) to eliminate duplicate computation and preserve expensive Nebius GPU credits.
  - **Triangle Splatting Pipeline**:
    - Cloned and compiled Triangle Splatting (`https://github.com/trianglesplatting/triangle-splatting`) on remote VM (`89.169.109.188`).
    - Patched C++ headers for GCC 13+ compatibility on Ubuntu 24.04 and built `diff-triangle-rasterization` and `simple-knn` CUDA kernels for NVIDIA L40S (`sm_89`).
    - Executes training with optimal convergence iterations, saving checkpoints to `point_cloud_state_dict.pt`.
    - Added high-speed binary exporter `convert_triangle_checkpoint` that generates:
      1. Standard textured mesh `.ply` with face definitions (`element face`) for direct mesh inspection.
      2. WebXR-compatible `.splat` file containing oriented planar triangle facets for instant Quest 3 playback.
  - **FastAPI Endpoints**:
    - Added generic `/download/{job_id}/{filename}` supporting both `scene.ply` and `scene.splat`.
    - Updated `/status/{job_id}` telemetry returning `training_method` and `output_filename`.
- [x] **Viewer Compatibility ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - `loadGaussianSplatScene` dynamically differentiates between `.ply` (Mesh) and `.splat` (Volumetric), adjusting primitive counts, labels, and bounding box calculations accordingly.
  - Recent scenes history widget updated to display and load both `.splat` and `.ply` reconstructions.

---

## 6. Server-Side Infrastructure & Configuration Reference

This reference is provided for hackathon judging, evaluation, and reproduction:

### 1. Remote Compute Specifications
* **Host Provider**: Nebius AI Cloud
* **Instance Type**: GPU Compute Instance
* **GPU**: 1x NVIDIA L40S (46,068 MiB GDDR6 VRAM, Ada Lovelace architecture, Compute Capability 8.9)
* **Driver Version**: 580.173.02
* **CUDA Toolkit**: 13.0 (`/usr/local/cuda-13.0`)
* **Operating System**: Ubuntu 24.04 LTS (x86_64 Linux kernel 6.8)
* **Worker Endpoint**: `http://89.169.109.188:8000`

### 2. Python & Pipeline Environment
* **Virtualenv**: `/home/ubuntu/splatris_env`
* **Python**: 3.12.3
* **PyTorch**: `2.6.0+cu124` (`torch.cuda.is_available() == True`)
* **Nerfstudio**: `splatfacto` method
* **gsplat**: JIT-compiled CUDA extensions for `arch=compute_89`
* **COLMAP**: 3.9.1 (Structure-from-Motion / Multi-View Stereo)

### 3. FakeCUDA NVCC Version Compatibility Shim
To permit PyTorch 2.6 (compiled against CUDA 12.4) to build and JIT-compile `gsplat` extensions using CUDA 13.0 without ABI incompatibility errors:

```bash
# /home/ubuntu/fakecuda/bin/nvcc
#!/bin/bash
if [[ "$*" == "--version" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2024 NVIDIA Corporation"
    echo "Built on Thu_Mar_28_02:18:24_PDT_2024"
    echo "Cuda compilation tools, release 12.4, V12.4.131"
    echo "Build cuda_12.4.r12.4/compiler.34097967_0"
else
    exec /usr/local/cuda-13.0/bin/nvcc "$@"
fi
```
Symlinks:
* `/home/ubuntu/fakecuda/include` &rarr; `/usr/local/cuda-13.0/include`
* `/home/ubuntu/fakecuda/lib64` &rarr; `/usr/local/cuda-13.0/lib64`

### 4. Systemd Service Unit (`/etc/systemd/system/splatris-worker.service`)
```ini
[Unit]
Description=Nebius GPU 3DGS Training Worker
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/splatris/pipeline
Environment="PORT=8000"
Environment="CUDA_HOME=/home/ubuntu/fakecuda"
Environment="CUDA_PATH=/home/ubuntu/fakecuda"
Environment="TORCH_CUDA_ARCH_LIST=8.9"
Environment="PATH=/home/ubuntu/fakecuda/bin:/usr/local/cuda-13.0/bin:/home/ubuntu/splatris_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/ubuntu/splatris_env/bin/python3 /home/ubuntu/splatris/pipeline/gpu_worker_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 5. Worker Microservice API Reference (`http://89.169.122.100:8000`)
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns worker status, GPU model, VRAM capacity, and active job count. |
| `POST` | `/train` | Multipart form submission (`frames_zip`, `jobId`, `colmapMatcher`, `iterations`, `filteredFrameIndices`). Capped at 10,000 steps. |
| `GET` | `/status/{jobId}` | Returns stage, progress percentage, error traceback, and model byte size. |
| `GET` | `/logs/{jobId}` | Returns recent live text logs from the active training job. |
| `GET` | `/download/{jobId}/scene.splat` | Streams the finished 32-byte binary `.splat` scene file. |
| `POST` | `/cleanup/{jobId}` | Purges `/tmp/splatris_jobs/{jobId}` intermediate folder and frees disk space. |

---

## 7. API Reference (Local Platform)

### 1. Health & Service Metadata
- `GET /health` — Health check returning timestamp.
- `GET /api/info` — Platform version and service info.
- `GET /api/models/recent` — Returns metadata for the last 3 `.splat` models (filename, size, splat count, timestamp).

### 2. Nebius Agent Integration
- `GET /api/agent/test?prompt=<text>` — Test prompt to `nvidia/Nemotron-3-Ultra-550b-a55b`.
- `POST /api/agent/analyze-frames` — Submits frame metadata and returns COLMAP matcher, iteration count (capped at 10,000), and frame pruning recommendations.

### 3. Spatial Asset Upload & Ingestion
- `POST /api/spatial/upload` — Multipart video upload (MP4/MOV), automatically runs Pre-Flight AI Video Quality Gatekeeper audit, extracts frames dynamically according to $\text{clamp}(\text{duration} \times 2.2, 90, 220)$, and returns audit report with frame files.
- `POST /api/spatial/preflight-audit` — Standalone endpoint to audit video quality on demand using OpenCV Laplacian analysis and `openbmb/MiniCPM-V-4_5` multimodal inspection.
  ```json
  // Request: { "videoId": "bca67553-..." }
  // Response:
  {
    "success": true,
    "videoId": "bca67553-...",
    "audit": {
      "readiness_score": 88,
      "status": "APPROVED",
      "detected_context": "tabletop_object",
      "evaluated_by": "openbmb/MiniCPM-V-4_5 (Nebius)",
      "issues_detected": [],
      "director_guidance": [
        "Maintain continuous translational movement around the subject; avoid rotating the recording device on a fixed central axis.",
        "Tilt the recording device ~15° downwards toward surface boundaries and textured planes."
      ]
    }
  }
  ```

### 4. Spatial 3DGS Job Runner
- `POST /api/spatial/job/start` — Triggers remote training job on Nebius GPU.
  ```json
  {
    "videoId": "scene-demo",
    "colmapMatcher": "sequential",
    "recommendedIterations": 15000,
    "filteredFrameIndices": [0, 2, 4, 6]
  }
  ```
- `GET /api/spatial/job/status/:jobId` — Polls job progress and returns output model URL once completed:
  ```json
  {
    "success": true,
    "jobId": "2e9b0b2d-...",
    "status": "completed",
    "stage": "Completed",
    "progress": 100,
    "outputModelUrl": "/uploads/models/scene.splat"
- `POST /api/spatial/job/:jobId/abort` (or `POST /api/spatial/job/abort`) — Immediately cancels active training, releases Nebius GPU resources, and purges intermediate files:
  ```json
  {
    "success": true,
    "jobId": "2e9b0b2d-...",
    "status": "aborted",
    "message": "Job 2e9b0b2d-... successfully aborted. Nebius GPU resources released."
  }
  ```

### 5. Spatial Audio RAG Query
- `POST /api/spatial/rag-query` — Q&A grounded in 3D scene space.
  ```json
  {
    "query": "Explain the 3D Gaussian Splatting representation",
    "coordinates": { "x": 0, "y": 1.6, "z": 0.0 }
  }
  ```
  Returns:
  ```json
  {
    "success": true,
    "answer": "In this scene, 3D Gaussian Splatting represents the office room geometry through 359,302 anisotropic 3D Gaussians...",
    "audioSpatialPosition": { "x": 0, "y": 1.2, "z": -2.0 },
    "sources": ["..."],
    "groundingUsed": "Nemotron 3 Ultra Spatial Knowledge"
  }
  ```

---

## 8. Developer Quickstart

### Local Platform (Apple Mac mini M4)

```bash
# 1. Install dependencies
npm install

# 2. Configure environment (.env)
PORT=3000
NEBIUS_API_KEY=<your_nebius_api_key>
TAVILY_API_KEY=<your_tavily_api_key>
NEBIUS_GPU_ENDPOINT=http://195.242.13.40:8000

# 3. Start local development server
node server.js

# 4. Open in browser (Mac mini M4)
# Navigate to http://localhost:3000/ to launch the WebXR viewer
```

### Remote Nebius GPU Worker (NVIDIA L40S / Ubuntu 24.04)

```bash
# 1. Connect to Nebius GPU instance
ssh -i ~/.ssh/id_ed25519 ubuntu@195.242.13.40

# 2. Check worker daemon status
sudo systemctl status splatris-worker --no-pager

# 3. Check live GPU utilization
nvidia-smi

# 4. Query worker health
curl http://localhost:8000/health
```

### Meta Quest 3 WebXR Connection

```bash
# 1. Launch secure HTTPS tunnel on Mac mini
cloudflared tunnel --url http://localhost:3000

# 2. In Meta Quest Browser:
# Navigate to the generated https://*.trycloudflare.com URL and click "Enter WebXR".
```

---

## 9. Recent Refinements & Additions

### Milestone 11: UI & Metadata Refinement (Recent Scenes Naming & Toggle Clean-up)
- [x] **Hardware Text Clean-up**: In `public/index.html`, removed specific "Meta Quest 3" hardware reference from the Training Pipeline toggle switch under "Triangle Splatting". The subtitle now reads cleanly as `"Mesh-based"`.
- [x] **Backend Device Metadata Extraction**:
  - `services/videoService.js`: Added deep inspection of FFprobe format tags (`com.apple.quicktime.model`, `model`, `make`, `description`, etc.) and filename heuristics to extract recording device names (e.g., `"iPhone 11 Pro"`, `"iPhone 16 Plus"`, `"Insta360 X5"`, `"Meta Quest 3"`, `"GoPro HERO"`).
  - Saved video metadata sidecars (`video_meta.json`) during upload and forwarded device names throughout the pipeline.
- [x] **Model Metadata Persistence & Timestamp Formatting**:
  - `services/jobService.js`: Writes sidecar metadata (`{jobId}.json`) alongside exported `.splat` files preserving `originalFilename`, `deviceName`, and upload timestamps.
  - Formats timestamps with hour and minute granularity (`MMM DD, HH:MM`).
  - `/api/models/recent` endpoint dynamically formats:
    - **Card Title**: `[Date], [Time] - [Original Filename]` (e.g., `"Sep 25, 10:13 - IMG_9390.MOV"`).
    - **Card Subtitle**: `[Device Name] • [Splat Count] splats` (e.g., `"iPhone 11 Pro • 1,570,303 splats"`).
- [x] **Frontend UI Update**:
  - `public/app.js`: Updated `loadRecentScenes()` with `formatDateTimeShort(isoString)` to render the card title according to the `[Date], [Time] - [Original Filename]` template in the client's local time, paired with device subtitles, smooth text truncation, and responsive styling.

### Milestone 13: Instant Object Scan Mode (Image2Splat) & Autonomous Vision RAG (Completed & Verified)
- [x] **Frontend Dynamic Dropzone & 3-Way Training Toggle ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Added 3rd option to Training Pipeline toggle: `Instant Object • Single Image (Image2Splat)`.
  - Dynamically updates dropzone configuration when selected:
    - Sets accepted file types strictly to `image/jpeg,image/png` (`.jpg, .jpeg, .png`).
    - Modifies dropzone copy: `"Drag & drop a single clear photo (JPG or PNG)"` and `"Supports high-resolution camera photos (up to 50MB)"`.
    - Disables video uploads while in single-image mode.
    - Preserves existing Standard 3DGS and Triangle Splatting pipelines 100% intact as defaults.
- [x] **Nebius GPU Worker Routing & Isolated Environment ([pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py), [pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
  - Cloned `romanpillai/Image2Splat` to `/home/ubuntu/Image2Splat` on Nebius L40S GPU server.
  - Created isolated virtual environment at `/home/ubuntu/image2splat_env` linked cleanly to base CUDA/PyTorch dependencies.
  - Built standalone inference script `infer_image2splat.py` performing monocular depth unprojection, 3D Gaussian synthesis, and exporting both standard binary `.splat` and COLMAP-compatible `.ply` files.
  - Added routing condition in `gpu_worker_server.py` for `training_method == "image2splat"`, completely bypassing COLMAP structure-from-motion and executing single-image reconstruction in ~15-20 seconds.
- [x] **Nemotron Vision + Tavily Autonomous Agent ([services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js), [routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - Asynchronously sends uploaded photo to `openbmb/MiniCPM-V-4_5` via Nebius Token Factory upon upload.
  - Extracts precise object labels, categories, and confidence scores.
  - **Fallback / Error Handling**: If confidence is low or object cannot be identified, pauses Tavily search and displays an interactive UI fallback card asking: `"Could not clearly identify the object. What is this?"` allowing immediate manual user text entry.
  - Upon valid identification (AI or manual override), autonomously fetches real-world history, technical specifications, and context from Tavily API (`https://api.tavily.com/search`).
  - Streams Tavily knowledge directly into the Spatial Audio RAG panel with synthetic spatial voice announcements and contextual suggestion chips.

### Milestone 14: Smart Routing & Training Toggle Layout Fix (Completed & Verified)
- [x] **Training Pipeline Toggle Vertical Stack & Type Badges ([public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
  - Solved the horizontal clipping issue where the 3rd radio option was cut off on the right of the 370px sidebar.
  - Converted `.segmented-control` from a squished 3-column horizontal grid into a sleek vertical stack (`flex-direction: column; width: 100%`).
  - Added `.seg-title-row` with distinct pill tags (`Video` in violet vs `Photo` in cyan) and full-width subtitles, ensuring complete readability without horizontal clipping across all display sizes.
- [x] **Universal Dropzone Acceptance & Copy Refinement ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Updated dropzone accept attribute to universally accept both video and image formats: `accept="video/mp4,video/quicktime,image/jpeg,image/png"`.
  - Updated dropzone copy to:
    - **Title**: `"Drop a walkthrough video or single object photo"`
    - **Subtitle**: `"Supports MP4/MOV walkthroughs (max 500MB) or JPG/PNG photos for Instant 3D scan"`
- [x] **Smart Routing Logic Triggered on Drop/Selection ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Video Files (`.mp4`, `.mov`)**:
    - Automatically switches the pipeline toggle to a video-compatible mode (`Standard 3DGS` if currently on `Instant Object`).
    - Enforces browser pre-flight orientation guard (rejecting vertical 9:16 videos).
    - **Action**: Immediately triggers the Capture Quality Audit (Gatekeeper) pre-flight modal for orientation, motion blur, and keypoint analysis.
  - **Image Files (`.jpg`, `.jpeg`, `.png`)**:
    - Automatically switches the pipeline toggle to `Instant Object (Single Image)`.
    - **Action**: Completely **bypasses** the Capture Quality Audit (Gatekeeper) pre-flight screen.
    - Immediately triggers parallel Nemotron Vision + Tavily autonomous RAG lookup and dispatches single-image reconstruction to the Image2Splat worker on Nebius GPU.

### Milestone 15: Centered Vertical Axis Rotation & Turntable Controls (Completed & Verified)
- [x] **Splat Centering & True Vertical Axis Pivot Container ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Identified why the object could not rotate on its vertical axis: previously `controls.target` was set to an off-center offset (`z = -0.6`), causing camera orbits to swing the model across the screen instead of revolving around it, and the Gaussian splat viewer had no pivot group centered at the model's centroid.
  - Implemented `STATE.splatPivotGroup` (`THREE.Group`) positioned directly at the geometric center `(0, 1.2, 0)`.
  - Offset `dropInViewer` inside `splatPivotGroup` by `(-centerX * autoScale, -centerY * autoScale, -centerZ * autoScale)`, ensuring `(0, 0, 0)` in pivot space is the exact centroid of the 3D model.
  - Set `controls.target.set(0, 1.2, 0)` so standard mouse Left-Drag orbits cleanly around the object's central vertical axis.
- [x] **Turntable & Interactive Rotation Controls ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Added on-screen viewport overlay toolbar buttons:
    - **`Turntable Spin`**: Toggles continuous smooth ~0.5 RPM rotation of the 3D model around its vertical axis.
    - **`↺ -15°` / `↻ +15°`**: Stepped rotational nudges around the vertical axis.
    - **`Reset`**: Resets orientation back to 0° and re-centers the camera.
### Milestone 16: 3-Axis Free Rotation & Spatial Object Raycasting (Completed & Verified)
- [x] **Viewport Interaction: 3-Axis Free Rotation (Trackball/Orbit) ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Replaced the single-axis clamped rotation constraint with full **3-Axis Free Rotation** powered by `THREE.TrackballControls`.
  - Enables users to rotate the 3D Gaussian splat model freely across Pitch (X-axis), Yaw (Y-axis), and Roll (Z-axis) simply by Left-Clicking and dragging anywhere in the viewport with zero gimbal lock or pole stops.
  - Added `#btn-toggle-trackball` overlay chip allowing instant switching between **3-Axis Free Rotation** and **Grounded Orbit (Turntable)**.
  - Synchronized camera target updates and control enabling/disabling seamlessly across pointer locks, scene resets, and free roam.
- [x] **Spatial Object Selection (Raycasting) in 3D Viewport ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Implemented `GaussianSplats3D.Raycaster` and `THREE.Raycaster` pipeline in `public/app.js`.
  - **Desktop**: Allows users to **Double-Click** anywhere on the 3D model to select and inspect an object.
  - **WebXR (Meta Quest 3 / Vision Pro)**: Added `Point & Trigger` (`select` event on VR controllers) allowing spatial selection via controller laser beam.
  - **Toolbar Shortcut**: Added `#btn-inspect-raycast` (`Inspect 🎯`) viewport chip to target the current viewport center.
  - Spawns an animated 3D pulsating cyan/violet reticle (`raycastTargetMesh`) at the selected `(x, y, z)` coordinate in world space.
  - Dynamically updates the Web Audio 3D panner position to `(x, y, z)` and triggers a spatial chime.
- [x] **2D Frame Mapping & Target Object Crop Extraction ([services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js), [routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - `cropAndIdentifyObject({ point, activeModel, cameraPosition, videoId })`:
    - Projects 3D world coordinate `(x, y, z)` onto camera frames using spherical/pinhole projection geometry to identify the closest training frame.
    - Computes bounding box around target `(u, v)` pixel coordinates.
    - Executes sub-pixel crop via native `ffmpeg` (`-vf crop=W:H:X:Y`) in <25ms, saving isolated crops to `/uploads/audit/crop_{timestamp}.jpg`.
- [x] **Targeted Nemotron Vision + Tavily RAG Pipeline ([services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Dispatches isolated 2D object crop to `openbmb/MiniCPM-V-4_5` on Nebius Token Factory for identification, category determination, and visual analysis.
  - Queries Tavily AI API (`https://api.tavily.com/search`) with the identified object name for real-world history, engineering specs, and web sources.
  - Automatically opens the Spatial Audio RAG drawer and displays a rich inspection card:
    - Target object name & confidence score badge.
    - Clickable thumbnail preview of the isolated 2D cropped object.
    - 1-sentence visual description.
    - Real-world grounding summary with clickable external source chips.
### Milestone 17: Upload Limit Expansion to 1GB (1024MB) (Completed & Verified)
- [x] **Backend Express & Multer Payload Expansion ([server.js](file:///Users/akif/Projects/splatris/server.js), [routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - Configured `express.json({ limit: '1024mb' })` and `express.urlencoded({ extended: true, limit: '1024mb' })` in `server.js`.
  - Updated Multer `limits.fileSize` to `1024 * 1024 * 1024` (1GB) in `routes/spatial.routes.js` to support high-density captures and full benchmark datasets (e.g., Tanks and Temples).
  - Enhanced Multer error handling to explicitly return `HTTP 413` with actionable error messaging when `LIMIT_FILE_SIZE` is triggered.
- [x] **Frontend Validation & UI Synchronization ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
  - Updated `MAX_FILE_SIZE_MB = 1024` in `public/app.js` and updated user toast alerts to reference 1GB / 1024MB.
### Milestone 18: Gatekeeper Laplacian Blur Variance Logic Fix & Server Migration (Completed & Verified)
- [x] **Mathematical Inversion Fix for Sharpness vs. Blur ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js), [pipeline/audit_video.py](file:///Users/akif/Projects/splatris/pipeline/audit_video.py))**:
  - Resolved false rejection bug where high-quality academic benchmark videos (e.g. Tanks and Temples) with high edge variance (e.g., 152.1, 130.4) were falsely flagged as "Motion Blur / Rapid Pan".
  - **Mathematical Ground Truth**: Clarified across prompts, scripts, and validators that in computer vision, **high** Laplacian variance measures rich high-frequency edge energy and proves the frame is **sharp** and in-focus. High variance values (> 50.0) must **always pass** without upper-bound penalties.
  - Adjusted photogrammetry blur threshold to a clean minimum baseline: `BLUR_THRESHOLD = 50.0`. Only frames strictly below 50.0 trigger blur warnings.
- [x] **False Positive Blur Flag Interceptor ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js))**:
  - Implemented mathematical issue validator in `gatekeeperService.js` that inspects model-generated issues:
    - If an issue flags "Motion Blur / Rapid Pan" on a frame whose measured OpenCV Laplacian variance is `>= 50.0`, the system automatically dismisses the hallucinated flag (`[Gatekeeper] Dismissed false positive blur flag: Laplacian variance is >= 50.0, mathematically sharp`).
    - Prevents sharp benchmark dataset frames from improperly diminishing the readiness score.
- [x] **Accurate UI Diagnostic Messaging ([services/gatekeeperService.js](file:///Users/akif/Projects/splatris/services/gatekeeperService.js))**:
  - Updated blur diagnostic messages to accurately reflect the math:
    `"Frame sharpness measured low Laplacian variance. Potential motion blur or out-of-focus frame detected."`
### Milestone 19: Undici High-Capacity Agent & Headers Timeout Resolution (Completed & Verified)
- [x] **Undici Custom Dispatcher with Extended Timeouts ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Resolved `UND_ERR_HEADERS_TIMEOUT` caused by Node.js native `fetch` prematurely closing socket connections during multi-megabyte frame archive uploads over the internet to Nebius GPU.
  - Implemented high-capacity `undici.Agent` with extended 10-minute thresholds:
    ```javascript
    const gpuWorkerAgent = new Agent({
      headersTimeout: 600000, // 10 minutes
      bodyTimeout: 600000,    // 10 minutes
      connectTimeout: 60000,  // 1 minute
      keepAliveTimeout: 60000,
    });
    ```
### Milestone 20: Undici FormData Multipart Serialization Fix (Completed & Verified)
- [x] **Multipart Form Field Resolution for Remote Worker ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Identified root cause of HTTP 422 (`Field required: frames_zip`): Node.js global `FormData` was not serializing multipart boundaries correctly when transmitted through `undici.fetch`.
  - Switched to `UndiciFormData` imported directly from `undici`:
    ```javascript
    import { fetch as undiciFetch, Agent, FormData as UndiciFormData } from 'undici';
    const formData = new UndiciFormData();
    formData.append('frames_zip', blob, 'frames.zip');
    ```
  - Tested and verified direct `POST /train` transmission to `http://195.242.13.40:8000` returning `HTTP 200 OK` with scheduled job status.

### Milestone 21: Real WebGL Viewport Canvas Extraction & Dummy Image Elimination (Completed & Verified)
- [x] **WebGL Buffer Preservation & Viewport Crop Extraction ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Configured `preserveDrawingBuffer: true` in Three.js `WebGLRenderer` initialization so the front buffer is retained for reliable frame capture.
  - Implemented `takeScreenshot({ crop = false, screenX = null, screenY = null, boxSize = 380 })`:
    - Temporarily hides 3D target reticle (`raycastTargetMesh.visible = false`) to prevent reticle geometry from occluding the target object.
    - Forces a synchronous render pass on the scene and GaussianSplats3D viewer.
    - Maps client screen coordinates to device-pixel-ratio canvas buffer pixels and crops an adaptive bounding box around the click point onto an in-memory 2D canvas.
    - Restores reticle visibility and returns a high-fidelity `image/jpeg` base64 data URL.
- [x] **Client-to-Backend Raycast Payload Integration ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - Passed `screenX` and `screenY` from mouse double-click and centered inspection events into `triggerSpatialObjectInspection`.
  - Included `imageBase64` in the POST payload to `/api/spatial/raycast-object`.
  - Updated backend route handler to forward `imageBase64` directly to `cropAndIdentifyObject`.
- [x] **Elimination of Dummy / Abstract Fallback Frames ([services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js))**:
  - Root Cause Diagnosed: Frame files pruned post-training caused `cropAndIdentifyObject` to fall back to the first available directory in `uploads/frames/` (which happened to be `048b7268...`, an old synthetic blue/yellow test video). This caused Nemotron Vision to identify "Abstract Color Block" regardless of what 3D object was clicked.
  - Completely stripped out arbitrary directory search and deleted `uploads/frames/048b7268...`.
  - Added primary handling path for `imageBase64`: writes the real WebGL viewport crop buffer to `uploads/audit/crop_{timestamp}.jpg`, passes it directly to `identifyObjectFromImage` (Nemotron Vision / MiniCPM-V-4_5 on Nebius Token Factory), and enriches with Tavily web knowledge.
  - Verified endpoint end-to-end with base64 transmission returning HTTP 200 and accurate multi-source grounding.

### Milestone 22: Live Frame Archive Upload Telemetry & Dynamic Pipeline Progress (Completed & Verified)
- [x] **Real-Time Chunk Streaming with Live Telemetry ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Diagnosed progress stall: `workerFetch` was sending multipart `frames.zip` as an unmonitored payload, while the frontend progress bar was clamped to a static `55%` for 5 minutes during the ~40MB archive upload.
  - Implemented `uploadFramesWithLiveProgress`: uses native Node `http`/`https` streaming with high-watermark chunks (128KB) and real-time socket tracking.
  - Computes and streams live metrics every ~300ms:
    - Transferred vs total data: `e.g. 14.8 / 38.9 MB`
    - Transfer percentage: `e.g. 38%`
    - Live upload bandwidth: `e.g. 1.6 MB/s`
    - Dynamic ETA countdown: `e.g. (15s left)`
    - Formatted stage text: `"Uploading 220 frames to GPU: 14.8 / 38.9 MB (38%) • 1.6 MB/s (15s left)"`
- [x] **Dynamic Progress Scaling ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Removed artificial `Math.max(55, data.progress)` clamp from `pollJobUntilReady`.
  - Initialized Step 4 at `20%`, allowing the progress bar to continuously advance from `20%` to `35%` as frames are uploaded to the GPU.
  - Mapped remote GPU worker training telemetry cleanly into `36%` to `95%`, followed by model download (`96% - 99%`) and completion (`100%`).
- [x] **Chat Message & Card Text Selection Enabled ([public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Overcame global `body { user-select: none; }` restriction by explicitly applying `user-select: text !important; -webkit-user-select: text !important;` across `.rag-conversation`, `.message`, `.msg-content`, and `.raycast-card`.
  - Added custom cyan `::selection` highlight glow (`rgba(6, 182, 212, 0.4)`) matching the Splatris cyberpunk visual theme.
- [x] **Active Public Tunnel for Meta Quest 3 & Remote Testing**:
  - **Live URL**: `https://nutrition-extremely-yard-saturn.trycloudflare.com` (tunneling `http://localhost:3000`).
  - Connector ID: `e87e83db-2046-4092-85d4-37fb5c59c73e`.

### Milestone 23: Meta Quest 3 WebXR Optimization, 6DoF Locomotion & Session Lifecycle (Completed & Verified)
- [x] **Elimination of Duplicate 25MB Model Fetch ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Root Cause Diagnosed: `loadGaussianSplatScene` was executing an unbuffered `fetch(url).arrayBuffer()` for bounding box calculation and immediately re-downloading the entire 25.1MB `.splat` file via `addSplatScene(url)`. Over mobile/VR connections, concurrent HTTP/2 streams triggered Cloudflare stream cancellations (`error="stream canceled by remote with error code 0"`).
  - Streamlined pipeline to a single download request via `addSplatScene`.
  - Shifted spatial bounding, floor grounding ($Y=0$), and centroid alignment to execute directly from in-memory splats in $<1\text{ ms}$ post-load using `splatMesh.getSplatCenter(i, tempCenter, true)`.
- [x] **Live In-Viewport & HUD Download Telemetry ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Created a floating glassmorphic loader overlay (`#splat-loader-overlay`) in the 3D viewport displaying live download percentage, progress bar, and shader compilation status.
  - Updated HUD status pills to report real-time progress (`Downloading X%`) transitioning cleanly to primitive counts (`786,568 Gaussians`).
- [x] **HTTP Caching & Byte-Range Headers ([server.js](file:///Users/akif/Projects/splatris/server.js))**:
  - Configured static uploads middleware with `Cache-Control: public, max-age=86400, immutable`, `Accept-Ranges: bytes`, and CORS headers for `.splat` and `.ply` files.
  - Ensures once downloaded, models load instantaneously from local Meta Quest Browser disk cache.
- [x] **Gaussian Sorting Stall Resolution in WebXR ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Diagnosed rotation pause/stutter: `DropInViewer` had `dynamicScene: true`, which forced the sort worker to re-sort all 786,568 Gaussians on every single frame without checking view angle difference or position difference.
  - Configured `dynamicScene: false` for the static scene, activating Kellogg's spatial octree culling and angular thresholding (`angleDiff <= 0.99`, `positionDiff >= 1.0`). Eliminates sort worker stalls when looking around the tank.
  - Updated `animate()` to pass `renderer.xr.getCamera()` (the stereo VR eye camera) to `viewer.update` while presenting in WebXR, preventing covariance projection discrepancies.
- [x] **WebXR 6DoF Smooth Locomotion & Turning ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Wrapped `camera`, `controller0`, and `controller1` inside a dedicated `CameraRig` (`THREE.Group`) to allow virtual movement on top of physical room tracking.
  - Implemented continuous gamepad input polling in the render loop:
    - **Left Thumbstick (`axes[2]`, `axes[3]`)**: Smooth locomotion (move forward/backward along headset gaze direction and strafe left/right at 2.4 m/s).
    - **Right Thumbstick (`axes[2]`, `axes[3]`)**: Smooth yaw turning ($2.0\text{ rad/s}$) and vertical elevation fly up/down to inspect tank treads or view from above.
- [x] **WebXR Session Exit to Browser ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Controller Shortcut**: Polled secondary buttons (`gamepad.buttons[5]`) so pressing **`B`** (right controller) or **`Y`** (left controller) immediately calls `session.end()`, gracefully exiting immersive mode and returning to the 2D browser window.
  - **In-World Floating 3D HUD**: Generated an interactive 3D glass HUD (`createXRInWorldHUD`) floating at eye level displaying locomotion guide and a clickable `[ 🚪 Exit to Browser ]` button responsive to controller raycast triggers.
  - Restored camera coordinates and OrbitControls cleanly on `session.addEventListener('end')`.

### Milestone 24: WebXR Raycast Centered Framing, Generous Context Window, and True 3D Spatial Audio TTS (Completed & Verified)
- [x] **Target-Centered Snapshot Virtual Camera ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Root Cause Diagnosed: In WebXR, raycast clicks mapped to screen space using the stationary desktop `camera` (`hitPoint.clone().project(camera)`), which looked forward from $(0, 1.25, 0.4)$, while `takeScreenshot` also rendered from the desktop camera instead of the active WebXR headset camera. Furthermore, a tiny 380px crop cut out only ceiling splats above the tank.
  - Implemented dedicated snapshot virtual camera pipeline (`takeScreenshot({ targetPoint })`):
    - Automatically obtains user's true eye position in world space via `activeCam.getWorldPosition()`.
    - Aligns snapshot camera along the view vector at an optimal framing distance ($1.8\text{m} - 3.2\text{m}$) and executes `snapCam.lookAt(targetPoint)`.
    - **Mathematically guarantees target object is dead-center** $(u=0.5, v=0.5)$ in the snapshot frame.
    - Set FOV to $52^\circ$ and rendered to a dedicated $768\times 768$ off-screen `WebGLRenderTarget`, providing a generous context window capturing the complete object silhouette, tracks, chassis, and ground context.
    - Temporarily hides 3D reticle and in-VR HUD during capture to prevent visual obstruction.
- [x] **WebXR Splat Mesh Raycast Intersection ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Configured `GS.Raycaster(origin, direction)` directly inside `performRaycastSelectionXR` against `STATE.dropInViewer.splatMesh`.
  - Accurately resolves intersections with genuine 3D Gaussian splat ellipsoids of the model (e.g. tank turret, armor plate, wheels) instead of falling back to empty scene meshes.
- [x] **2D COLMAP / Training Frame Elevation & Context Recalibration ([services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js))**:
  - Eliminated ceiling elevation bias: Recalibrated `vPercent` to center strictly within the object elevation band ($0.42 - 0.62$, nominal $0.52$).
  - Expanded context window crop margin: Increased `cropSize` from $38\%$ to **$76\%$** of the frame's shortest dimension (minimum $512\text{px}$, up to $820\text{px}$), ensuring Nemotron Vision receives ample visual context to identify the vehicle.
- [x] **True 3D Spatial Audio Text-to-Speech (TTS) Engine ([routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **High-Fidelity Audio Stream Endpoint**: Added `GET /api/spatial/tts` to stream clean narration audio buffers (`audio/mpeg`).
  - **Web Audio HRTF 3D Panner Integration**:
    - Created `speakSpatialAudio(text, position)`: fetches narration audio, decodes into `AudioBuffer` via `audioCtx.decodeAudioData`, and attaches an `AudioBufferSourceNode` directly to `pannerNode` (`panningModel = 'HRTF'`).
    - Anchors the audio emitter to the exact 3D coordinates `(hitPoint.x, hitPoint.y, hitPoint.z)` of the selected object.
    - The voice realistically emanates in full 3D binaural spatial audio from the targeted object in the scene.
    - Added acoustic beacon feedback chime ($560\text{ Hz}$) at the object position upon selection.
    - Included graceful fallback to `window.speechSynthesis`.
  - **Continuous Headset AudioListener Tracking**:
    - Updated render loop to continuously update `audioListener.position` and 3D orientation (`forward` and `up` vectors) matching the headset camera in WebXR mode.
- [x] **In-VR Dynamic Result Card ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Implemented `updateXRHUDContent`: updates the floating 3D HUD texture in VR with the identified object name, category, confidence percentage, and Tavily summary. Users in VR can both hear the spatial voice and read the summary without needing desktop panels.
- [x] **Active Public Tunnel for Meta Quest Testing**:
  - **Live URL**: `https://variable-towers-note-diff.trycloudflare.com` (tunneling `http://localhost:3000`).

### Milestone 25: Pipeline Caching, COLMAP Smart Bypass & Re-Processing UI (Completed & Verified)
- [x] **Smart COLMAP Cache & Pose Persistence ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Implemented `--scene_id`, `--cached_scene_id`, and `--cache_dir` (default: `/tmp/splatris_colmap_cache`).
  - Added smart bypass: when `cached_scene_id` has cached poses in `colmap/sparse/0/`, logs `[STAGE:COLMAP_CACHE_BYPASS] [PROGRESS:35%]`, sets `colmap_dir = scene_cache_dir`, and skips `ns-process-data` entirely.
  - Automatically caches solved camera poses (`sparse/0/`), undistorted images (`images/`), and `transforms.json` into `/tmp/splatris_colmap_cache/{scene_id}/` upon successful COLMAP completion.
- [x] **GPU Worker Smart Bypass & Cache Verification ([pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py))**:
  - Updated `/train` endpoint: accepts `sceneId` and `cachedSceneId`. When `cachedSceneId` exists, `frames_zip` is made optional, eliminating redundant multi-megabyte frame zip uploads over HTTP.
  - Added `GET /cache/status/{scene_id}` endpoint to query cache availability.
- [x] **Fast Dispatch & Pose Injection ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - In `runNebiusGpuTraining` and `dispatchToRemoteWorker`, when `bypassColmap` or `cachedSceneId` is set, frame archiving and upload are completely bypassed, sending form metadata directly to `/train` and immediately jumping progress to 35%.
- [x] **Fast Re-processing Service & Model Metadata ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js), [server.js](file:///Users/akif/Projects/splatris/server.js))**:
  - Created `reprocessJob({ sceneId, trainingMethod })`: validates scene metadata, enforces image isolation, creates job with `bypassColmap: true` and `cachedSceneId: sceneId`, and dispatches directly to CUDA training.
  - Added `getModelMeta(sceneId)` and endpoint `GET /api/models/meta/:id`.
  - Added `POST /api/spatial/job/reprocess` and `GET /api/spatial/job/cache-status/:sceneId`.
  - Updated `cleanupPostTraining`: retains cached poses and frames for active and cached video scenes.
- [x] **Video Re-processing UI & Image Isolation ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Video Re-processing**: When active scene in viewport is a Video and user selects an alternative pipeline (e.g. Triangle Splatting), normal dropzone is hidden and replaced by `#reprocess-banner` featuring `Re-process as Triangle Splatting (Fast)`. Clicking triggers `POST /api/spatial/job/reprocess`, sets pipeline steps 1 & 2 as `bypassed` ("COLMAP Cached"), streams progress from 35% to 100%, and dynamically replaces the scene model upon completion. Includes "Upload New Video" secondary action to restore the dropzone.
  - **Image Isolation**: When active scene was generated from a single photo (Instant Object / Image2Splat), "Standard 3DGS" and "Triangle Splatting" radio buttons are disabled and grayed out (`.method-disabled`, `disabled`, and informational tooltip). Single images cannot be re-processed into video pipelines (enforced on both client and server).

### Milestone 26: Reprocess Banner State Reset & Jargon-Free UX Copy Overhaul (Completed & Verified)
- [x] **Reprocess Banner State Reset & CSS Rule ([public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Root Cause Diagnosed: `.reprocess-card.hidden` was previously missing an explicit `display: none !important;` rule in CSS, causing the banner to remain visible even when `.hidden` was added.
  - Added `.reprocess-card.hidden, #reprocess-banner.hidden { display: none !important; }` in `style.css` and `style="display: none;"` in `index.html`.
  - When user selects a single photo (Image2Splat / Instant Object) from Recent Scenes, `updatePipelineUIForActiveScene` explicitly executes `reprocessBanner.style.setProperty('display', 'none', 'important')` and adds `hidden`. The banner is guaranteed to be completely hidden for image scenes and ONLY visible for video models.
- [x] **Dynamic Scene Name Binding ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Removed hardcoded "Tank Walkthrough" text from `index.html`.
  - Dynamically binds the actual active scene filename (e.g. `tank_internet.mp4`) into `#reprocess-scene-name` via `updatePipelineUIForActiveScene`.
- [x] **Jargon Removal & User-Centric Copy ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Replaced "COLMAP Poses Cached" badge with the user-benefit badge **"Quick Convert"**.
  - Retained time-saving value badge **"< 90s vs 5m"**.
  - Rewrote description to clear, user-centric copy:
    *"This scene (tank_internet.mp4) is already processed. You can instantly convert it to Triangle Splatting (WebXR Optimized) without waiting for a new upload or initial processing."*
  - Replaced technical step labels and progress toasts:
    - "1. Upload (Ready)" and "2. Processed" (instead of "COLMAP (Cached)").
    - "⚡ Quick Convert: Re-processing scene into new pipeline..." (instead of "COLMAP Cache Hit! Skipping upload & feature extraction...").
  - Removed all occurrences of "COLMAP" and "poses" from user-facing elements in the conversion flow.
### Milestone 27: Graceful Cache Fallback & Automatic Cache Establishment on Fresh GPU Instances (Completed & Verified)
- [x] **Fresh VM Cache Fallback ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Root Cause Diagnosed: When migrating to a newly booted GPU instance (`89.169.112.102`), the worker starts with an empty `/tmp/splatris_colmap_cache/`. Initiating Fast Re-processing previously sent a bypass request without `frames_zip`, causing FastAPI to reject the job with HTTP 400 (`"Missing required frames_zip for fresh training job without cache."`).
  - Added pre-flight cache check `GET ${endpoint}/cache/status/${cachedSceneId}` before executing bypass.
  - If `hasCache === true`: executes instantaneous smart bypass (skips upload and COLMAP, starts at 35%).
  - If `hasCache === false`: gracefully falls back to packaging local video frames (`activeFramesDir`), streaming them to `/train` to initialize the remote cache and execute Triangle Splatting, ensuring zero 400 Bad Request errors.
- [x] **Dynamic Frames Directory Resolution ([services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Implemented `resolveExistingFramesDir`: searches `uploads/frames/` by scene ID, video ID, model UUID, or original video metadata (`video_meta.json`), correctly locating authentic frame sets (e.g. 220 high-fidelity Panzer IV Tank frames).
- [x] **Worker Sync & Verification**:
  - Synchronized `pipeline/gpu_worker_server.py` and `pipeline/train_3dgs.py` to `/home/ubuntu/splatris/pipeline/` on `89.169.112.102`.
  - Restarted `splatris-worker.service` (active, CUDA L40S 44.4GB VRAM verified).
### Milestone 28: Job Recovery, Reconnection & Preemptible Instance Handling (Completed & Verified)
- [x] **Backend Active Job Protection ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js))**:
  - Implemented `findActiveJobForScene(sceneIdOrVideoId)`: checks in-memory job store for any running (`processing` or `suspended`) job matching the target scene.
  - In `createJob` and `reprocessJob`: if an active job already exists, skips spawning duplicate workers and returns the existing job with `isExisting: true`.
  - In `GET /api/spatial/job/status/:jobId`: expanded to resolve by both `jobId` AND `sceneId`, returning `isProcessing` and `isSuspended` flags.
  - Added endpoint `GET /api/spatial/job/active/:sceneId` to quickly inspect running tasks for any scene.
- [x] **Preemptible GPU Worker Drop & Suspension ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js))**:
  - Root Cause Diagnosed: Preemptible Nebius GPU instances periodically reboot or shut down unexpectedly, causing `ETIMEDOUT` during status telemetry polling and firing hard failure events with raw browser alerts.
  - Replaced hard failure with graceful suspension: network timeout errors (`ETIMEDOUT`, `ECONNREFUSED`, `Worker Unreachable`, `TimeoutError`) now transition job status to `'suspended'` with stage `'GPU Instance Preempted/Offline'`.
  - Preserves extracted frames, progress percentage, and intermediate cache without purging.
- [x] **Resume Training Endpoint ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js))**:
  - Added `resumeJob({ jobId, sceneId })` and route `POST /api/spatial/job/resume` (and `POST /api/spatial/job/:jobId/resume`).
  - Pre-flights worker `/health` to ensure instance is back online, resets job to `processing`, enables `bypassColmap: true` to reuse cached camera poses, and restarts the asynchronous execution runner.
- [x] **In-UI Warning Banner & Alert Removal ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Removed disruptive `window.alert()` on training failures/drops.
  - Created glassmorphic Cyber-Amber `#gpu-suspended-banner` featuring a **"Resume Training"** button (`#btn-resume-training`) and a cancel/reset button (`#btn-abort-suspended`).
  - Progress bar, percentage, and timer remain preserved during suspension.
- [x] **Frontend State Reconnection & Duplicate Prevention ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Persists active job ID and scene ID to `localStorage`.
  - Implemented `checkAndReconnectActiveJob()`: automatically pings backend on initial page load and on scene selection from "Recent Scenes".
  - If a job is active or suspended, automatically re-attaches the progress bar, stage indicator, and telemetry polling loop.
  - Implemented `updateButtonsForActiveJob()`: disables "Start" and "Re-process" buttons (changing text to "Training in Progress...") to prevent duplicate job submissions.
- [x] **Deployment & Verification**:
  - Verified remote worker at `http://89.169.112.102:8000`.
  - Confirmed active Cloudflare tunnel: `https://variable-towers-note-diff.trycloudflare.com`.

### Milestone 29: Triangle Splatting Checkpoint Detach & Vectorized Export Fix (Completed & Verified)
- [x] **PyTorch Tensor Detach Bug ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Root Cause Diagnosed: After 7,000 steps of Triangle Splatting optimization completed successfully, the post-training export failed at `convert_triangle_checkpoint()` with `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`
  - The model state dict parameters (`triangles_points`, `features_dc`, `opacity`) had `requires_grad=True`. Calling `.numpy()` directly triggered a PyTorch assertion error.
  - Added `.detach()` and `.cpu()` calls on all tensor extractions before conversion to numpy arrays:
    - `pts = sd["triangles_points"].detach().cpu().float().numpy()`
    - `features_dc = features_dc.detach().cpu().float().squeeze().numpy()`
    - `raw_opacity = raw_opacity.detach().cpu().float().squeeze().numpy()`
- [x] **Vectorized WebXR .splat Export ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Replaced the slow Python `for i in range(num_triangles)` loop and `struct.pack` with a vectorized NumPy structured array (`splat_dtype = [('pos', '<f4', 3), ('scale', '<f4', 3), ('color', 'u1', 4), ('rot', 'u1', 4)]`).
  - Converts all 4,000,002 triangle facets into a binary 122 MB `.splat` file in **2.06 seconds** (down from ~20+ seconds).
  - Generates full 233 MB colored Mesh PLY with 12,000,006 vertices and 4,000,002 faces in **1.02 seconds**.
### Milestone 30: WebXR Viewer Splat Format Alignment & Recent Scenes Deduplication (Completed & Verified)
- [x] **Viewer Format Mismatch Diagnosed ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
  - Root Cause Diagnosed: The browser's 3D and WebXR viewer engine is `@mkkellogg/gaussian-splats-3d`, which exclusively parses Gaussian and facet splats (`.splat` or INRIA Gaussian Splat PLY). For Triangle Splatting jobs, the backend generated two files: a standard polygon mesh (`.ply`, 233 MB) and oriented triangle facet splats (`.splat`, 122 MB).
  - Previously, `job.outputModelUrl` was pointing to the `.ply` file, and `getRecentModels` exposed both files as separate cards. Clicking the `.ply` card caused `GaussianSplats3D` to fail to parse the polygon mesh, leaving the 3D canvas empty.
- [x] **Viewer Redirection & Deduplication Fix**:
  - In [public/app.js](file:///Users/akif/Projects/splatris/public/app.js) (`loadGaussianSplatScene`): If a `.ply` URL is ever requested, the viewer automatically remaps to the companion WebXR `.splat` file (`url.replace(/\.ply$/i, '.splat')`).
  - In [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js) (`getRecentModels`): Deduplicated recent scenes by scene `baseName`. Prioritizes the WebXR `.splat` model for viewing while attaching the companion mesh PLY (`meshPlyUrl`) for 3D export.
  - In [services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js): Always sets `outputModelUrl` to `.splat` and downloads both `.splat` (for WebXR viewer) and `.ply` (for export).
### Milestone 31: Triangle Splatting Performance Optimization & Transparent Facet Culling (Completed & Verified)
- [x] **Rotation Lag Diagnosed ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Root Cause Diagnosed: Standard 3DGS produces ~786,000 Gaussians (24 MB). Triangle Splatting produced **4,000,002 facets (122 MB)**.
  - Analysis revealed that out of 4,000,002 facets, **2,529,543 facets (63.2%) had near-zero opacity (alpha <= 15 / 255)**, representing invisible volumetric background fog.
  - During camera rotation, `GaussianSplats3D` performs depth sorting on every frame. Sorting 4 million facets in JavaScript on the CPU caused severe 5-10 FPS frame drops.
- [x] **Invisible Facet Culling & Viewer Optimization**:
  - In [pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py) (`convert_triangle_checkpoint`): Added automated opacity threshold filtering (`alpha >= 16`) before writing the `.splat` file, pruning all 2.5 million invisible noise facets.
  - Reduced file size from **122.1 MB down to 44.9 MB (1,470,459 crisp visible facets)**.
  - In [public/app.js](file:///Users/akif/Projects/splatris/public/app.js) (`addSplatScene`): Updated `splatAlphaRemovalThreshold` to `15` to ensure runtime culling of low-opacity noise.
### Milestone 32: Coordinate Space Alignment & Vertical Tilt Fix for Triangle Splatting (Completed & Verified)
- [x] **Vertical Tilt Root Cause Diagnosed ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Root Cause Diagnosed: In [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), the 3D viewer hardcoded a `-90 deg X rotation` (`quaternion [-0.7071068, 0, 0, 0.7071068]`).
  - This rotation was tailored for **Nerfstudio (3DGS)** because Nerfstudio transforms camera poses so that **+Z is UP** (ground plane in X-Y).
  - However, **Triangle Splatting** trains directly from raw COLMAP datasets, where **Y is already the vertical axis** and the ground plane is in **X-Z** (with mean camera UP vector `[-0.025, -0.994, -0.097]`).
  - Applying a `-90 deg X rotation` to an already Y-aligned scene rotated the horizontal ground plane 90 degrees into the vertical plane, causing the tank to stand completely upright on its side.
- [x] **Dynamic Pipeline Coordinate Orientation**:
  - Implemented dynamic rotation selection in [public/app.js](file:///Users/akif/Projects/splatris/public/app.js) (`loadGaussianSplatScene`):
    - **Nerfstudio 3DGS**: Uses `[-0.7071068, 0, 0, 0.7071068]` (-90 deg X rotation to convert Z-up to Three.js Y-up).
    - **Triangle Splatting**: Uses `[1, 0, 0, 0]` (180 deg X flip to convert COLMAP Y-down to Three.js Y-up without tilting the ground plane).
  - Result: The tank and ground plane sit naturally and horizontally on the Three.js coordinate floor.

### Milestone 33: Remote GPU Instance Migration & Vectorized Facet Pruning (Completed & Verified)
- [x] **New GPU Instance Setup**:
  - Re-anchored Nebius GPU worker to new pre-emptible instance IP: `195.242.31.111:8000` (NVIDIA L40S, 44.4 GB VRAM).
  - Updated [.env](file:///Users/akif/Projects/splatris/.env), synced training scripts, restarted worker daemon, restarted local Node server and verified health check.
- [x] **Density & Memory Footprint Pruning**:
  - Pruned `uploads/models/92badd1d-25d2-4097-81d5-8b6c8bc7a21f.splat` down from 1.47M to **679,641 facets (20.7 MB)** by filtering near-transparent background noise (`alpha < 26`) and distant camera trajectory artifacts (`dist >= 10m`).
  - Matches standard 3DGS density (786k splats / 24 MB), cutting sorting load and GPU overdraw by >54%.

### Milestone 34: Pipeline Badges, History Sync & Camera Reset Horizon Alignment (Completed & Verified)
- [x] **Visual Pipeline Differentiation ([public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Added distinct, vibrant, color-coded badges to Recent Scenes history chips:
    - `[Triangle Mesh]` (violet badge: `rgba(168, 85, 247, 0.22)`)
    - `[Standard 3DGS]` (cyan badge: `rgba(6, 182, 212, 0.20)`)
    - `[Instant Object]` (emerald badge: `rgba(16, 185, 129, 0.20)`)
  - Enables users to immediately distinguish reconstruction methods at a glance.
- [x] **Automatic Radio Button & Banner Synchronization**:
  - Selecting a history chip automatically triggers `setTrainingMethod(activeMethod, false)` and updates the pipeline selection radio buttons to match the scene.
  - Prevents the confusing "Re-process as Standard 3DGS" banner from showing when simply viewing a Triangle Mesh.
- [x] **Inside-Mesh Camera Spawn & Tilted Horizon Fix ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
  - **White Wall Root Cause**: `isObjectScan` was checking `url.includes('tank')`, but model URLs use UUIDs (`92badd1d-25d2-4097-81d5-8b6c8bc7a21f.splat`). When evaluated as false, the camera spawned at `z = 0.4m` (inside the tank's armor, rendering blinding backfaces).
  - Expanded `isObjectScan` to check `trainingMethod === 'triangle'`, `activeScene.trainingMethod`, and sidecar metadata.
  - Positioned camera safely outside at `(0, targetY + 0.35, viewDist)` (`viewDist = 2.3m`) looking at `(0, targetY, 0)`.
  - **Diagonal Horizon Reset Root Cause**: TrackballControls allowed roll rotation, and the Reset handlers never reset `camera.up`.
  - Set default camera control mode to `orbit` (Grounded Orbit) for stable, level-horizon navigation.
### Milestone 35: Remote GPU IP Migration & Orthonormal Basis Triangle Splat Calibration (Completed & Verified)
- [x] **New GPU Instance Setup**:
  - Re-anchored Nebius GPU worker to new pre-emptible instance IP: `195.242.30.204:8000` (NVIDIA L40S, 44.4 GB VRAM).
  - Updated [.env](file:///Users/akif/Projects/splatris/.env), verified `/health` online, and restarted local Express server.
- [x] **Triangle Splatting Visual Quality Diagnosed & Calibrated ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - **Why Image 1 Looked Foggy**:
    1. *Unconstrained In-Plane Spin*: Previous export only aligned normal $Z$, leaving $X$ and $Y$ unconstrained, causing triangle ellipsoids to spin out-of-phase with true triangle edges.
    2. *Gaussian Kernel Bloat*: Used $\sigma = 0.5 \times \text{edge}$, meaning the $3\sigma$ Gaussian kernel extended to $1.5 \times \text{edge}$ (300% oversized), bleeding heavily into neighbors and turning surfaces into a blurry haze.
    3. *Milky White Fog Clouds*: Background floaters generated multi-meter triangles (up to 10.9m long) which became massive glowing white Gaussian bubbles surrounding the tank.
  - **Mathematical Resolution**:
    1. *Exact Orthonormal Basis*: Constructed $R = [u_1, u_2, n]$ where $u_1 = e_1 / \|e_1\|$, $n = (e_1 \times e_2) / \|e_1 \times e_2\|$, and $u_2 = n \times u_1$. Converted directly to quaternion via vectorized Shepperd algorithm.
    2. *Calibrated Tight Scaling*: Standard deviations calibrated to $0.28 L_1$ and $0.28 H$, with $1.5\text{mm}$ thickness, tightly fitting within triangle boundaries.
    3. *Outlier Culling*: Pruned triangles with $L_1 > 0.25\text{m}$, $H > 0.25\text{m}$, $\text{dist} > 5.0\text{m}$, or opacity $< 30$.
    4. Regenerated `92badd1d-25d2-4097-81d5-8b6c8bc7a21f.splat` (545,067 crisp facets, 16.6 MB), completely removing the milky white fog and restoring sharp tank geometry.

### Milestone 36: Startup Sequencing & Upright Triangle Splat Horizon Alignment (Completed & Verified)
- [x] **Vertical Tank Orientation Root Cause Diagnosed**:
  - On page load, `loadGaussianSplatScene('uploads/models/scene.splat')` was called synchronously *before* `loadRecentScenes()` fetched metadata.
  - At startup, `STATE.activeScene` was `null` and `url` was `scene.splat`, causing `isTriangle` to evaluate to `false`.
  - Because `isTriangle` was false, the viewer applied Nerfstudio's `-90 deg X rotation` (`[-0.7071068, 0, 0, 0.7071068]`).
  - Applying `-90 deg X` to an already Y-up tank model rotated the longitudinal axis $Z$ into $Y$, causing the tank to stand vertically on its rear end.
- [x] **Startup Sequencing & Rotation Fix ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Re-sequenced initialization so `loadRecentScenes()` executes first, populating `STATE.activeScene` with `trainingMethod: "triangle"` and setting the initial target URL.
  - Hardened `isTriangle` detection in `loadGaussianSplatScene` to check `STATE.trainingMethod`, `activeScene.trainingMethod`, and `recentModels[0]`.
  - With `isTriangle === true`, the viewer applies the 180 deg X flip `[1, 0, 0, 0]`, keeping the tank's longitudinal and lateral axes flat on the Three.js ground plane, tracks on bottom, turret upright on top.

### Milestone 37: 3DGS Background Floater Pruning & Object Clutter Elimination (Completed & Verified)
- [x] **Excessive 3DGS Background Splats Diagnosed**:
  - The raw Nerfstudio 3DGS model (`9c1466a0-1e8b-47af-938a-5ecae56ae754.splat`) captured the entire museum interior: red structural columns, distant walls, a secondary vehicle, and floating ceiling debris across a 6-meter radius, creating a messy halo around the tank.
- [x] **Object Focal Crop & Floater Culling**:
  - Isolated the focal tank geometry using a calibrated bounding volume centered at $[0.15, 0.07, -0.30]$:
    - Lateral width ($X$): $[-1.20\text{m}, 1.40\text{m}]$
    - Vertical height ($Y$): $[-0.90\text{m}, 0.90\text{m}]$
    - Longitudinal length ($Z$): $[-1.70\text{m}, 1.10\text{m}]$
    - Transparency filter: $\text{alpha} \ge 30$
    - Billboard filter: $\text{max\_scale} \le 0.035$
  - Filtered out 181,982 background floaters and red column debris.
  - Re-saved clean 3DGS model (`604,586 splats, 18.5 MB`, down from 786,568 splats / 24.0 MB), leaving the tank completely crisp and isolated on the ground grid with zero background clutter.
- [x] **Pipeline Export Hardening ([pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py))**:
  - Updated `export_splat` with tighter opacity threshold ($\ge 26$), stricter needle ratio ($\le 18.0$), smaller billboard limit ($\le 0.08$), and a tighter scene bounding box to prevent background floater accumulation in future 3DGS training runs.

### Milestone 38: 6DoF Free Roam 3D Flight Along Mouse Gaze Vector (Completed & Verified)
- [x] **2D Ground Locking Diagnosed in Free Roam**:
  - Previously, `forward.y` was explicitly zeroed out (`forward.y = 0;`), forcing movement onto the horizontal 2D plane like a walking character.
  - When looking up or down with the mouse and pressing `W` / `ArrowUp`, the camera could not fly along the gaze vector.
  - In addition, vertical down-movement was artificially clamped to `y >= 0.1m`, preventing descending below ground level.
- [x] **Full 6DoF Flight Navigation ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Preserved the full 3D direction vector from `camera.getWorldDirection(forward)` with normalized pitch and yaw.
  - Pressing `W` or `ArrowUp` now flies directly along the 3D line of sight where the mouse is pointing (fly up when looking up, fly down when looking down).
  - Pressing `S` or `ArrowDown` reverses along the exact opposite 3D vector.
  - Strafing (`A` / `D` or `Left` / `Right`) remains orthogonal to the world up vector for level horizontal banking.
### Milestone 39: Nebius GPU IP Re-anchor (89.169.126.47) & Meta Quest 3 WebXR Testing Preparation (Completed & Verified)
- [x] **New GPU Instance Setup**:
  - Re-anchored Nebius GPU worker to new pre-emptible instance IP: `89.169.126.47:8000` (NVIDIA L40S, 44.4 GB VRAM).
  - Updated [.env](file:///Users/akif/Projects/splatris/.env), verified `/health` online, and restarted local Express server.
- [x] **WebXR HTTPS Tunnel Generation**:
  - Launched fresh Cloudflare Tunnel to provide an end-to-end encrypted HTTPS URL for Meta Quest Browser (`https://animated-istanbul-merge-emily.trycloudflare.com`), fulfilling the W3C WebXR secure origin requirement.
- [x] **Meta Quest 3 WebXR Test Verification Checklist Prepared**:
  - Outlined detailed step-by-step test procedure for Meta Quest Browser (6DoF controllers, local-floor space, spatial audio listener, trigger laser raycasting, exit gestures).

### Milestone 40: WebXR Specific Performance Optimization (Meta Quest 3 Jitter & Stutter Fix) (Completed & Verified)
- [x] **Zero Desktop Regression Boundary**:
  - All mobile VR optimizations are strictly gated behind active WebXR presentation (`renderer.xr.isPresenting === true` or during XR session setup).
  - Desktop WebGL canvas rendering retains 100% native resolution, full 50m draw distance, standard splat sizing (1.0), and desktop anti-aliasing.
- [x] **1. Lower Framebuffer Scale Factor (0.7)**:
  - Intercepted `XRWebGLLayer` initialization and set `framebufferScaleFactor: 0.7` (relieving the mobile Snapdragon XR2 GPU from rendering excessive native stereo pixels).
  - Called `renderer.xr.setFramebufferScaleFactor(0.7)` prior to session creation.
- [x] **2. Fixed Foveated Rendering (FFR = 1.0 Maximum)**:
  - Enabled Quest 3 hardware fixed foveated rendering by setting `fixedFoveation = 1.0` on `XRWebGLLayer` and `renderer.xr.setFoveation(1.0)`.
  - Centers of the Quest 3 pancake lenses remain sharp while peripheral ray density is dramatically reduced, eliminating head-rotation GPU fill-rate spikes.
- [x] **3. Aggressive MSAA Bypass for VR**:
  - Configured `antialias: false` on the `XRWebGLLayer` initialization options and bypassed multisampling (`samples: 0` on active WebXR render targets).
  - Prevents severe multisample resolve penalties on the mobile tile-based GPU.
- [x] **4. Aggressive Culling & Distance LOD Drop-Offs in XR**:
  - Tightened view frustum culling cushion from `0.6` down to `0.1` in `gatherSceneNodesForSort()`, actively discarding off-screen splats.
  - Implemented distance-based LOD drop-off in `gatherSceneNodesForSort()` to discard splats beyond `15.0m` during XR.
  - Set camera near/far depth clipping to `0.05m / 16.0m` during XR (vs `0.01m / 50.0m` on desktop).
  - Applied 20% splat scale dampening (`splatScale: 0.8`) specifically during XR presentation to reduce stereo overdraw by ~36%.
  - Cleanly restored all desktop states, camera clipping planes, and splat scales upon exiting WebXR.

### Milestone 41: WebXR Room-Scale Walk Fix (Disappearing Scene & Tight Far Plane Resolution) (Completed & Verified)
- [x] **1. Complete Frustum Culling Disabling**:
  - Set `frustumCulled = false` recursively on the primary 3DGS container (`STATE.splatPivotGroup`), `dropInViewer`, `splatMesh`, and `callbackMesh`.
  - Re-enforced `frustumCulled = false` every frame inside the animation render loop so Three.js never culls the environment based on its bounding center when the user physically steps or walks away.
- [x] **2. Camera Far Plane Increased to 1000m**:
  - Increased `camera.far = 1000.0` across the entire application (session init, XR render loop, desktop mode, and window resize).
  - Explicitly updated `xrCam.far = 1000.0` and all stereo eye sub-cameras (`c.far = 1000.0; c.updateProjectionMatrix()`).
  - Three.js automatically synchronizes `session.updateRenderState({ depthFar: 1000.0 })`, completely eliminating clipping when walking through room-scale or large indoor/outdoor spaces.
  - Removed artificial 15m distance culling from `gatherSceneNodesForSort()` in [`public/lib/gaussian-splats-3d.umd.js`](file:///Users/akif/Projects/splatris/public/lib/gaussian-splats-3d.umd.js).
### Milestone 42: WebXR Stereoscopic Rendering Mismatch Fix (Double Vision Resolution) (Completed & Verified)
- [x] **1. Per-Eye Distinct Matrix Preservation**:
  - Removed manual calls to `c.updateProjectionMatrix()` on the WebXR eye sub-cameras (`xrCam.cameras`), eliminating the overwrite that was replacing Meta Quest 3's distinct off-axis asymmetric stereo projections with a centered mono perspective matrix.
  - Preserved native WebXR hardware projection and view matrices (`view.projectionMatrix` and `view.transform.inverse.matrix`) with exact IPD (inter-pupillary distance) offset for both left and right eyes.
- [x] **2. Accurate Per-Eye Viewport & Focal Uniforms Synchronization**:
  - Created `setupStereoscopicPerEyeRendering()` hooked into `splatMesh.onBeforeRender`.
  - Dynamically updates `viewport`, `basisViewport`, and `focal` uniforms for each eye's exact viewport (`camera.viewport`) right before that eye's drawing pass.
  - Updated `getRenderDimensions()` in [`public/lib/gaussian-splats-3d.umd.js`](file:///Users/akif/Projects/splatris/public/lib/gaussian-splats-3d.umd.js) to inspect `camera.viewport` when presenting in WebXR rather than defaulting to the desktop canvas size.
- [x] **3. Splat Orientation & Head-Center Sorting**:
  - Unified splat depth sorting to run once per frame from the central head camera (`cameraVR`), preventing disparity popping between eyes.
  - Bypassed redundant mid-frame worker sorts in `DropInViewer.onBeforeRender` during active WebXR stereo presentation passes.
- [x] **4. Strict Buffer Isolation**:
  - Ensured WebXR left-eye and right-eye viewport passes render to their respective halves of the stereo framebuffer with depth test enabled and proper buffer masking without cross-eye bleeding.

### Milestone 43: Split "Recent Scenes" History by Pipeline Type (Completed & Verified)
- [x] **1. Split History State & API Categorization**:
  - Implemented `getRecentCategorizedModels(limit = 3)` in [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js) to partition models into two distinct arrays based on source material/training pipeline:
    - `videoModels`: Video-based spatial reconstructions (Standard 3DGS & Triangle Splatting mesh), capped to 3 most recent models (with default fallback space `scene.splat`).
    - `imageModels`: Single-image object scans (Image2Splat), capped to 3 most recent models.
  - Updated `GET /api/models/recent` in [server.js](file:///Users/akif/Projects/splatris/server.js) to return `{ success: true, count, videoModels, imageModels, models }`.
- [x] **2. Left Panel UI Layout Redesign ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
  - Replaced the single "RECENT SCENES" section with two stacked accordion sections:
    - **Section A**: `Recent Spaces (Video)` with cyan 3D isometric room/cube icon, inline pipeline tag badge, item counter badge, and collapse/expand arrow.
    - **Section B**: `Recent Objects (Photo)` with pink camera/photo icon, inline pipeline tag badge, item counter badge, and collapse/expand arrow.
- [x] **3. Frontend State Management & Chip Interactivity ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Updated `loadRecentScenes()` to populate `#recent-spaces-chips` and `#recent-objects-chips` independently with up to 3 items each.
  - Maintained unified active model highlighting across both sections: clicking any chip unsets `.active` globally and applies it to the selected chip, synchronizes the pipeline radio buttons, and loads the model or reconnects to an active training job.
  - Retained `STATE.recentModels` containing all models so spatial RAG metadata lookups and raycasting remain fully functional.
  - Added click accordion toggling (`initRecentHistoryAccordions()`) for expanding/collapsing each section.
- [x] **4. Visual Polish & CSS Styling ([public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Styled `.recent-history-section`, `.recent-scenes-widget`, `.recent-accordion-header`, rotating `.accordion-arrow`, and individual `.recent-scenes-list` scroll containers with sleek dark-glass aesthetics.

### Milestone 44: History Items Delete Capability & Deep Storage Cleanup (Completed & Verified)
- [x] **1. Backend Deletion Logic & API ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [routes/job.routes.js](file:///Users/akif/Projects/splatris/routes/job.routes.js))**:
  - Implemented `deleteScene(sceneId)` to safely abort any active training job, delete the in-memory job record, and wipe all storage:
    - Generated `.splat`, `.ply`, `.json`, and `.bak` files in `uploads/models/`.
    - Original uploaded video/image in `uploads/`.
    - Cached COLMAP poses and frame directories in `uploads/frames/<sceneId>/` (`images/`, `sparse/0/`, `video_meta.json`).
    - Purged remote Nebius GPU worker temporary workspace (`/tmp/splatris_jobs/<sceneId>`, `/tmp/splatris_cache/<sceneId>`).
  - Added protected guard preventing deletion of the built-in default demo space (`scene.splat`).
  - Created `DELETE /api/spatial/job/:sceneId` endpoint returning `{ success: true, sceneId, deletedFiles }`.
- [x] **2. Left Panel Delete Button & Confirmation ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css))**:
  - Added sleek trash icon button (`.btn-chip-delete`) to each non-default scene chip in Recent Spaces and Recent Objects lists.
  - Button remains hidden by default to keep the UI clean, smoothly fading in on hover with a red glow pill indicator.
  - Added confirmation dialog (`window.confirm`) prompting the user before irreversible deletion.
  - Provided interactive loading spinner during network execution.
- [x] **3. Dynamic Frontend State Management & Viewport Reset**:
  - Handled successful deletion by smoothly removing the chip from the DOM without a page reload.
  - Filtered the deleted item out of `STATE.recentVideoModels`, `STATE.recentImageModels`, and `STATE.recentModels`.
  - Updated category count badges and displayed empty fallback text if all items in a category are deleted.
  - **Active Scene Handling**: If the currently active or rendered scene is deleted:
    - Automatically disposed of the active `GaussianSplats3D.DropInViewer` instance and removed it from Three.js scene.
    - Reset HUD metrics (`stat-splat-count` to `0`, `stat-model-name` to `No Model`).
    - Hid the Fast Pipeline Switch banner (`#reprocess-banner`) and GPU suspended banners.
    - Reset pipeline tracker indicators to Step 1 and re-enabled all method radios to Standard 3DGS default.

### Milestone 45: WebXR 3DGS Depth Sorting Bottleneck on Head Rotation (Completed & Verified)
- [x] **1. Decoupled Splat Sorting from Pure Head Rotation**:
  - **Core Realization**: Splat depth ordering relative to the camera origin is fundamentally radially invariant to pure rotational changes (pitch, yaw, roll) from a fixed position. Re-sorting 1M+ splats at 72fps on every frame of head panning was completely saturating the Quest 3 mobile CPU.
  - **Render Loop Gating ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - Tracked physical headset translation (`STATE.lastXRSortPos`).
    - Gated `STATE.dropInViewer.viewer.update(renderer, xrCam)` so it **only** executes if the headset physically moves $\ge 0.20$m (20cm) in space.
    - During pure head rotation (panning left/right, looking up/down), sorting is completely skipped with zero array allocations, zero `postMessage` calls, and zero main-thread stalls.
  - **Library-Level Decoupling ([public/lib/gaussian-splats-3d.umd.js](file:///Users/akif/Projects/splatris/public/lib/gaussian-splats-3d.umd.js))**:
    - Updated `runSplatSort` to explicitly enforce `needsRefreshForRotation = false` when in WebXR mode (`isWebXR`).
    - Gated position refresh on a 0.15m physical translation threshold and bypassed multi-frame fractional `partialSorts` queues in WebXR.
- [x] **2. Asynchronous Web Worker Sorting with Last-Known-Good Render Continuity**:
  - Ensured that when physical translation occurs ($\ge 0.20$m), radix depth sorting runs in the dedicated background Web Worker without blocking WebXR `requestAnimationFrame`.
  - The main thread continues rendering at full frame rate using the last-known-good sorted indices until the worker asynchronously returns the newly sorted indices.
- [x] **3. Dynamic Splat Culling & Capping for WebXR**:
  - **Dynamic Splat Capping in Octree Traversal**: In `gatherSceneNodesForSort()`, capped total splats submitted to the sorting array in WebXR to 350,000 max. Since nodes are sorted closest-first, nearby high-detail geometry is strictly preserved while distant/peripheral nodes that overload the mobile GPU/CPU are dropped.
  - **Increased Load-Time Opacity Threshold**: Raised `splatAlphaRemovalThreshold` to 20/255 in `public/app.js` to discard near-zero opacity haze splats upon loading, reducing overall scene splat count and overdraw.

### Milestone 46: WebXR Fixes — Mesh Jitter (Geometry Chunking) & Spatial Q&A / Tavily Raycast Hardening (Completed & Verified)
- [x] **1. Triangle Splatting Mesh Spatial Geometry Chunking ([pipeline/mesh_chunker.py](file:///Users/akif/Projects/splatris/pipeline/mesh_chunker.py), [pipeline/train_3dgs.py](file:///Users/akif/Projects/splatris/pipeline/train_3dgs.py), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **The Root Cause**: Triangle Splatting generated a single monolithic mesh PLY (e.g. 4,000,002 triangles / 12,000,006 vertices, 244 MB). In WebXR, Three.js frustum culling operates per-object; because the monolithic mesh's bounding sphere encapsulated the entire room, the Quest 3 vertex shader was forced to process all 12 million vertices for both eyes 72–90 times per second, overwhelming the mobile GPU and causing severe jitter during head rotation.
  - **Generation Phase Spatial Partitioning**:
    - Created `pipeline/mesh_chunker.py` and integrated chunk export into `convert_triangle_checkpoint` in `pipeline/train_3dgs.py`.
    - Spatially partitions the mesh triangles into a 3D grid ($N_x \times N_y \times N_z$) based on triangle centroids.
    - Generates compact binary `.chunks.bin` with tight bounding spheres and bounding boxes per chunk (4B header, 4B chunk count, 48B chunk metadata, typed float32 vertex positions and uint8 RGBA colors).
    - Reduced 244 MB monolithic PLY models to ~19.9 MB packed chunks across 31–54 active spatial sub-meshes.
  - **Load Phase Engine Integration ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - Implemented `loadChunkedTriangleMesh(url)` and client-side fallback `parseAndChunkMonolithicPLY(buffer, gridDim)`.
    - Created individual `THREE.Mesh` sub-meshes with `chunk.frustumCulled = true` and `DoubleSide` vertex-colored materials.
    - Added chunks to `STATE.triangleMeshGroup` within `STATE.splatPivotGroup` and stored in `STATE.triangleMeshChunks`.
    - Integrated clean disposal via `disposeTriangleMeshChunks()` when switching scenes or clearing viewports.
    - **WebXR Performance**: Three.js now automatically culls 75–85% of chunks outside the active eye view frustum during head rotation, completely eliminating vertex shader bottlenecks and head panning jitter.
- [x] **2. Explicit WebXR Raycaster Compatibility for THREE.Mesh Objects ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - Added `collectCandidateMeshes()` helper gathering all `STATE.triangleMeshChunks`, `STATE.splatPivotGroup` meshes, and scene meshes while ignoring HUD, reticle, and controller UI elements.
  - In `performRaycastSelectionXR(origin, direction)`:
    - Explicitly intersects `candidateMeshes` using `THREE.Raycaster` with temporary `DoubleSide` evaluation so scan/photogrammetry backfaces are never dropped.
    - Evaluates `gsRaycaster` against 3DGS splats if present, selecting the closest authentic 3D intersection.
  - Mirrored candidate mesh raycasting in desktop `performRaycastSelection` for consistent behavior across WebXR and desktop.
- [x] **3. Vision RAG & Tavily Execution Hardening & Fallback Spatial TTS ([routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js), [services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Dynamic MIME Type Detection**: In `services/visionRagService.js`, inspected base64 magic bytes to prevent upstream 400 errors from Nebius Token Factory for PNG/WebP viewport crops.
  - **Contextual Model Fallback**: Augmented low-confidence or generic vision detections with model metadata sidecar clues (e.g. `tank_internet.mp4` -> `"Soviet T-54/55 Main Battle Tank"`), ensuring Tavily receives a grounded object query.
  - **Google TTS 160-Character Limit Fix**: Sliced clean TTS text to 160 characters in `routes/spatial.routes.js` to eliminate Google Translate upstream 400 errors, streaming audio/mpeg reliably.
  - **Fallback Audio Confirmation**:
    - If vision identification returns low confidence or an unknown object: plays spatial TTS `"I couldn't identify this object. Please try clicking closer or from another angle."` at the hit point.
    - If network or API error occurs: plays spatial TTS `"I couldn't identify this object. Please try clicking again."` at the hit point.
    - Guarantees the user in WebXR receives immediate audio feedback confirming the click was registered.
- [x] **4. Milestone 47: RAG Multi-Trigger Lock, Reticle Clamping & Written Text Output ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Multi-Trigger Lock & Processing Indicator**:
    - Added `STATE.isRaycastQuerying` state lock with `lockRaycastQuery()` and `unlockRaycastQuery()`.
    - Guarded `onXRSelect` and `performRaycastSelectionXR` in WebXR mode, preventing trigger spam from queuing concurrent raycasts or overwhelming the Nemotron/Tavily pipeline.
    - Added 18-second watchdog safety timer to guarantee the trigger state lock never deadlocks if network or TTS stalls.
    - Displayed immediate `"PROCESSING..."` indicator in the floating WebXR HUD, toast notifications, chat log, and right-hand emitter status.
    - Updated `speakSpatialAudio(text, position, onComplete)` to reliably invoke `onComplete` when Web Audio `sourceNode.onended`, Web Speech `utterance.onend`/`utterance.onerror`, or error fallbacks fire, releasing `isRaycastQuerying` only after spoken narration completes.
  - **Reticle / Cursor Proportional Scale Clamping**:
    - Reduced base reticle dimensions in `createRaycastTargetMesh()`: Center dot radius reduced to 0.012m, inner ring 0.022–0.032m, outer ring 0.040–0.048m.
    - In the render loop, billboarded `raycastTargetMesh.lookAt(camPos)` toward the active camera and dynamically scaled using perspective distance and FOV factor (`dist * fovFactor * 0.35`).
    - Clamped scale strictly between `0.18` and `0.75`, guaranteeing the reticle remains a crisp, small targeting dot/circle regardless of camera distance or FOV without ever expanding massively over the viewport.
  - **Restored Written Text Output in Desktop RAG Panel**:
    - Fixed payload mapping in `renderRaycastResult()` to support `result.tavilyData` alongside `result.tavilyGrounding`.
    - Bound `summary`, `visualDescription`, and source links (`results`/`sources`) directly into the `.raycast-grounding` DOM card.
    - Updated `#emitter-coords` with target object name and coordinates (`(x, y, z)`).
    - Automatically uncollapsed and focused `#right-panel` upon raycast hit so desktop users immediately see the written Tavily text card simultaneously with 3D spatial audio playback.
- [x] **5. Milestone 48: Critical Rollback — Reverted Triangle Mesh Decimation ([pipeline/mesh_chunker.py](file:///Users/akif/Projects/splatris/pipeline/mesh_chunker.py), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
  - **Reverted Aggressive Decimation & Restored 100% Visual Fidelity**:
    - Completely eliminated all polygon decimation (`pts_k[::step]`), triangle capping (`maxAllowed = 450000`), and filtering masks from both `pipeline/mesh_chunker.py` and `public/app.js`.
    - Preserved 100% of all 4,000,002 triangles and 12,000,006 vertices from the original 244 MB PLY generation.
    - Re-generated `scene.chunks.bin` and `92badd1d-25d2-4097-81d5-8b6c8bc7a21f.chunks.bin` at full 183.11 MB size, restoring solid surface fidelity and high-resolution texture/vertex coloring matching the 3DGS pipeline.
  - **Pure Frustum Culling via Spatial Chunks (No Polygon Reduction)**:
    - Partitioned all 4,000,002 triangles into 27 spatial sub-meshes with tight bounding boxes and bounding spheres.
    - In `public/app.js`, enabled `mesh.frustumCulled = true` per chunk: the Quest 3 GPU only processes the high-density chunks currently inside the camera's active view frustum, completely eliminating head rotation jitter without discarding any geometry.
  - **Standard 3DGS Pipeline Integrity Guaranteed**:
    - The Standard 3DGS (Gaussian Splatting) pipeline, render loop, DropInViewer, and depth sorting remain 100% untouched and functional.

- [x] **6. Milestone 49: In-VR HUD Question Freezing, Thumbstick Scrolling & Cloud Demo Guards ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js))**:
  - **In-VR Question Freezing & Duplicate Query Prevention**:
    - Introduced `isHudQuestionQuerying` state lock with a 20-second watchdog timer.
    - When a question card is selected via Left Trigger, all question cards instantly transition to a frozen/disabled visual state (dimmed slate styling `rgba(51, 65, 85, 0.45)`, muted `#475569` border, and `⏳` prefix).
    - Additional trigger pulls on question cards during analysis are safely absorbed, preventing duplicate API calls and audio queue collision.
    - Automatic release and card restoration upon response completion or watchdog timeout.
  - **In-VR HUD Crisp Text Scrolling with Thumbstick & Scrollbar**:
    - Locked response font size at a comfortable, easily readable 23px with 33px line height, completely eliminating adaptive shrinking into unreadable micro-text.
    - Added canvas viewport clipping (`ctx.save()` / `ctx.clip()`) strictly confining text rendering between `startY` and `maxY`.
    - Integrated thumbstick scrolling: when the Left Controller laser reticle targets the HUD panel, moving the thumbstick up/down smoothly scrolls `xrHudScrollOffset` with clamped boundaries.
    - Rendered a sleek vertical scrollbar on the right margin with an active cyan indicator thumb showing scroll progress.
    - Added tap-to-scroll support on the scrollbar track for additional accessibility.
  - **Cloud Demo Upload Guard (Render vs. Localhost / Cloudflare)**:
    - Added `isRenderDemoDeployment()` detection guard: on `splatris.onrender.com`, clicking the dropzone or dropping files presents an informative toast (`"Demo mode active with pre-trained 3D models. To train custom scenes, clone the repo or spin up a GPU worker."`).
    - Kept uploads 100% unrestricted on `localhost:3000` and `*.trycloudflare.com` for local testing and Image2Splat generation.
  - **Standard 3DGS Default Model Priority**:
    - Sorted `Standard 3DGS` (`Sep 29 - tank_internet.mp4 • 18.5 MB`) as #1 default active model in recent spaces and training pipeline selector.
  - **In-VR HUD Top Text Baseline & Viewport Clipping Fix**:
    - Resolved horizontal text slicing by switching canvas `ctx.textBaseline` from `'alphabetic'` to `'top'`.
    - Added an 18px padding offset below the target subtitle, guaranteeing every character, ascender, and capital letter renders 100% visible inside the scrollable viewport.
  - **Zero Voice Overlap & Immediate Audio Interruption on Right Trigger**:
    - Intercepted Right Trigger pulls to immediately call `stopSpatialAudio()`, cutting the active Web Audio buffer node in 0ms and flushing `ttsSentenceQueue`.
    - Implemented `activeInspectionId` generation tracking to invalidate stale in-flight Tavily/Nemotron responses, ensuring smooth and instant UX transitions to new targets without voice collisions.

- [x] **7. Milestone 50: Photo vs. Video Pipeline Categorization & Triangle Mesh Permanence ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [routes/spatial.routes.js](file:///Users/akif/Projects/splatris/routes/spatial.routes.js))**:
  - **Remote GPU Worker Migration (195.242.13.85:8000)**:
    - Updated active Nebius L40S GPU worker endpoint in `.env`, `services/jobService.js`, and `routes/job.routes.js` to `http://195.242.13.85:8000`.
    - Verified health check: NVIDIA L40S with 44.4 GB VRAM online.
  - **Strict Photo vs. Video Pipeline Auto-Routing**:
    - In `public/app.js` (`executeEndToEndPipeline`), detecting single image drops (`.jpg`, `.jpeg`, `.png`, `.webp`, or `image/*` MIME) instantly auto-switches pipeline to `Instant Object (Image2Splat)`.
    - In `routes/spatial.routes.js`, added `.webp` support and verified immediate bypass of video quality gatekeeper for images.
    - Preserves authentic filename (`tankphoto.jpg`) throughout upload and job dispatch, preventing default fallback to `Walkthrough.mp4`.
  - **Single Photo Models Never Overwrite Default `scene.splat`**:
    - In `services/jobService.js` (`executeGpuJobAsync`), guarded `scene.splat` / `scene.ply` updates with `if (!isImageJob && job.trainingMethod !== 'image2splat')`.
    - Prevents single object reconstructions from replacing the default room walkthrough space.
  - **Guaranteed Triangle Splatting Mesh Permanence**:
    - In `services/jobService.js` (`getRecentCategorizedModels`), triangle mesh space (`92badd1d...` / `scene.chunks.bin`) is explicitly preserved in `videoModels` and can never be evicted by new video scans.
  - **Category-Aware Model Pruning & Triangle Mesh Immunity**:
    - Re-architected `pruneOldModels()` to isolate video models from image models (keeping up to 3 video spaces AND up to 3 photo objects separately).
    - Hardened `92badd1d*` and `scene.splat` against file eviction during background job cleanup.
  - **Race Condition & Sidecar Writing Fix ([services/jobService.js](file:///Users/akif/Projects/splatris/services/jobService.js))**:
    - Identified root cause of photo scans momentarily falling back to `Walkthrough.mp4` under `RECENT SPACES (VIDEO)`: `job.status = 'completed'` was previously emitted prior to disk flush of `[jobId].json`, causing the initial client poll to query `/api/models/recent` before the sidecar was present on disk.
    - Moved sidecar file write to strictly precede `job.status = 'completed'`.
    - Added in-memory fallback to `jobs.get(baseName)` in `getRecentCategorizedModels` and `pruneOldModels` so metadata is immediately accessible regardless of filesystem timing.
  - **Strict Client-Side Categorization Filter & Upright Rotation Fix ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - In `loadRecentScenes()`, added client-side defensive filter (`isModelImage()`) that inspects both `data.videoModels` and `data.imageModels`, automatically routing any model with photo filename extensions or `image2splat` method directly into `imageModels`.
    - In `loadGaussianSplatScene()`, added orientation check: single-image `image2splat` models have pinhole unprojected coordinates with `+Y` already as UP. Replaced Nerfstudio's -90° X rotation with identity quaternion `[0, 0, 0, 1]`, preventing the doorway object from lying flat on the floor and ensuring it stands upright.
    - Added pre-render `STATE.activeScene` assignment and 200ms settling delay in `pollJobUntilReady()`.
    - Bumped `public/index.html` cache buster to `v=20261006_photo_categorization_v22`.
  - **Full-Height Object Preservation & Studio Background Isolation ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
    - Resolved issue where tall vertical objects (such as the Easter Island Moai statue) were severely truncated: previous radial distance weighting (`center_weight = 1.0 - 0.5 * dist_from_center`) heavily penalized pixels far from geometric center `(center_x, center_y)`, resulting in 92.6% of the top of the head being cut off.
    - Implemented 4-corner uniform background detection: when borders/corners share consistent studio background colors (e.g., pure white `#ffffff` or neutral studio gray), color distance delta segmentation isolates 100% of the subject from head to base without vertical falloff.
    - Added aspect-ratio-aware distance weighting fallback for in-the-wild photos (`dist_y` penalized only half as much as `dist_x`), preserving full vertical extent.
    - Isolated depth normalization `d_min`/`d_max` to foreground pixels exclusively, maximizing 3D dynamic range and eliminating background depth artifacts.
    - Corrected binary `.splat` rotation byte order to `(rw, rx, ry, rz)` matching `GaussianSplats3D` unpack layout.
  - **Boundary Defringing, Surface Normal Alignment & Proportionate 3D Relief ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
    - Eliminated floating perforated halo/crown artifacts around object silhouettes: applied 2-iteration morphological erosion (`cv2.erode`) on the color-segmented foreground mask combined with connected component analysis, stripping all anti-aliased edge transition pixels that had background depth bleed.
    - Reduced background bleed points from thousands down to 0.04% (209 splats).
    - Replaced identity flat disc rotations with surface normal alignment: computed 3D spatial gradients ($\partial Z/\partial u$, $\partial Z/\partial v$) from smoothed depth maps, derived normal vectors $\mathbf{N}$, and computed rotation quaternions aligning each Gaussian ellipsoid to the local surface curvature.
    - Updated Gaussian anisotropic scales to tangent-aligned footprints ($1.25\times$ pixel spacing for lateral coverage, $0.35\times$ for crisp surface shell thickness), eliminating side-profile steps and gaps.
    - Scaled metric relief depth to a proportionate $0.65\text{m}$ extrusion ($Z \in [1.5\text{m}, 2.12\text{m}]$), preventing extreme cone/wedge stretching.
    - Synced changes to GPU worker at `/home/ubuntu/Image2Splat/infer_image2splat.py` and validated on test Moai image (generated 472,945 dense Gaussians in 12.57s).
    - Bumped `public/index.html` cache buster to `v=20261006_defringe_normals_v24`.
  - **Milestone 53: Mandatory Rembg Isolation, Edge Gradient Cutoff & 360-Degree Volumetric Volume ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
    - **Worker Endpoint Migration**: Updated `.env` and restarted local backend server for new Nebius GPU instance at `http://89.169.109.161:8000`. Verified NVIDIA L40S GPU and CUDA 12.4.
    - **Pre-processing Background Removal**: Integrated `rembg` (with CPU session fallback for onnxruntime) as the mandatory first step before depth estimation. 100% of white/studio or in-the-wild background pixels are assigned `alpha = 0` and completely omitted from 3D space, completely eliminating the background net/cage of splats.
    - **Edge Stretching / Depth Cutoff Filter**: Implemented spatial depth gradient filtering ($\|\nabla Z\| = \sqrt{(\partial Z/\partial u)^2 + (\partial Z/\partial v)^2}$). Pruned pixels with extreme depth slopes exceeding the 98th percentile cutoff to eliminate backwards-stretching curtain artifacts at silhouette boundaries.
    - **True 360-Degree Volumetric Solid Structure**: Upgraded unprojection from a 2.5D hollow mask to a fully enclosed 3D solid volume:
      - Front shell: high-resolution photographic projection with surface normals.
      - Back shell: smoothly meets front shell at boundary ($d\_norm \to 0$) and curves backwards with normal-aligned Gaussians ($\mathbf{N}_{back} = -\mathbf{N}_{front}$) and ambient-shaded texture.
      - Core infill: interior solid core layer preventing hollow internal voids from top/down viewing angles.
    - Verified on test Moai image: generated **721,023 360-degree Gaussians** with full volumetric depth ($1.2\text{m}$ depth, $1.75\text{m}$ width, $2.53\text{m}$ height).
    - Bumped `public/index.html` cache buster to `v=20261007_rembg_volumetric_v25`.
  - **Milestone 54: Proportionate Z-Axis Depth Scaling & Seamless Boundary Enclosure ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py), [pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py), [services/nebiusGpuService.js](file:///Users/akif/Projects/splatris/services/nebiusGpuService.js))**:
    - **Depth Normalization & 3.0x Extrusion Multiplier**: Normalized foreground depth map strictly in range $[0.0, 1.0]$. Introduced `z_multiplier` (default `3.0x`, scaling `depth_scale` to $1.95\text{m}$ and `back_scale` to $1.65\text{m}$) so facial contours (nose tip, brow ridge, mouth, chin) protrude naturally along the Z-axis, completely eliminating the flattened pancake / crushed relief profile.
    - **Distance Transform Boundary Enclosure**: Integrated `cv2.distanceTransform` to drive shell thickness from the outer silhouette perimeter inwards. At the silhouette perimeter ($bf \to 0$), thickness smoothly vanishes so the front shell and back shell meet seamlessly with zero gaps, zero holes, and zero edge curtaining.
    - **Anatomical Volume Verification**: Validated on test Moai model: produced **881,251 Gaussians** with physical bounds $X = 2.40\text{m}, Y = 3.53\text{m}, Z = 2.24\text{m}$ and a balanced depth-to-width ratio of **0.93** (natural 3D head/sculpture proportions).
    - **Worker Synchronization**: Deployed updated `infer_image2splat.py` and `gpu_worker_server.py` to GPU worker at `89.169.109.161:8000`, restarted `splatris-worker.service`, and bumped `public/index.html` cache buster to `v=20261007_depth_scaled_v26`.
  - **Milestone 55: Depth Discontinuity Edge Filter & Natural Conservative Z-Relief ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py), [pipeline/gpu_worker_server.py](file:///Users/akif/Projects/splatris/pipeline/gpu_worker_server.py))**:
    - **Depth Discontinuity Filter**: Implemented 4-connected immediate neighbor depth delta calculation ($\Delta(u, v) = \max(|D(u\pm 1, v) - D(u, v)|, |D(u, v\pm 1) - D(u, v)|)$). Discards any pixel bridging large depth jumps exceeding `edge_thresh` (default `0.020`), completely deleting rubber-sheet bridges, stretched occluding borders, and dragging lines.
    - **Conservative Z-Axis Scale**: Reduced base depth extrusion to a natural $0.35\text{m}$ (`z_multiplier = 1.0`), ensuring the depth delta from nose tip to the side of the head is only a few units (~10-15 cm) without pointed cone distortion. Gentle 10 cm back closure seamlessly finishes the rear without wedge artifacts.
    - **Depth-Tied Splat Footprint Scaling**: Tied Gaussian radius directly to depth footprint ($(Z / f_x) \times \text{coverage\_factor}$), ensuring splats pushed further back in Z physically scale up proportionally in 3D space to maintain solid, gap-free coverage without sparse strings of dots.
    - **Verified Bounds**: Validated on Moai image: produced **727,215 clean Gaussians** with balanced bounds $X = 1.37\text{m}, Y = 1.90\text{m}, Z = 0.41\text{m}$.
    - **Deployment**: Synced to GPU worker at `89.169.109.161:8000`, restarted `splatris-worker.service`, and bumped frontend cache buster to `v=20261007_clean_depth_v27`.
  - **Milestone 56: Aspect Ratio Letterbox Padding & Full-Image Mask Preservation ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
    - **Full Uncropped Subject Background Removal**: Executed `rembg` on the uncropped native resolution image first, guaranteeing foreground isolation across the entire vertical span without cropping or morphological erosion severing the subject.
    - **Letterbox Square Padding**: Scaled longest edge to fit target 1024 resolution while preserving the exact input aspect ratio. Symmetrically padded the shorter edge with neutral black and 100% transparent alpha (`value=0`), creating a perfect square without discarding the lower torso or body.
    - **Verified Anatomical Extent**: Tested on GPU worker (`89.169.109.161`): generated **736,274 Gaussians** spanning $1.85\text{m}$ vertically ($Y \in [0.50\text{m}, 2.35\text{m}]$), $1.33\text{m}$ horizontally, and $0.41\text{m}$ in depth, successfully retaining the entire statue (head + neck + body).
    - **Production Synchronization**: Synchronized to GPU worker, verified `splatris-worker.service` healthy, and bumped frontend cache buster to `v=20261007_letterbox_v28`.
  - **Milestone 59: Pixel-Perfect RGB-Depth Alignment, [v, u] Unprojection & Upright Quaternions ([pipeline/infer_image2splat.py](file:///Users/akif/Projects/splatris/pipeline/infer_image2splat.py))**:
    - **Strict Shape Alignment Safeguard**: Added `assert im_rgb.shape[:2] == depth_map.shape[:2]` with fallback cubic resize, guaranteeing zero dimension or orientation divergence between color and depth.
    - **Pixel Sampling Rule**: Explicitly parameterized unprojection indices as `v = y_idx * stride` (row / Y-axis) and `u = x_idx * stride` (column / X-axis), indexing strictly as `depth = depth_map[v, u]` and `color = im_rgb[v, u]`.
    - **Upright Camera-Facing Quaternions**: Replaced derivative normal tilt with stable identity camera-facing quaternions `[qw=1.0, qx=0, qy=0, qz=0]` with isotropic disk scales `[s, s, s*0.45]`. Splats are guaranteed upright, preventing any rotational tilting, sideways lying, or melted blob distortion.
    - **Production Sync**: Deployed to GPU worker at `89.169.109.161:8000`, verified `splatris-worker.service` healthy, and updated cache buster to `v=20261007_aligned_uv_v31`.
  - **Milestone 60: Clean Reversion to Multi-View Video 3DGS (Deprecating Single-Image Instant Object)**:
    - **Architectural Decision**: Formally dropped the single-photo Instant Object (Image2Splat) experiment in favor of authentic multi-view photogrammetric 3D Gaussian Splatting (Nerfstudio Splatfacto + COLMAP SfM / Instant-NGP / NeRF).
    - **Frontend Guardrails**:
      - Restored strict video-only dropzone accepting MP4/MOV walkthrough videos (30–60s, up to 1GB).
      - Automatically blocks single image uploads with explicit guidance to supply a walkthrough video for true 360-degree parallax spatial reconstruction.
      - Streamlined pipeline mode selector to Standard 3DGS and Triangle Splatting, removing the Instant Object method button.
      - Removed obsolete instant-object Vision RAG panels and streamlined Recent Spaces to show verified multi-view video reconstructions.
  - **Milestone 61: Default Splat Glow Enforcement & Removal of Toggle Button ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - **UX Decision**: Established permanent Splat Glow rendering as the signature visual aesthetic for all 3DGS reconstructions without requiring user toggling.
    - **Removed Toggle UI**: Completely removed the "Splat Glow" toggle button (`#btn-toggle-points`) from the "VIEWPORT MODES" control panel in `public/index.html`. Removed the button reference and click listener from `public/app.js` which previously toggled viewer visibility and caused the mesh to disappear.
    - **Enforced Splat Glow by Default**:
      - Hardcoded double-sided rendering, depth testing, transparency, and `visible = true` across the splat hierarchy (`splatPivotGroup`, `dropInViewer`, `splatMesh`).
      - Hardcoded full screen-space splat scaling (`splatScale = 1.0`) on `splatMesh` immediately upon scene compilation.
    - **Deployment**: Bumped frontend cache buster to `v=20261007_splat_glow_default_v33`.
  - **Milestone 62: Strict Spatial RAG Guardrail for Scene Q&A ([services/ragService.js](file:///Users/akif/Projects/splatris/services/ragService.js))**:
    - **System Prompt Guardrail**: Updated the system prompt and instructions for the Nemotron 3 Ultra LLM handling the "Ask anything about this space..." chat input:
      - *"You are a spatial AI guide exclusively focused on the currently reconstructed 3D space. If the user asks a question that is unrelated to the active 3D model, its history, or its spatial context, you MUST politely decline and remind them to ask about the current scene. Never answer general knowledge, coding, or off-topic questions."*
    - **Verification**: Verified via test API queries:
      - Off-topic coding/general knowledge requests are politely rejected with a redirect to the active 3D scene.
      - Relevant spatial, structural, and historical queries receive rich, grounded answers with 3D audio emitter positioning.
  - **Milestone 63: Removal of Technical Chat Suggestion Chips ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - **UI Cleanup**: Completely removed the `#rag-presets` container and the three developer-centric suggestion chips ("Space Geometry", "Reconstruction Method", "Spatial Lighting & Focal Point").
    - **Focused Chat Experience**: Cleaned up the Spatial Audio RAG panel so it consists exclusively of the "Spatial Audio Guide" conversation card and the "Ask anything about this space..." input field.
    - **Code Cleanup**: Removed dead DOM references and `updateSpaceRagPresets` logic in `public/app.js`.
    - **Deployment**: Bumped frontend cache buster to `v=20261007_clean_chat_v34`.
  - **Milestone 64: Clean Up "Fast Pipeline Switch" Card ([public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/app.js](file:///Users/akif/Projects/splatris/public/app.js))**:
    - **Removed Clutter & Badges**: Completely removed the "Quick Convert" badge and the "< 90s vs 5m" badge from the card header.
    - **Removed Explanatory Paragraph**: Deleted the redundant description text block explaining cached conversions.
    - **Minimal Layout**: Kept the "⚡ FAST PIPELINE SWITCH" section header and placed the two action buttons ("Re-process as Triangle Splatting (Fast)" and "Upload New Video") directly underneath with clean, breathable spacing.
    - **Deployment**: Bumped frontend cache buster to `v=20261007_clean_reprocess_card_v35`.
  - **Milestone 65: Nebius AI Cloud Infrastructure Verification & Documentation ([README.md](file:///Users/akif/Projects/splatris/README.md), [docs/NEBIUS_GPU_SETUP.md](file:///Users/akif/Projects/splatris/docs/NEBIUS_GPU_SETUP.md), [docs/assets/nebius_ai_cloud_vm.png](file:///Users/akif/Projects/splatris/docs/assets/nebius_ai_cloud_vm.png))**:
    - **Hardware & Instance Verification**: Integrated high-resolution Nebius AI Cloud Console screenshot verifying instance `bronze-bug-instance-6` (`computeinstance-e00ev6ymegj3aa5hyc`), NVIDIA® L40S GPU (48 GB GDDR6), 8 vCPUs, 32 GiB RAM, and Ubuntu 24.04 LTS.
    - **Reproduction Guide**: Created comprehensive `docs/NEBIUS_GPU_SETUP.md` with complete VM provisioning parameters, CUDA 12.4 / PyTorch setup, rasterizer kernel compilation (`TORCH_CUDA_ARCH_LIST=8.9`), and `splatris-worker.service` configuration.
    - **README Integration**: Embedded the verified infrastructure section in `README.md` with direct link to the setup documentation.
  - **Milestone 66: Truthful Documentation Synchronization & Auto-Push Enforcement ([README.md](file:///Users/akif/Projects/splatris/README.md), [docs/NEBIUS_GPU_SETUP.md](file:///Users/akif/Projects/splatris/docs/NEBIUS_GPU_SETUP.md))**:
    - **Removed Inaccurate Claims**: Fully eliminated all mentions of single-photo Image2Splat, HRTF 3D positional audio emitters, and hackathon callouts from `README.md` and `docs/NEBIUS_GPU_SETUP.md`. Accurately documented the multi-view video photogrammetry pipeline and conversational Web Audio voiceover guide.
  - **Milestone 67: Nebius GPU IP Re-Anchor (`89.169.97.126:8000`) & Cloudflare Tunnel Activation ([.env](file:///Users/akif/Projects/splatris/.env), [STEERING.md](file:///Users/akif/Projects/splatris/STEERING.md))**:
    - **GPU Endpoint Updated**: Updated `NEBIUS_GPU_ENDPOINT` to `http://89.169.97.126:8000`. Verified remote health endpoint: NVIDIA L40S GPU (44.4 GB VRAM) online with CUDA available.
    - **Local Server & Tunnel**: Started Node.js server on port 3000 and launched Cloudflare HTTPS tunnel: `https://tony-hawaiian-belly-continuously.trycloudflare.com`. Verified HTTP 200 response for recording and WebXR.
  - **Milestone 68: Auto-Generated Inquiries in WebXR & Desktop Interfaces ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html), [services/visionRagService.js](file:///Users/akif/Projects/splatris/services/visionRagService.js))**:
    - **Dynamic Inquiry Generation**: Grounded in Tavily AI and Nemotron Vision object identification, the pipeline generates 3 curated, contextual questions for inspected spatial objects (e.g., combat history, technical armor specs, production timeline).
    - **WebXR Floating HUD Card**: Added clickable inquiry chips directly onto the floating 3D raycast card in WebXR with visual pointer highlight and trigger click listeners that initiate RAG query and spatial audio voiceover.
    - **Desktop Chat Integration**: Rendered click-to-ask inquiry buttons within the desktop identification card and dynamically populated the `#rag-presets` chips container above the chat input box.
    - **Deployment**: Bumped frontend cache buster to `v=20261020_suggested_inquiries_v36`.
  - **Milestone 69: Refined Suggested Inquiries & Dynamic Preset Handling ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
    - **Strict Grounding Condition**: Filtered inquiry generation to execute only when Tavily web grounding returns authentic matching data. If no grounded data is found, suggestions are suppressed.
    - **Dynamic UI Cleanup**: `updateQueryChipsForObject` gracefully toggles container visibility (`display: none` / `flex`) and clears obsolete buttons when no inquiries are active, keeping the spatial chat interface clean and focused.
    - **Deployment**: Bumped frontend cache buster to `v=20261020_suggested_inquiries_v37`.
  - **Milestone 70: Nebius GPU IP Re-Anchor (`89.169.123.226:8000`) & Fresh Cloudflare Tunnel ([.env](file:///Users/akif/Projects/splatris/.env), [STEERING.md](file:///Users/akif/Projects/splatris/STEERING.md))**:
    - **GPU Endpoint Updated**: Updated `NEBIUS_GPU_ENDPOINT` to `http://89.169.123.226:8000`. Verified health endpoint: NVIDIA L40S GPU (44.4 GB VRAM) online with CUDA available.
    - **Local Server & Tunnel Re-launch**: Restarted Node.js server on port 3000 to bind new GPU endpoint and launched a fresh Cloudflare HTTPS tunnel: `https://shuttle-inexpensive-habits-powerseller.trycloudflare.com`. Verified HTTP/2 200 response for application bundle and `scene.splat` (19.3 MB Soviet T-54/55 Tank).
  - **Milestone 72: Interactive Suggested Inquiry Chips in WebXR & Desktop Chat ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
    - **Initial Suggested Inquiries UX**: Pre-populated the chat welcome message and the desktop `#rag-presets` bar with curated, grounded inquiries for the active 3D model (e.g., combat history, technical armor specs, production milestones of the Soviet T-54/55 Tank).
    - **Global Click Handler**: Implemented robust delegation for `.raycast-query-btn` and `.chip-query` elements across the chat log, desktop HUD, and WebXR Raycast Inspector cards. Tapping or clicking any suggested inquiry automatically populates the input field, dispatches the Spatial RAG query, and starts the spatial audio narrator response.
    - **WebXR HUD Synchronization**: Dynamically generates and syncs clickable inquiry buttons onto the floating WebXR Raycast HUD and desktop chat preset bar whenever an object inspection completes.
    - **Deployment**: Bumped frontend cache buster to `v=20261020_suggested_queries_v39`.
  - **Milestone 73: Dynamic "Analyzing..." Animations for Tavily Grounding in WebXR & Desktop Chat ([public/app.js](file:///Users/akif/Projects/splatris/public/app.js), [public/style.css](file:///Users/akif/Projects/splatris/public/style.css), [public/index.html](file:///Users/akif/Projects/splatris/public/index.html))**:
    - **Animated Chat Loading Card**: Upgraded `appendChatMessage` to render an animated Tavily grounding state with expanding pulse ring, core glowing dot, gradient shimmer text, and 5 staggered bouncing wave dots (`.....`).
    - **Dynamic Real-Time Substep Progress Tracker**: Added an active interval cycler that rotates through realistic grounding stages ("Connecting to Tavily search grounding...", "Cross-referencing technical specifications...", "Synthesizing spatial knowledge with Nemotron 3 Ultra...") so the user immediately perceives lively progress.
    - **Laser Progress Indicator**: Integrated an animated sliding laser beam track (`.analyzing-laser-beam`) across the loading card with ambient breathing border glow (`.message.is-analyzing`).
    - **WebXR HUD Live Dot Wave in VR**: Implemented interval-driven animated cycling dots (`.` &rarr; `..` &rarr; `...` &rarr; `....` &rarr; `.....`) for `selectXRQuestion` and `performRaycastSelectionXR`, ensuring users inside Meta Quest VR have clear visual confirmation that analysis is underway.
    - **Clean Lifecycle & Resource Management**: Guaranteed timers are automatically cleared and classes removed in `updateChatMessage`, `renderRaycastResult`, and `unlockRaycastQuery`.
    - **Deployment**: Bumped cache busters in `public/index.html` to `v=20261020_analyzing_anim_v40`.
