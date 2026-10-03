import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import { createWriteStream, createReadStream } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import crypto from 'node:crypto';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fetch as undiciFetch, Agent, FormData as UndiciFormData } from 'undici';

const MODELS_DIR = path.resolve('uploads', 'models');

// High-capacity HTTP agent for heavy GPU worker streaming (10 min timeouts, zero premature socket drops)
const gpuWorkerAgent = new Agent({
  headersTimeout: 600000, // 10 minutes
  bodyTimeout: 600000,    // 10 minutes
  connectTimeout: 60000,  // 1 minute
  keepAliveTimeout: 60000,
});

// Custom fetch wrapper utilizing gpuWorkerAgent
const workerFetch = (url, options = {}) => {
  return undiciFetch(url, {
    ...options,
    dispatcher: gpuWorkerAgent,
  });
};

/**
 * Formats low-level fetch/network failures into user-friendly, actionable diagnostic messages.
 */
function formatNetworkError(err, endpoint, context = 'connection') {
  const causeCode = err.cause?.code || err.code;
  const causeMsg = String(err.cause?.message || err.message || '');

  if (causeCode === 'ECONNREFUSED' || causeMsg.includes('ECONNREFUSED')) {
    return `Worker Offline: Connection refused at ${endpoint} (ECONNREFUSED during ${context}). The remote FastAPI worker process (splatris-worker.service) is not running on port 8000.`;
  }
  if (causeCode === 'ETIMEDOUT' || causeMsg.includes('ETIMEDOUT') || err.name === 'TimeoutError') {
    if (context === 'training submission') {
      return `Upload Timed Out: Transfer of frame archive to ${endpoint} timed out. The video archive was too large to upload within the allotted window.`;
    }
    return `Worker Unreachable: Connection timed out at ${endpoint} (ETIMEDOUT during ${context}). The Nebius GPU instance may be stopped, the public IP may have changed, or port 8000 is blocked by security firewall rules.`;
  }
  if (causeCode === 'ENOTFOUND' || causeMsg.includes('ENOTFOUND')) {
    return `Worker Unreachable: Hostname/IP not found for ${endpoint} (ENOTFOUND). Please verify NEBIUS_GPU_ENDPOINT in .env.`;
  }
  if (causeCode === 'ECONNRESET' || causeMsg.includes('ECONNRESET')) {
    return `Worker Connection Reset at ${endpoint} (ECONNRESET during ${context}). The remote process may have crashed or restarted.`;
  }
  return `Worker Offline/Unreachable at ${endpoint} (${causeCode || 'FETCH_FAILED'}: ${causeMsg || err.message})`;
}

/**
 * Streams the frame zip archive to the Nebius GPU worker with live chunk-by-chunk progress.
 * Provides real-time calculation of uploaded MBs, upload percentage, speed (MB/s), and ETA (seconds).
 */
async function uploadFramesWithLiveProgress({
  endpoint,
  zipPath,
  frameCount,
  fields,
  onProgress,
  timeoutMs = 600000,
}) {
  const url = new URL(`${endpoint}/train`);
  const isHttps = url.protocol === 'https:';
  const transport = isHttps ? https : http;

  const stat = await fs.stat(zipPath);
  const totalFileSize = stat.size;
  const boundary = '----WebKitFormBoundary' + crypto.randomBytes(16).toString('hex');

  let prefix = '';
  for (const [name, val] of Object.entries(fields)) {
    if (val !== undefined && val !== null) {
      prefix += `--${boundary}\r\n`;
      prefix += `Content-Disposition: form-data; name="${name}"\r\n\r\n`;
      prefix += `${val}\r\n`;
    }
  }

  prefix += `--${boundary}\r\n`;
  prefix += `Content-Disposition: form-data; name="frames_zip"; filename="frames.zip"\r\n`;
  prefix += `Content-Type: application/zip\r\n\r\n`;

  const suffix = `\r\n--${boundary}--\r\n`;

  const prefixBuf = Buffer.from(prefix);
  const suffixBuf = Buffer.from(suffix);
  const totalContentLength = prefixBuf.length + totalFileSize + suffixBuf.length;

  const totalMb = (totalFileSize / (1024 * 1024)).toFixed(1);

  return new Promise((resolve, reject) => {
    let uploadedBytes = 0;
    const startTime = Date.now();
    let lastProgressTime = 0;

    const req = transport.request(
      {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + (url.search || ''),
        method: 'POST',
        headers: {
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': totalContentLength,
        },
        timeout: timeoutMs,
      },
      (res) => {
        let responseBody = '';
        res.on('data', (chunk) => {
          responseBody += chunk;
        });
        res.on('end', () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            try {
              const parsed = JSON.parse(responseBody);
              resolve(parsed);
            } catch {
              resolve({ success: true, raw: responseBody });
            }
          } else {
            reject(new Error(`Nebius GPU worker rejected training job (${res.statusCode}): ${responseBody}`));
          }
        });
      }
    );

    req.on('timeout', () => {
      req.destroy(new Error(`Upload timed out after ${Math.round(timeoutMs / 1000)}s`));
    });

    req.on('error', (err) => {
      reject(err);
    });

    // Write form fields prefix
    req.write(prefixBuf);

    // Stream the zip file with chunk progress tracking
    const fileStream = createReadStream(zipPath, { highWaterMark: 128 * 1024 });

    fileStream.on('data', (chunk) => {
      uploadedBytes += chunk.length;
      req.write(chunk);

      const now = Date.now();
      // Throttle updates to every ~300ms or on completion
      if (now - lastProgressTime >= 300 || uploadedBytes === totalFileSize) {
        lastProgressTime = now;
        const pct = Math.min(100, Math.round((uploadedBytes / totalFileSize) * 100));
        const uploadedMb = (uploadedBytes / (1024 * 1024)).toFixed(1);
        const elapsedSec = Math.max(0.1, (now - startTime) / 1000);
        const speedBytes = uploadedBytes / elapsedSec;
        const speedMb = (speedBytes / (1024 * 1024)).toFixed(1);
        const remainingBytes = Math.max(0, totalFileSize - uploadedBytes);
        const etaSec = speedBytes > 0 ? Math.ceil(remainingBytes / speedBytes) : 0;

        // Progress smoothly scales from 20% to 35% during frame archive upload
        const pipelineProgress = Math.round(20 + (pct * 0.15));
        const stageMsg = `Uploading ${frameCount} frames to GPU: ${uploadedMb} / ${totalMb} MB (${pct}%) • ${speedMb} MB/s (${etaSec}s left)`;

        onProgress(pipelineProgress, stageMsg);
      }
    });

    fileStream.on('end', () => {
      req.write(suffixBuf);
      req.end();
      onProgress(35, `Frame archive received by GPU (${totalMb} MB). Queuing 3DGS pipeline...`);
    });

    fileStream.on('error', (err) => {
      req.destroy(err);
      reject(err);
    });
  });
}

/**
 * Dispatches 3DGS training to a remote Nebius GPU Worker via HTTP REST API.
 */
async function resolveExistingFramesDir(primaryDir, targetId, sceneId, videoId) {
  // 1. Check primaryDir
  if (primaryDir) {
    try {
      const st = await fs.stat(primaryDir);
      if (st.isDirectory()) {
        const files = await fs.readdir(primaryDir);
        if (files.some((f) => f.endsWith('.jpg') || f.endsWith('.png'))) return primaryDir;
      }
    } catch {}
  }

  // 2. Check candidate paths in uploads/frames
  const idsToCheck = [targetId, sceneId, videoId].filter(Boolean);
  const baseFramesDir = path.resolve('uploads', 'frames');

  for (const id of idsToCheck) {
    const candidate = path.join(baseFramesDir, id);
    try {
      const st = await fs.stat(candidate);
      if (st.isDirectory()) {
        const files = await fs.readdir(candidate);
        if (files.some((f) => f.endsWith('.jpg') || f.endsWith('.png'))) return candidate;
      }
    } catch {}
  }

  // 3. Search directory with video_meta.json or tank frames
  try {
    const entries = await fs.readdir(baseFramesDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const sub = path.join(baseFramesDir, entry.name);
      try {
        const raw = await fs.readFile(path.join(sub, 'video_meta.json'), 'utf-8');
        const meta = JSON.parse(raw);
        if (meta.originalFilename && (meta.originalFilename.includes('tank') || idsToCheck.some((id) => meta.videoId === id || meta.originalFilename === id))) {
          const files = await fs.readdir(sub);
          if (files.some((f) => f.endsWith('.jpg') || f.endsWith('.png'))) return sub;
        }
      } catch {}
    }
  } catch {}

  return primaryDir;
}

async function dispatchToRemoteWorker({
  workerEndpoint,
  jobId,
  videoId,
  sceneId,
  cachedSceneId,
  bypassColmap = false,
  framesDir,
  colmapMatcher,
  recommendedIterations,
  filteredFrameIndices,
  trainingMethod = '3dgs',
  onProgress,
}) {
  const endpoint = workerEndpoint.replace(/\/$/, '');

  // 1. Pre-flight Check: verify remote Nebius GPU worker is reachable before heavy work
  onProgress(5, 'Checking Nebius GPU worker availability...');
  try {
    const healthRes = await workerFetch(`${endpoint}/health`, {
      signal: AbortSignal.timeout(6000),
    });
    if (!healthRes.ok) {
      throw new Error(`Worker returned HTTP ${healthRes.status} ${healthRes.statusText}`);
    }
  } catch (err) {
    const errorMsg = formatNetworkError(err, endpoint, 'health check');
    console.error(`[NebiusGpuService] Health check failed:`, errorMsg);
    throw new Error(errorMsg);
  }

  // Enforce step limit guard: strictly cap iterations to 10,000
  const targetIterations = Math.min(Number(recommendedIterations) || 10000, 10000);
  const effectiveSceneId = sceneId || cachedSceneId || videoId;

  // 2. Check if the remote worker actually has cached poses for cachedSceneId
  let remoteHasCache = false;
  if (bypassColmap || cachedSceneId) {
    try {
      const cacheCheckRes = await workerFetch(`${endpoint}/cache/status/${cachedSceneId || effectiveSceneId}`, {
        signal: AbortSignal.timeout(4000),
      });
      if (cacheCheckRes.ok) {
        const cacheData = await cacheCheckRes.json();
        remoteHasCache = Boolean(cacheData.hasCache);
      }
    } catch (e) {
      console.warn('[NebiusGpuService] Remote cache status check notice:', e.message);
    }
  }

  // SMART BYPASS: Skip Step 1 (Upload) and Step 2 (COLMAP SfM) ONLY IF remote worker has cached poses!
  if (remoteHasCache) {
    onProgress(35, '⚡ Quick Convert: Re-processing scene using existing processed data...');
    console.log(`[NebiusGpuService] Smart Bypass Active for scene "${effectiveSceneId}". Skipping upload & COLMAP.`);

    const fields = {
      jobId,
      sceneId: effectiveSceneId,
      cachedSceneId: cachedSceneId || effectiveSceneId,
      colmapMatcher,
      iterations: String(targetIterations),
      densify_grad_thresh: '0.0004',
      downscale_factor: '2',
      filteredFrameIndices: Array.isArray(filteredFrameIndices) ? filteredFrameIndices.join(',') : '',
      training_method: trainingMethod || '3dgs',
    };

    const formData = new UndiciFormData();
    for (const [k, v] of Object.entries(fields)) {
      formData.append(k, v);
    }

    try {
      const trainRes = await workerFetch(`${endpoint}/train`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(30000),
      });
      if (!trainRes.ok) {
        const text = await trainRes.text().catch(() => '');
        throw new Error(`Worker rejected bypass job (${trainRes.status}): ${text}`);
      }
    } catch (err) {
      const errorMsg = formatNetworkError(err, endpoint, 'training submission (smart bypass)');
      console.error(`[NebiusGpuService] Failed to submit bypass training job:`, errorMsg);
      throw new Error(errorMsg);
    }
  } else {
    // Normal Path or Fallback (cache absent on worker, e.g. newly deployed GPU instance)
    const activeFramesDir = await resolveExistingFramesDir(framesDir, effectiveSceneId, sceneId, videoId);

    if (bypassColmap || cachedSceneId) {
      console.log(`[NebiusGpuService] Remote worker cache missing for "${effectiveSceneId}". Establishing initial cache from ${activeFramesDir}...`);
      onProgress(10, 'Initializing scene cache on new GPU instance: Packaging frames...');
    } else {
      onProgress(10, 'Packaging video frames for Nebius GPU worker...');
    }

    // Create a zip archive of the frames using child_process zip or tar
    const zipPath = path.join(path.dirname(activeFramesDir), `${jobId}_frames.zip`);
    let frameFiles = [];
    try {
      frameFiles = await fs.readdir(activeFramesDir);
    } catch (err) {
      throw new Error(`Scene cache is not present on this newly deployed GPU server, and original frames were not found in ${activeFramesDir}. Please upload the walkthrough video to process the initial space.`);
    }

    let jpgFiles = frameFiles
      .filter((f) => f.endsWith('.jpg') || f.endsWith('.png'))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

    if (jpgFiles.length === 0) {
      throw new Error(`No frame images found in ${activeFramesDir}`);
    }

    // Dynamic frame budget guard: cap to max 220 frames with uniform stride
    const MAX_FRAMES = 220;
    if (jpgFiles.length > MAX_FRAMES) {
      const total = jpgFiles.length;
      const keepIndices = new Set();
      for (let i = 0; i < MAX_FRAMES; i++) {
        keepIndices.add(Math.round((i * (total - 1)) / (MAX_FRAMES - 1)));
      }
      const toDelete = jpgFiles.filter((_, idx) => !keepIndices.has(idx));
      jpgFiles = jpgFiles.filter((_, idx) => keepIndices.has(idx));
      await Promise.all(toDelete.map((f) => fs.unlink(path.join(framesDir, f)).catch(() => {})));
      console.log(`[NebiusGpuService] Decimated frames from ${total} to ${jpgFiles.length} (cap: ${MAX_FRAMES}) for wide baseline packaging`);
    }

    // Create zip file
    await new Promise((resolve, reject) => {
      const zipProc = spawn('zip', ['-j', zipPath, ...jpgFiles.map((f) => path.join(framesDir, f))]);
      zipProc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`Failed to zip frames (code ${code})`));
      });
    });

    const fields = {
      jobId,
      sceneId: effectiveSceneId,
      colmapMatcher,
      iterations: String(targetIterations),
      densify_grad_thresh: '0.0004',
      downscale_factor: '2',
      filteredFrameIndices: Array.isArray(filteredFrameIndices) ? filteredFrameIndices.join(',') : '',
      training_method: trainingMethod || '3dgs',
    };

    try {
      await uploadFramesWithLiveProgress({
        endpoint,
        zipPath,
        frameCount: jpgFiles.length,
        fields,
        onProgress,
        timeoutMs: 600000, // 10 minutes timeout for uploading large frame archives
      });
    } catch (err) {
      const errorMsg = formatNetworkError(err, endpoint, 'training submission');
      console.error(`[NebiusGpuService] Failed to submit training job:`, errorMsg);
      throw new Error(errorMsg);
    } finally {
      // Clean up local temp zip
      await fs.unlink(zipPath).catch(() => {});
    }
  }

  onProgress(36, 'Job queued on Nebius GPU; starting live telemetry polling...');

  // Poll status until completion
  // Target model path is always .splat for native WebXR / GaussianSplats3D viewer
  const targetModelPath = path.join(MODELS_DIR, `${jobId}.splat`);
  const targetPlyPath = path.join(MODELS_DIR, `${jobId}.ply`);
  await fs.mkdir(MODELS_DIR, { recursive: true });

  let consecutivePollErrors = 0;
  const MAX_POLL_ERRORS = 8; // ~16 seconds of continuous connection drop

  while (true) {
    await new Promise((r) => setTimeout(r, 2000));

    let statusRes;
    try {
      statusRes = await workerFetch(`${endpoint}/status/${jobId}`, {
        signal: AbortSignal.timeout(8000),
      });
      consecutivePollErrors = 0;
    } catch (pollErr) {
      consecutivePollErrors++;
      console.warn(`[NebiusGpuService] Status poll failed (${consecutivePollErrors}/${MAX_POLL_ERRORS}):`, pollErr.message);
      if (consecutivePollErrors >= MAX_POLL_ERRORS) {
        throw new Error(formatNetworkError(pollErr, endpoint, 'status telemetry polling'));
      }
      continue;
    }

    if (!statusRes.ok) continue;

    const statusData = await statusRes.json().catch(() => null);
    if (!statusData) continue;

    if (statusData.status === 'processing') {
      const gpuPct = typeof statusData.progress === 'number' ? statusData.progress : 5;
      // Map GPU worker's 0-100% progress into pipeline's 36%-95% range
      const overallPct = Math.min(95, Math.max(36, Math.round(36 + (gpuPct * 0.59))));
      onProgress(overallPct, statusData.stage || 'Optimizing 3D scene on CUDA...');
    } else if (statusData.status === 'completed') {
      const activeMethod = statusData.training_method || trainingMethod;
      onProgress(95, `Downloading compiled WebXR 3D model (.splat) from Nebius GPU...`);

      let downloadRes;
      try {
        downloadRes = await workerFetch(`${endpoint}/download/${jobId}/scene.splat`, {
          signal: AbortSignal.timeout(300000),
        });
      } catch (dlErr) {
        throw new Error(formatNetworkError(dlErr, endpoint, 'model download'));
      }

      if (!downloadRes.ok) {
        throw new Error(`Failed to download compiled model from GPU worker (${downloadRes.status})`);
      }

      const dest = createWriteStream(targetModelPath);
      // @ts-ignore
      await pipeline(downloadRes.body, dest);

      // If Triangle Splatting, also download the companion Mesh PLY for 3D export
      if (activeMethod === 'triangle') {
        try {
          const plyRes = await workerFetch(`${endpoint}/download/${jobId}/scene.ply`, {
            signal: AbortSignal.timeout(300000),
          });
          if (plyRes.ok) {
            const plyDest = createWriteStream(targetPlyPath);
            // @ts-ignore
            await pipeline(plyRes.body, plyDest);
            console.log(`[NebiusService] Downloaded companion Triangle Mesh PLY: ${targetPlyPath}`);
          }
        } catch (plyErr) {
          console.warn(`[NebiusService] Notice downloading companion PLY:`, plyErr.message);
        }
      }

      // Purge intermediate /tmp/splatris_jobs/<jobId> on worker to prevent disk exhaustion
      try {
        await workerFetch(`${endpoint}/cleanup/${jobId}`, { method: 'POST', signal: AbortSignal.timeout(5000) }).catch(() => {});
        console.log(`[NebiusService] Triggered remote worker cleanup for job ${jobId}`);
      } catch (cleanErr) {
        console.warn(`[NebiusService] Remote cleanup notice:`, cleanErr.message);
      }

      onProgress(100, 'Completed');
      return targetModelPath;
    } else if (statusData.status === 'aborted') {
      throw new Error(`Job ${jobId} was aborted by user.`);
    } else if (statusData.status === 'failed') {
      throw new Error(`Nebius GPU training failed: ${statusData.error || 'Unknown worker error'}`);
    }
  }
}

/**
 * Dispatches 3DGS training locally using pipeline/train_3dgs.py.
 */
async function dispatchLocalRunner({
  jobId,
  videoId,
  sceneId,
  cachedSceneId,
  bypassColmap = false,
  framesDir,
  colmapMatcher,
  recommendedIterations,
  filteredFrameIndices,
  trainingMethod = '3dgs',
  onProgress,
}) {
  onProgress(15, 'Launching local 3DGS execution runner...');

  const outputDir = path.join(MODELS_DIR, `job_${jobId}`);
  await fs.mkdir(outputDir, { recursive: true });

  const scriptPath = path.resolve('pipeline', 'train_3dgs.py');
  const targetSplatPath = path.join(MODELS_DIR, `${jobId}.splat`);

  const args = [
    scriptPath,
    '--images_dir', framesDir || '',
    '--output_dir', outputDir,
    '--matcher', colmapMatcher,
    '--iterations', String(recommendedIterations),
    '--training_method', trainingMethod,
    '--output_splat', `${jobId}.splat`,
    '--output_ply', `${jobId}.ply`,
  ];

  if (sceneId || videoId) args.push('--scene_id', sceneId || videoId);
  if (cachedSceneId) args.push('--cached_scene_id', cachedSceneId);

  if (Array.isArray(filteredFrameIndices) && filteredFrameIndices.length > 0) {
    args.push('--frame_indices', filteredFrameIndices.join(','));
  }

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', args, {
      env: { ...process.env },
    });

    let stderrBuffer = '';

    proc.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');
      for (const line of lines) {
        const lineStr = line.trim();
        if (!lineStr) continue;

        if (lineStr.includes('[STAGE:')) {
          try {
            const stagePart = lineStr.split('[STAGE:')[1].split(']')[0];
            const progPart = lineStr.split('[PROGRESS:')[1].split('%]')[0];
            const details = lineStr.split('%]')[1]?.trim() || '';
            const stageName = stagePart.replace(/_/g, ' ').toLowerCase();
            const formattedStage = stageName.charAt(0).toUpperCase() + stageName.slice(1);
            onProgress(parseInt(progPart, 10), `${formattedStage}: ${details}`);
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    });

    proc.stderr.on('data', (data) => {
      stderrBuffer += data.toString();
    });

    proc.on('close', async (code) => {
      if (code === 0) {
        const isTriangle = trainingMethod === 'triangle';
        const generatedModel = isTriangle
          ? path.join(outputDir, `${jobId}.ply`)
          : path.join(outputDir, `${jobId}.splat`);
        const targetModelPath = isTriangle
          ? path.join(MODELS_DIR, `${jobId}.ply`)
          : path.join(MODELS_DIR, `${jobId}.splat`);

        try {
          await fs.copyFile(generatedModel, targetModelPath);
          onProgress(100, 'Completed');
          resolve(targetModelPath);
        } catch (err) {
          reject(new Error(`Trained model file not found at ${generatedModel}: ${err.message}`));
        }
      } else {
        const errLines = stderrBuffer.trim().split('\n');
        const lastErr = errLines.slice(-3).join(' ') || `Process exited with code ${code}`;
        reject(new Error(`3DGS training failed: ${lastErr}`));
      }
    });

    proc.on('error', (err) => {
      reject(new Error(`Failed to spawn Python 3DGS runner: ${err.message}`));
    });
  });
}

/**
 * Main entry point for executing real 3DGS training.
 * @param {object} params
 * @param {string} params.jobId - Unique job UUID.
 * @param {string} params.videoId - Target video ID (frames in uploads/frames/{videoId}).
 * @param {string} [params.sceneId] - Scene ID for COLMAP caching.
 * @param {string} [params.cachedSceneId] - Scene ID to reuse cached COLMAP poses.
 * @param {boolean} [params.bypassColmap=false] - Whether to bypass upload and COLMAP.
 * @param {string} [params.colmapMatcher="sequential"] - "sequential" or "exhaustive".
 * @param {number} [params.recommendedIterations=7000] - 3DGS iterations.
 * @param {number[]} [params.filteredFrameIndices=[]] - Selected frames.
 * @param {string} [params.trainingMethod="3dgs"] - '3dgs' or 'triangle'.
 * @param {function} params.onProgress - Callback (progress: number, stage: string).
 * @returns {Promise<string>} Path to the resulting binary .splat or .ply file.
 */
export async function runNebiusGpuTraining({
  jobId,
  videoId,
  sceneId,
  cachedSceneId,
  bypassColmap = false,
  colmapMatcher = 'sequential',
  recommendedIterations = 7000,
  filteredFrameIndices = [],
  trainingMethod = '3dgs',
  onProgress = () => { },
}) {
  const effectiveSceneId = sceneId || cachedSceneId || videoId;
  const framesDir = await resolveExistingFramesDir(
    path.resolve('uploads', 'frames', videoId || effectiveSceneId),
    effectiveSceneId,
    sceneId,
    videoId
  );

  // Validate that frames exist only if not bypassing with existing COLMAP cache
  if (!bypassColmap && !cachedSceneId) {
    try {
      const stats = await fs.stat(framesDir);
      if (!stats.isDirectory()) {
        throw new Error(`Frames directory not found: ${framesDir}`);
      }
    } catch (err) {
      throw new Error(`Video frames directory does not exist for videoId "${videoId}": ${err.message}`);
    }
  }

  const workerEndpoint = process.env.NEBIUS_GPU_ENDPOINT;

  if (workerEndpoint) {
    console.log(`[Nebius GPU] Dispatching job ${jobId} (${trainingMethod}, bypass=${bypassColmap}) to remote worker at ${workerEndpoint}`);
    return await dispatchToRemoteWorker({
      workerEndpoint,
      jobId,
      videoId,
      sceneId: effectiveSceneId,
      cachedSceneId: cachedSceneId || (bypassColmap ? effectiveSceneId : null),
      bypassColmap,
      framesDir,
      colmapMatcher,
      recommendedIterations,
      filteredFrameIndices,
      trainingMethod,
      onProgress,
    });
  }

  console.log(`[Nebius GPU] Running execution runner locally for job ${jobId} (${trainingMethod}, bypass=${bypassColmap})`);
  return await dispatchLocalRunner({
    jobId,
    videoId,
    sceneId: effectiveSceneId,
    cachedSceneId: cachedSceneId || (bypassColmap ? effectiveSceneId : null),
    bypassColmap,
    framesDir,
    colmapMatcher,
    recommendedIterations,
    filteredFrameIndices,
    trainingMethod,
    onProgress,
  });
}

/**
 * Aborts a running 3DGS training job on Nebius GPU.
 * Calls worker /abort/:jobId and triggers SSH kill switch as backup to ensure all COLMAP/Python processes are terminated.
 * @param {string} jobId
 */
export async function abortNebiusGpuJob(jobId) {
  const workerEndpoint = process.env.NEBIUS_GPU_ENDPOINT;
  const results = { remoteAborted: false, sshTerminated: false };

  if (workerEndpoint) {
    const endpoint = workerEndpoint.replace(/\/$/, '');
    try {
      const res = await workerFetch(`${endpoint}/abort/${jobId}`, {
        method: 'POST',
        signal: AbortSignal.timeout(8000),
      });
      if (res.ok) {
        results.remoteAborted = true;
        console.log(`[NebiusGpuService] Remote worker acknowledged abort for job ${jobId}`);
      }
    } catch (err) {
      console.warn(`[NebiusGpuService] HTTP abort endpoint warning for job ${jobId}: ${err.message}`);
    }
  }

  // Backup SSH kill switch: guarantee immediate termination of active processes and disk purge
  try {
    const workerIp = (process.env.NEBIUS_GPU_ENDPOINT || 'http://195.242.13.40:8000')
      .replace('http://', '')
      .replace(':8000', '');
    const { exec } = await import('node:child_process');
    exec(
      `ssh -o StrictHostKeyChecking=no ubuntu@${workerIp} "pkill -9 -f ${jobId} || true; pkill -9 -f train_3dgs.py || true; rm -rf /tmp/splatris_jobs/${jobId}"`,
      (err) => {
        if (err) console.warn('[NebiusGpuService] SSH abort kill switch warning:', err.message);
        else {
          results.sshTerminated = true;
          console.log(`[NebiusGpuService] Remote worker processes terminated and directory purged via SSH for job ${jobId}`);
        }
      }
    );
  } catch (sshErr) {
    console.warn('[NebiusGpuService] SSH kill switch error:', sshErr.message);
  }

  return results;
}

export default {
  runNebiusGpuTraining,
  abortNebiusGpuJob,
};

