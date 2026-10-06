import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import { runNebiusGpuTraining, abortNebiusGpuJob } from './nebiusGpuService.js';

const MODELS_DIR = path.resolve('uploads', 'models');
const DEFAULT_SPLAT_PATH = path.join(MODELS_DIR, 'scene.splat');

// In-memory store for 3DGS training jobs
const jobs = new Map();

/**
 * Searches for an existing active (processing or suspended) job matching a sceneId or videoId.
 * @param {string} sceneIdOrVideoId
 * @returns {object|null}
 */
export function findActiveJobForScene(sceneIdOrVideoId) {
  if (!sceneIdOrVideoId) return null;
  for (const job of jobs.values()) {
    const matchesScene =
      job.sceneId === sceneIdOrVideoId ||
      job.videoId === sceneIdOrVideoId ||
      job.cachedSceneId === sceneIdOrVideoId;
    if (matchesScene && (job.status === 'processing' || job.status === 'suspended')) {
      return job;
    }
  }
  return null;
}

/**
 * Creates and launches a real 3D Gaussian Splatting training job on Nebius GPU.
 * @param {object} params
 * @param {string} params.videoId - Target video identifier (frames stored in uploads/frames/{videoId}).
 * @param {string} [params.colmapMatcher="sequential"] - "sequential" or "exhaustive".
 * @param {number} [params.recommendedIterations=10000] - 3DGS training iterations (capped at 10000).
 * @param {number[]} [params.filteredFrameIndices=[]] - Selected frames for reconstruction.
 * @returns {object} Initial job object.
 */
export function createJob({
  videoId,
  originalFilename = null,
  deviceName = null,
  uploadTimestamp = null,
  colmapMatcher = 'vocab_tree',
  recommendedIterations = 10000,
  filteredFrameIndices = [],
  trainingMethod = '3dgs',
  bypassColmap = false,
  cachedSceneId = null,
  sceneId = null,
}) {
  const jobId = crypto.randomUUID();
  const method = ['triangle', 'image2splat'].includes(trainingMethod) ? trainingMethod : '3dgs';
  const outputExt = 'splat'; // GaussianSplats3D viewer natively renders .splat; Mesh .ply is preserved as companion asset
  const outputModelUrl = `/uploads/models/${jobId}.${outputExt}`;

  // Step Limit Guard: strictly cap iterations to 10,000
  const clampedIterations = Math.min(Number(recommendedIterations) || 10000, 10000);

  const isBypass = Boolean(bypassColmap || cachedSceneId);
  const effectiveCachedId = cachedSceneId || (isBypass ? (sceneId || videoId) : null);
  const effectiveSceneId = sceneId || effectiveCachedId || videoId;

  // 1. Backend Active Job Protection: Check if a job is already processing or suspended for this scene
  const existingJob = findActiveJobForScene(effectiveSceneId);
  if (existingJob) {
    console.log(`[JobService] Active job ${existingJob.jobId} (${existingJob.status}, ${existingJob.progress}%) already running for scene "${effectiveSceneId}". Returning existing job.`);
    return { ...existingJob, isExisting: true };
  }

  const job = {
    jobId,
    videoId: videoId || 'room-scan',
    sceneId: effectiveSceneId,
    cachedSceneId: effectiveCachedId,
    bypassColmap: isBypass,
    originalFilename: originalFilename || (videoId ? `${videoId}.mp4` : 'Walkthrough.mp4'),
    deviceName: deviceName || 'Mobile Camera',
    uploadTimestamp: uploadTimestamp || new Date().toISOString(),
    colmapMatcher,
    recommendedIterations: clampedIterations,
    filteredFrameIndices,
    trainingMethod: method,
    status: 'processing', // 'processing' | 'completed' | 'failed' | 'aborted'
    stage: isBypass
      ? '⚡ COLMAP Cache Hit! Skipping upload & feature extraction...'
      : 'Submitting job to Nebius GPU pipeline',
    progress: isBypass ? 35 : 5,
    outputModelUrl,
    error: null,
    createdAt: new Date().toISOString(),
    completedAt: null,
  };

  // If videoId was uploaded, attempt to load any cached video_meta.json
  if (videoId && !isBypass) {
    const metaPath = path.resolve('uploads', 'frames', videoId, 'video_meta.json');
    fs.readFile(metaPath, 'utf-8').then((raw) => {
      try {
        const parsed = JSON.parse(raw);
        if (parsed.originalFilename) job.originalFilename = parsed.originalFilename;
        if (parsed.deviceName) job.deviceName = parsed.deviceName;
        if (parsed.uploadTimestamp) job.uploadTimestamp = parsed.uploadTimestamp;
      } catch { }
    }).catch(() => { });
  }

  jobs.set(jobId, job);

  // Launch the real Nebius GPU worker asynchronously
  executeGpuJobAsync(job);

  return job;
}

/**
 * Executes the real training pipeline and streams live status updates into the job record.
 */
async function executeGpuJobAsync(job) {
  try {
    await fs.mkdir(MODELS_DIR, { recursive: true });

    const modelPath = await runNebiusGpuTraining({
      jobId: job.jobId,
      videoId: job.videoId,
      sceneId: job.sceneId || job.videoId,
      cachedSceneId: job.cachedSceneId,
      bypassColmap: job.bypassColmap,
      colmapMatcher: job.colmapMatcher,
      recommendedIterations: job.recommendedIterations,
      filteredFrameIndices: job.filteredFrameIndices,
      trainingMethod: job.trainingMethod || '3dgs',
      onProgress: (progress, stage) => {
        if (job.status === 'aborted') return;
        job.progress = progress;
        job.stage = stage;
      },
    });

    if (job.status === 'aborted') return;

    job.status = 'completed';
    job.stage = 'Completed';
    job.progress = 100;
    job.completedAt = new Date().toISOString();
    job.outputModelUrl = `/uploads/models/${path.basename(modelPath)}`;

    // Persist model metadata sidecar (.json) for meaningful naming in Recent Scenes
    const modelMeta = {
      jobId: job.jobId,
      originalFilename: job.originalFilename || 'Walkthrough.mp4',
      deviceName: job.deviceName || 'Mobile Camera',
      uploadTimestamp: job.uploadTimestamp || job.createdAt,
      trainingMethod: job.trainingMethod || '3dgs',
      completedAt: job.completedAt,
    };
    await fs.writeFile(
      path.join(MODELS_DIR, `${job.jobId}.json`),
      JSON.stringify(modelMeta, null, 2)
    ).catch((err) => {
      console.warn('Notice: Could not write model metadata sidecar:', err.message);
    });

    // Also update default scene.splat or scene.ply to the newly trained model
    if (modelPath.endsWith('.splat')) {
      await fs.copyFile(modelPath, DEFAULT_SPLAT_PATH).catch((err) => {
        console.warn('Notice: Could not update default scene.splat:', err.message);
      });
    } else if (modelPath.endsWith('.ply')) {
      const defaultPlyPath = path.join(MODELS_DIR, 'scene.ply');
      await fs.copyFile(modelPath, defaultPlyPath).catch((err) => {
        console.warn('Notice: Could not update default scene.ply:', err.message);
      });
    }

    console.log(`[JobRunner] Job ${job.jobId} (${job.trainingMethod}, ${job.deviceName}, ${job.originalFilename}) successfully finished. Model saved at ${modelPath}`);

    // Automatic Post-Training Cleanup & Rolling Retention
    await cleanupPostTraining(job);
  } catch (error) {
    if (job.status === 'aborted') {
      console.log(`[JobRunner] Job ${job.jobId} was aborted by user. Skipping failure handler.`);
      return;
    }

    const errMsg = String(error.message || '');
    const isNetworkDrop =
      errMsg.includes('ETIMEDOUT') ||
      errMsg.includes('ECONNREFUSED') ||
      errMsg.includes('Worker Unreachable') ||
      errMsg.includes('Worker Offline') ||
      errMsg.includes('TimeoutError') ||
      errMsg.includes('FETCH_FAILED') ||
      errMsg.includes('ECONNRESET');

    if (isNetworkDrop) {
      console.warn(`[JobRunner] Preemptible GPU worker drop detected for job ${job.jobId}:`, errMsg);
      job.status = 'suspended';
      job.stage = 'GPU Instance Preempted/Offline';
      job.error = errMsg;
      // Do NOT run cleanup; retain extracted frames and state so the job can seamlessly resume!
      return;
    }

    console.error(`[JobRunner] Job ${job.jobId} failed:`, errMsg);
    job.status = 'failed';
    job.stage = 'Training Failed';
    job.error = errMsg || '3DGS training failed';
    job.completedAt = new Date().toISOString();
  }
}

/**
 * Automatically purges raw video, extracted frames, remote worker intermediates,
 * and maintains a rolling retention of the last 3 .splat models.
 */
export async function cleanupPostTraining(job) {
  console.log(`[JobService] Executing post-training cleanup for video: ${job.videoId}, job: ${job.jobId} (isBypass: ${job.bypassColmap})`);

  // 1. Delete extracted frames directory on local server (only if not a cached bypass job)
  if (job.videoId && !job.bypassColmap) {
    const framesDir = path.resolve('uploads', 'frames', job.videoId);
    await fs.rm(framesDir, { recursive: true, force: true }).catch((err) => {
      console.warn(`[Cleanup] Notice removing frames dir ${framesDir}:`, err.message);
    });

    // 2. Delete raw video from uploads/
    const uploadsDir = path.resolve('uploads');
    try {
      const files = await fs.readdir(uploadsDir);
      for (const file of files) {
        if (file.startsWith(job.videoId)) {
          const filePath = path.join(uploadsDir, file);
          await fs.unlink(filePath).catch(() => { });
          console.log(`[Cleanup] Deleted raw video: ${file}`);
        }
      }
    } catch (err) {
      console.warn('[Cleanup] Error deleting raw video:', err.message);
    }
  }

  // 3. Purge remote Nebius GPU worker intermediate directory via SSH
  try {
    const workerIp = (process.env.NEBIUS_GPU_ENDPOINT || 'http://195.242.13.40:8000')
      .replace('http://', '')
      .replace(':8000', '');
    const { exec } = await import('node:child_process');
    exec(`ssh -o StrictHostKeyChecking=no ubuntu@${workerIp} "rm -rf /tmp/splatris_jobs/${job.jobId}"`, (err) => {
      if (err) console.warn('[Cleanup] Remote SSH cleanup notice:', err.message);
      else console.log(`[Cleanup] Remote worker /tmp/splatris_jobs/${job.jobId} purged via SSH`);
    });
  } catch (e) { }

  // 4. Prune models in uploads/models/, keeping only the last 3
  await pruneOldModels(3);
}

/**
 * Keeps only the last N .splat models in uploads/models/, automatically pruning older ones.
 */
export async function pruneOldModels(keepCount = 3) {
  try {
    await fs.mkdir(MODELS_DIR, { recursive: true });
    const entries = await fs.readdir(MODELS_DIR, { withFileTypes: true });
    const jobSplats = [];

    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith('.splat') || entry.name === 'scene.splat') continue;
      const filePath = path.join(MODELS_DIR, entry.name);
      const stat = await fs.stat(filePath);
      jobSplats.push({ filePath, name: entry.name, mtime: stat.mtime });
    }

    // Sort newest first
    jobSplats.sort((a, b) => b.mtime.getTime() - a.mtime.getTime());

    // Prune models past the keepCount limit
    const toDelete = jobSplats.slice(keepCount);
    for (const item of toDelete) {
      await fs.unlink(item.filePath).catch(() => { });
      console.log(`[JobService] Pruned older model: ${item.name}`);
    }

    // Clean up any legacy job_ directories in uploads/models/
    for (const entry of entries) {
      if (entry.isDirectory() && entry.name.startsWith('job_')) {
        await fs.rm(path.join(MODELS_DIR, entry.name), { recursive: true, force: true }).catch(() => { });
      }
    }
  } catch (err) {
    console.warn('[JobService] Error during model pruning:', err.message);
  }
}

function formatDateTimeShort(isoString) {
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return 'Recent';
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[d.getMonth()];
    const day = d.getDate();
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    return `${month} ${day}, ${hours}:${minutes}`;
  } catch {
    return 'Recent';
  }
}

/**
 * Returns metadata for the last models in uploads/models/ categorized by spaces (video) and objects (image).
 */
export async function getRecentCategorizedModels(limit = 3) {
  try {
    await fs.mkdir(MODELS_DIR, { recursive: true });
    const entries = await fs.readdir(MODELS_DIR, { withFileTypes: true });
    const splatFiles = [];

    // Map sidecar JSON metadata by jobId/baseName
    const sidecars = new Map();
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith('.json')) {
        try {
          const raw = await fs.readFile(path.join(MODELS_DIR, entry.name), 'utf-8');
          const data = JSON.parse(raw);
          const base = entry.name.replace('.json', '');
          sidecars.set(base, data);
        } catch { }
      }
    }

    for (const entry of entries) {
      if (!entry.isFile() || (!entry.name.endsWith('.splat') && !entry.name.endsWith('.ply'))) continue;
      const filePath = path.join(MODELS_DIR, entry.name);
      const stat = await fs.stat(filePath);
      const sizeBytes = stat.size;
      const isPly = entry.name.endsWith('.ply');
      const splatCount = isPly ? Math.floor(sizeBytes / 45) : Math.floor(sizeBytes / 32);
      const sizeMb = (sizeBytes / (1024 * 1024)).toFixed(1);

      const baseName = entry.name.replace(/\.(splat|ply)$/, '');
      const meta = sidecars.get(baseName) || {};

      const originalFilename = meta.originalFilename || (baseName === 'scene' ? 'scene.splat' : 'Walkthrough.mp4');
      const deviceName = meta.deviceName || 'Mobile Camera';
      const timestamp = meta.uploadTimestamp || stat.mtime.toISOString();
      const dateStr = formatDateTimeShort(timestamp);

      // Card Title: [Date], [Time] - [Original Filename] (e.g., "Sep 25, 10:13 - IMG_9390.MOV")
      const title = `${dateStr} - ${originalFilename}`;
      // Card Subtitle: [Device Name] • [Splat Count] (e.g., "iPhone 11 Pro • 1,570,303 splats")
      const subtitle = `${deviceName} • ${splatCount.toLocaleString()} splats`;

      const isImage = meta.trainingMethod === 'image2splat' || (deviceName && deviceName.includes('Single Image'));
      const isVideo = !isImage;
      const trainingMethod = meta.trainingMethod || (isPly ? 'triangle' : '3dgs');

      const chunksPath = path.join(MODELS_DIR, `${baseName}.chunks.bin`);
      let chunksUrl = null;
      try {
        await fs.access(chunksPath);
        chunksUrl = `/uploads/models/${baseName}.chunks.bin`;
      } catch { }

      splatFiles.push({
        filename: entry.name,
        name: baseName,
        jobId: meta.jobId || baseName,
        title,
        subtitle,
        originalFilename,
        deviceName,
        trainingMethod,
        isImage,
        isVideo,
        url: `/uploads/models/${entry.name}`,
        chunksUrl,
        format: isPly ? 'ply' : 'splat',
        sizeBytes,
        sizeFormatted: `${sizeMb} MB`,
        splatCount,
        timestamp,
        isDefault: entry.name === 'scene.splat' || entry.name === 'scene.ply',
      });
    }

    // Deduplicate by baseName: if both .splat and .ply exist for the same scene, prioritize .splat for 3D viewing
    const modelsByBase = new Map();
    for (const model of splatFiles) {
      if (model.filename === 'scene.splat' || model.filename === 'scene.ply') continue;
      const existing = modelsByBase.get(model.name);
      if (!existing) {
        modelsByBase.set(model.name, model);
      } else if (existing.format === 'ply' && model.format === 'splat') {
        // Upgrade to .splat since it directly renders in GaussianSplats3D
        model.meshPlyUrl = existing.url;
        if (!model.chunksUrl && existing.chunksUrl) model.chunksUrl = existing.chunksUrl;
        modelsByBase.set(model.name, model);
      } else if (existing.format === 'splat' && model.format === 'ply') {
        existing.meshPlyUrl = model.url;
        if (!existing.chunksUrl && model.chunksUrl) existing.chunksUrl = model.chunksUrl;
      }
    }

    const jobSplats = Array.from(modelsByBase.values());
    const defaultSpace = splatFiles.find((m) => m.name === 'scene' && m.format === 'splat');

    const allModels = [...jobSplats];
    if (defaultSpace) {
      const isAlreadyPresent = allModels.some(
        (m) => m.name === 'scene' || (defaultSpace.jobId && m.jobId === defaultSpace.jobId) || (m.title === defaultSpace.title && m.sizeFormatted === defaultSpace.sizeFormatted)
      );
      if (!isAlreadyPresent) {
        allModels.push(defaultSpace);
      }
    }
    allModels.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    // Deduplicate models to guarantee no twin cards appear in recent spaces
    const seenKeys = new Set();
    const uniqueModels = [];
    for (const m of allModels) {
      const key = `${m.jobId || m.name}_${m.title}`;
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        uniqueModels.push(m);
      }
    }

    // Default first item to Standard 3DGS so it is active when opening the site first
    uniqueModels.sort((a, b) => {
      const aIs3dgs = (a.trainingMethod === '3dgs' || (!a.isImage && a.format === 'splat' && a.trainingMethod !== 'triangle')) ? 1 : 0;
      const bIs3dgs = (b.trainingMethod === '3dgs' || (!b.isImage && b.format === 'splat' && b.trainingMethod !== 'triangle')) ? 1 : 0;
      if (aIs3dgs !== bIs3dgs) return bIs3dgs - aIs3dgs;
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    });

    let videoModels = uniqueModels.filter((m) => !m.isImage).slice(0, limit);
    const imageModels = uniqueModels.filter((m) => m.isImage).slice(0, limit);

    // If videoModels is empty and defaultSpace exists, ensure defaultSpace is present
    if (videoModels.length === 0 && defaultSpace) {
      videoModels = [defaultSpace];
    }

    return {
      videoModels,
      imageModels,
      models: uniqueModels.slice(0, limit * 2),
    };
  } catch (err) {
    console.error('[JobService] Error retrieving categorized models:', err);
    return { videoModels: [], imageModels: [], models: [] };
  }
}

/**
 * Retrieves the most recent 3DGS models from the models directory.
 * @param {number} limit
 * @returns {Promise<Array<object>>}
 */
export async function getRecentModels(limit = 3) {
  const { models } = await getRecentCategorizedModels(limit);
  return models.slice(0, limit);
}

/**
 * Retrieves metadata for a specific scene or model ID.
 * @param {string} sceneId
 * @returns {Promise<object|null>}
 */
export async function getModelMeta(sceneId) {
  if (!sceneId) return null;
  const cleanId = String(sceneId).replace(/\.(splat|ply|json)$/, '');

  try {
    await fs.mkdir(MODELS_DIR, { recursive: true });

    // 1. Direct match on {cleanId}.json
    const directPath = path.join(MODELS_DIR, `${cleanId}.json`);
    try {
      const raw = await fs.readFile(directPath, 'utf-8');
      const data = JSON.parse(raw);
      const isImage = data.trainingMethod === 'image2splat' || (data.deviceName && data.deviceName.includes('Single Image'));
      return {
        ...data,
        sceneId: cleanId,
        isImage,
        isVideo: !isImage,
        found: true,
      };
    } catch { }

    // 2. Search all json sidecars in MODELS_DIR
    const files = await fs.readdir(MODELS_DIR);
    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      try {
        const raw = await fs.readFile(path.join(MODELS_DIR, file), 'utf-8');
        const data = JSON.parse(raw);
        if (data.jobId === cleanId || file.startsWith(cleanId) || file.replace('.json', '') === cleanId) {
          const isImage = data.trainingMethod === 'image2splat' || (data.deviceName && data.deviceName.includes('Single Image'));
          return {
            ...data,
            sceneId: data.jobId || file.replace('.json', ''),
            isImage,
            isVideo: !isImage,
            found: true,
          };
        }
      } catch { }
    }

    // 3. If sceneId is 'scene' or default, find the newest active model
    if (cleanId === 'scene') {
      const recent = await getRecentModels(1);
      if (recent.length > 0) {
        return {
          ...recent[0],
          sceneId: recent[0].jobId || recent[0].name,
          found: true,
        };
      }
    }
  } catch (err) {
    console.warn(`[JobService] Notice retrieving metadata for scene "${sceneId}":`, err.message);
  }

  return null;
}

/**
 * Re-processes an existing video scene using a different pipeline (e.g. Triangle Splatting)
 * by bypassing Step 1 (Upload) and Step 2 (COLMAP SfM), injecting cached poses directly into the training worker.
 *
 * @param {object} params
 * @param {string} params.sceneId - The unique scene ID of the existing model.
 * @param {string} params.trainingMethod - Target pipeline ("triangle" or "3dgs").
 * @returns {Promise<object>} The newly launched job object.
 */
export async function reprocessJob({ sceneId, trainingMethod }) {
  if (!sceneId) {
    throw new Error('Missing required "sceneId" parameter for re-processing.');
  }

  const targetMethod = String(trainingMethod || 'triangle').toLowerCase().trim();
  if (targetMethod === 'image2splat') {
    throw new Error('Cannot reprocess video into single-image Image2Splat pipeline.');
  }

  // 1. Image Isolation Guard: Verify this scene is a video and NOT a single-image scene
  const meta = await getModelMeta(sceneId);
  if (meta) {
    if (meta.trainingMethod === 'image2splat' || meta.isImage) {
      throw new Error('Single images cannot be re-processed into video-based pipelines. Image2Splat is isolated.');
    }
  }

  const effectiveSceneId = meta?.jobId || meta?.sceneId || sceneId;
  const originalFilename = meta?.originalFilename || `${effectiveSceneId}.mp4`;
  const deviceName = meta?.deviceName || 'Mobile Camera';

  // 2. Active Job Protection: Check existing before initiating new training
  const existingJob = findActiveJobForScene(effectiveSceneId);
  if (existingJob) {
    console.log(`[JobService] Active job ${existingJob.jobId} already active for scene "${effectiveSceneId}". Returning existing job.`);
    return { ...existingJob, isExisting: true };
  }

  console.log(`[JobService] Initiating Fast Re-processing for scene "${effectiveSceneId}" -> Pipeline: "${targetMethod}"`);

  // 3. Launch new job with COLMAP bypass & cachedSceneId
  const job = createJob({
    videoId: effectiveSceneId,
    sceneId: effectiveSceneId,
    cachedSceneId: effectiveSceneId,
    bypassColmap: true,
    originalFilename,
    deviceName,
    trainingMethod: targetMethod,
    recommendedIterations: 10000,
  });

  return job;
}

/**
 * Retrieves the status of a job by jobId OR sceneId / videoId.
 * @param {string} jobIdOrSceneId
 * @returns {object|null}
 */
export function getJob(jobIdOrSceneId) {
  if (!jobIdOrSceneId) return null;
  let job = jobs.has(jobIdOrSceneId) ? jobs.get(jobIdOrSceneId) : null;

  if (!job) {
    // 2. Match by sceneId, videoId, or cachedSceneId (prioritizing active jobs)
    const matches = [];
    for (const j of jobs.values()) {
      if (
        j.sceneId === jobIdOrSceneId ||
        j.videoId === jobIdOrSceneId ||
        j.cachedSceneId === jobIdOrSceneId
      ) {
        matches.push(j);
      }
    }

    if (matches.length > 0) {
      matches.sort((a, b) => {
        const aActive = ['processing', 'suspended'].includes(a.status) ? 1 : 0;
        const bActive = ['processing', 'suspended'].includes(b.status) ? 1 : 0;
        if (aActive !== bActive) return bActive - aActive;
        return new Date(b.createdAt) - new Date(a.createdAt);
      });
      job = matches[0];
    }
  }

  // 3. Disk-backed fallback: if output model exists on disk, resolve as completed
  const targetId = job ? job.jobId : jobIdOrSceneId;
  const plyPath = path.join(MODELS_DIR, `${targetId}.ply`);
  const splatPath = path.join(MODELS_DIR, `${targetId}.splat`);
  const jsonPath = path.join(MODELS_DIR, `${targetId}.json`);

  if (fsSync.existsSync(plyPath) || fsSync.existsSync(splatPath)) {
    const isPly = fsSync.existsSync(plyPath);
    let meta = {};
    try {
      if (fsSync.existsSync(jsonPath)) {
        meta = JSON.parse(fsSync.readFileSync(jsonPath, 'utf-8'));
      }
    } catch (e) { }

    if (job) {
      if (job.status !== 'completed') {
        job.status = 'completed';
        job.stage = 'Completed';
        job.progress = 100;
        job.error = null;
        job.outputModelUrl = isPly ? `/uploads/models/${targetId}.ply` : `/uploads/models/${targetId}.splat`;
      }
      return job;
    }

    return {
      jobId: targetId,
      sceneId: meta.sceneId || targetId,
      videoId: meta.videoId || targetId,
      status: 'completed',
      stage: 'Completed',
      progress: 100,
      trainingMethod: meta.trainingMethod || (isPly ? 'triangle' : '3dgs'),
      error: null,
      outputModelUrl: isPly ? `/uploads/models/${targetId}.ply` : `/uploads/models/${targetId}.splat`,
      createdAt: meta.uploadTimestamp || new Date().toISOString(),
      completedAt: meta.completedAt || new Date().toISOString(),
    };
  }

  return job;
}

/**
 * Resumes a suspended or preempted training job on Nebius GPU.
 * Utilizes existing COLMAP cache bypass to skip re-uploading and SfM.
 * @param {object} params
 * @param {string} [params.jobId]
 * @param {string} [params.sceneId]
 * @returns {Promise<object>}
 */
export async function resumeJob({ jobId, sceneId }) {
  const targetId = jobId || sceneId;
  if (!targetId) {
    throw new Error('Missing required "jobId" or "sceneId" parameter to resume.');
  }

  const job = getJob(targetId);
  if (!job) {
    throw new Error(`No training job found for "${targetId}".`);
  }

  if (job.status === 'completed') {
    return { success: true, message: 'Job is already completed.', job };
  }

  // Verify worker availability before resuming
  const workerEndpoint = process.env.NEBIUS_GPU_ENDPOINT || 'http://195.242.13.85:8000';
  const cleanEndpoint = workerEndpoint.replace(/\/$/, '');

  try {
    const healthRes = await fetch(`${cleanEndpoint}/health`, { signal: AbortSignal.timeout(4000) });
    if (!healthRes.ok) {
      throw new Error(`Worker returned HTTP ${healthRes.status}`);
    }
  } catch (err) {
    throw new Error(`Nebius GPU worker is currently unreachable at ${cleanEndpoint} (${err.message}). Please verify the GPU instance is running and port 8000 is open.`);
  }

  console.log(`[JobService] Resuming job ${job.jobId} for scene "${job.sceneId}" on ${cleanEndpoint}...`);

  job.status = 'processing';
  job.stage = 'Resuming training on Nebius GPU (Checking cache bypass)...';
  job.error = null;
  job.bypassColmap = true;
  job.cachedSceneId = job.cachedSceneId || job.sceneId || job.videoId;

  // Re-launch execution runner asynchronously
  executeGpuJobAsync(job);

  return {
    success: true,
    jobId: job.jobId,
    sceneId: job.sceneId,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    message: 'Training resumed successfully on Nebius GPU worker.',
  };
}

/**
 * Immediately aborts a running 3DGS training job, terminates remote processes on Nebius GPU,
 * and clears intermediate disk files.
 * @param {string} jobId
 */
export async function abortJob(jobId) {
  const job = jobs.get(jobId);
  console.log(`[JobService] Aborting job ${jobId}...`);

  if (job) {
    job.status = 'aborted';
    job.stage = 'Aborted by User';
    job.progress = 0;
    job.error = 'Job cancelled by user request.';
    job.completedAt = new Date().toISOString();
  }

  // Abort on Nebius GPU instance (HTTP worker /abort endpoint + SSH process kill switch)
  await abortNebiusGpuJob(jobId);

  // Clean up local intermediate frames and uploads for this job/video
  if (job?.videoId) {
    const framesDir = path.resolve('uploads', 'frames', job.videoId);
    await fs.rm(framesDir, { recursive: true, force: true }).catch(() => { });
    const uploadsDir = path.resolve('uploads');
    try {
      const files = await fs.readdir(uploadsDir);
      for (const file of files) {
        if (file.startsWith(job.videoId)) {
          await fs.unlink(path.join(uploadsDir, file)).catch(() => { });
          console.log(`[Cleanup] Deleted video on abort: ${file}`);
        }
      }
    } catch (e) { }
  }

  return { success: true, jobId, status: 'aborted' };
}

/**
 * Deletes a scene/job record from memory/database and wipes all associated files from storage:
 * - Model files (.splat, .ply, .json, .bak)
 * - Original uploaded video/image from uploads/
 * - Extracted frames & cached COLMAP data (images/, sparse/0/) from uploads/frames/
 * - Purges remote Nebius GPU worker temporary directories
 *
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export async function deleteScene(sceneId) {
  if (!sceneId) {
    throw new Error('Missing required sceneId to delete.');
  }

  const cleanId = String(sceneId).replace(/\.(splat|ply|json|chunks\.bin)$/, '').trim();
  console.log(`[JobService] Deleting scene and associated storage for "${cleanId}"...`);

  // 1. Retrieve metadata before deleting to discover associated videoId and filenames
  const meta = await getModelMeta(cleanId);
  const targetIds = new Set([cleanId]);
  if (meta?.jobId) targetIds.add(meta.jobId);
  if (meta?.sceneId) targetIds.add(meta.sceneId);
  if (meta?.videoId) targetIds.add(meta.videoId);

  // 2. Abort any active job running for this scene and remove from in-memory database
  for (const id of targetIds) {
    const activeJob = findActiveJobForScene(id);
    if (activeJob) {
      try {
        await abortJob(activeJob.jobId);
      } catch (err) {
        console.warn(`[JobService] Error aborting active job ${activeJob.jobId}:`, err.message);
      }
    }
  }

  for (const [jobId, job] of jobs.entries()) {
    if (
      targetIds.has(jobId) ||
      targetIds.has(job.sceneId) ||
      targetIds.has(job.videoId) ||
      targetIds.has(job.cachedSceneId)
    ) {
      jobs.delete(jobId);
      console.log(`[JobService] Deleted job record ${jobId} from memory store.`);
    }
  }

  const deletedFiles = [];

  // 3. Remove generated .splat, .ply, .json, and .bak models from uploads/models/
  try {
    const modelEntries = await fs.readdir(MODELS_DIR, { withFileTypes: true });
    for (const entry of modelEntries) {
      if (!entry.isFile()) continue;
      const base = entry.name.replace(/\.(splat|ply|json|bak|ksplat).*$/, '');
      if (targetIds.has(base) || entry.name.startsWith(cleanId)) {
        const filePath = path.join(MODELS_DIR, entry.name);
        await fs.unlink(filePath).catch(() => { });
        deletedFiles.push(`models/${entry.name}`);
        console.log(`[JobService] Deleted model file: ${entry.name}`);
      }
    }
  } catch (err) {
    console.warn('[JobService] Error cleaning model files:', err.message);
  }

  // 4. Remove original uploaded video/image from uploads/
  const uploadsDir = path.resolve('uploads');
  try {
    const uploadEntries = await fs.readdir(uploadsDir, { withFileTypes: true });
    for (const entry of uploadEntries) {
      if (!entry.isFile()) continue;
      const isMatch = Array.from(targetIds).some((id) => entry.name.startsWith(id));
      if (isMatch) {
        const filePath = path.join(uploadsDir, entry.name);
        await fs.unlink(filePath).catch(() => { });
        deletedFiles.push(`uploads/${entry.name}`);
        console.log(`[JobService] Deleted original upload file: ${entry.name}`);
      }
    }
  } catch (err) {
    console.warn('[JobService] Error cleaning upload files:', err.message);
  }

  // 5. Remove extracted frames & cached COLMAP data (images/ and sparse/0/)
  const framesBaseDir = path.resolve('uploads', 'frames');
  for (const id of targetIds) {
    const sceneFramesDir = path.join(framesBaseDir, id);
    try {
      await fs.rm(sceneFramesDir, { recursive: true, force: true });
      deletedFiles.push(`frames/${id}`);
      console.log(`[JobService] Deleted frames & COLMAP cache directory: ${sceneFramesDir}`);
    } catch (err) {
      // frames directory may not exist if post-training cleanup already removed it
    }
  }

  // 6. Purge remote Nebius GPU worker directory via SSH
  try {
    const workerIp = (process.env.NEBIUS_GPU_ENDPOINT || 'http://195.242.13.40:8000')
      .replace('http://', '')
      .replace(':8000', '');
    const { exec } = await import('node:child_process');
    for (const id of targetIds) {
      exec(`ssh -o StrictHostKeyChecking=no ubuntu@${workerIp} "rm -rf /tmp/splatris_jobs/${id} /tmp/splatris_cache/${id}"`, (err) => {
        if (!err) console.log(`[Cleanup] Remote worker /tmp/splatris_jobs/${id} purged via SSH`);
      });
    }
  } catch (e) { }

  return {
    success: true,
    sceneId: cleanId,
    deletedFiles,
    message: `Scene "${cleanId}" and all associated files successfully deleted from storage.`,
  };
}

export default {
  createJob,
  reprocessJob,
  resumeJob,
  deleteScene,
  findActiveJobForScene,
  getModelMeta,
  getJob,
  abortJob,
  getRecentModels,
  getRecentCategorizedModels,
  pruneOldModels,
  cleanupPostTraining,
};



