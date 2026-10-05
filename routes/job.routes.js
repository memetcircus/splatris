import express from 'express';
import {
  createJob,
  reprocessJob,
  resumeJob,
  findActiveJobForScene,
  getModelMeta,
  getJob,
  abortJob,
  deleteScene,
} from '../services/jobService.js';

const router = express.Router();

/**
 * POST /api/spatial/job/reprocess
 * Fast re-processing of an existing video scene using cached COLMAP poses.
 * Bypasses Step 1 (Upload) and Step 2 (COLMAP SfM).
 * Body: { sceneId, trainingMethod }
 */
router.post('/reprocess', async (req, res) => {
  try {
    const { sceneId, trainingMethod, training_method } = req.body || {};
    const method = trainingMethod || training_method || 'triangle';

    if (!sceneId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required "sceneId" parameter for re-processing.',
      });
    }

    const job = await reprocessJob({ sceneId, trainingMethod: method });

    res.status(202).json({
      success: true,
      jobId: job.jobId,
      sceneId: job.sceneId,
      status: job.status,
      stage: job.stage,
      progress: job.progress,
      isExisting: Boolean(job.isExisting),
      bypassedUpload: true,
      bypassedColmap: true,
      message: job.isExisting
        ? `Re-attached to active training job for scene "${sceneId}".`
        : `Fast re-processing initiated for scene "${sceneId}" using cached COLMAP poses.`,
    });
  } catch (error) {
    console.error('Failed to reprocess job:', error);
    res.status(400).json({
      success: false,
      error: error.message || 'Failed to reprocess job.',
    });
  }
});

/**
 * POST /api/spatial/job/resume
 * POST /api/spatial/job/:jobId/resume
 * Resumes a suspended or preempted training job on Nebius GPU using cached data.
 */
router.post(['/resume', '/:jobId/resume'], async (req, res) => {
  try {
    const jobId = req.params.jobId || req.body?.jobId;
    const sceneId = req.body?.sceneId;

    if (!jobId && !sceneId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required "jobId" or "sceneId" parameter to resume.',
      });
    }

    const result = await resumeJob({ jobId, sceneId });
    res.json({
      success: true,
      jobId: result.jobId,
      sceneId: result.sceneId,
      status: result.status,
      stage: result.stage,
      progress: result.progress,
      message: result.message || 'Training job resumed.',
    });
  } catch (error) {
    console.error('Failed to resume job:', error);
    res.status(400).json({
      success: false,
      error: error.message || 'Failed to resume training job.',
    });
  }
});

/**
 * GET /api/spatial/job/active/:sceneId
 * Returns any currently active (processing or suspended) job for the specified scene.
 */
router.get('/active/:sceneId', (req, res) => {
  try {
    const { sceneId } = req.params;
    const job = findActiveJobForScene(sceneId);
    res.json({
      success: true,
      hasActiveJob: Boolean(job),
      job: job
        ? {
          jobId: job.jobId,
          sceneId: job.sceneId,
          videoId: job.videoId,
          status: job.status,
          stage: job.stage,
          progress: job.progress,
          trainingMethod: job.trainingMethod,
          error: job.error || null,
          createdAt: job.createdAt,
          isSuspended: job.status === 'suspended',
          isProcessing: job.status === 'processing',
        }
        : null,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * GET /api/spatial/job/cache-status/:sceneId
 * Checks whether cached COLMAP data exists for a given scene ID.
 */
router.get('/cache-status/:sceneId', async (req, res) => {
  try {
    const { sceneId } = req.params;
    const meta = await getModelMeta(sceneId);

    // Check if it's a single image scene
    if (meta && (meta.trainingMethod === 'image2splat' || meta.isImage)) {
      return res.json({
        success: true,
        sceneId,
        hasCache: false,
        isImage: true,
        canReprocess: false,
        message: 'Single image scenes cannot be reprocessed into video pipelines.',
      });
    }

    let hasCache = true;
    try {
      const endpoint = process.env.NEBIUS_GPU_ENDPOINT || 'http://89.169.112.102:8000';
      const cleanEndpoint = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
      const resp = await fetch(`${cleanEndpoint}/cache/status/${sceneId}`, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        const data = await resp.json();
        hasCache = Boolean(data.hasCache);
      }
    } catch { }

    res.json({
      success: true,
      sceneId,
      hasCache,
      isImage: false,
      canReprocess: true,
      currentMethod: meta?.trainingMethod || '3dgs',
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/spatial/job/start
 * Triggers Nebius GPU training process (3DGS or Triangle Splatting).
 * Body: { videoId, colmapMatcher, recommendedIterations, filteredFrameIndices, training_method }
 */
router.post('/start', (req, res) => {
  try {
    const {
      videoId,
      originalFilename,
      deviceName,
      uploadTimestamp,
      colmapMatcher,
      recommendedIterations,
      filteredFrameIndices,
      training_method,
      trainingMethod,
    } = req.body || {};

    const rawMethod = String(training_method || trainingMethod || '3dgs').toLowerCase().trim();
    const effectiveTrainingMethod = ['triangle', 'image2splat'].includes(rawMethod) ? rawMethod : '3dgs';

    const job = createJob({
      videoId: videoId || 'demo-scene',
      originalFilename: originalFilename || null,
      deviceName: deviceName || null,
      uploadTimestamp: uploadTimestamp || null,
      colmapMatcher: colmapMatcher || 'vocab_tree',
      recommendedIterations: Number(recommendedIterations) || 10000,
      filteredFrameIndices: Array.isArray(filteredFrameIndices) ? filteredFrameIndices : [],
      trainingMethod: effectiveTrainingMethod,
    });

    res.status(202).json({
      success: true,
      jobId: job.jobId,
      sceneId: job.sceneId,
      status: job.status,
      stage: job.stage,
      progress: job.progress,
      isExisting: Boolean(job.isExisting),
      createdAt: job.createdAt,
      message: job.isExisting
        ? 'Re-attached to active training job for this scene.'
        : 'Nebius GPU 3D Gaussian Splatting job initiated.',
    });
  } catch (error) {
    console.error('Failed to start 3DGS job:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to start 3DGS training job.',
    });
  }
});

/**
 * POST /api/spatial/job/:jobId/abort
 * POST /api/spatial/job/abort
 * Immediately terminates active COLMAP / 3DGS training on Nebius GPU, purges job data, and releases GPU resources.
 */
router.post(['/:jobId/abort', '/abort'], async (req, res) => {
  try {
    const jobId = req.params.jobId || req.body?.jobId;
    if (!jobId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required "jobId" parameter.',
      });
    }

    const result = await abortJob(jobId);
    res.json({
      success: true,
      jobId,
      status: 'aborted',
      message: `Job ${jobId} successfully aborted. Nebius GPU resources released.`,
    });
  } catch (error) {
    console.error(`Error aborting job:`, error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to abort job.',
    });
  }
});

/**
 * GET /api/spatial/job/status/:jobId
 * Returns the current training status and output 3D model URL by jobId OR sceneId.
 */
router.get('/status/:jobId', (req, res) => {
  try {
    const { jobId } = req.params;
    const job = getJob(jobId);

    if (!job) {
      return res.status(404).json({
        success: false,
        error: `Job or scene with ID "${jobId}" not found.`,
      });
    }

    res.json({
      success: true,
      jobId: job.jobId,
      sceneId: job.sceneId,
      videoId: job.videoId,
      status: job.status, // "processing" | "suspended" | "completed" | "failed" | "aborted"
      stage: job.stage,
      progress: job.progress,
      trainingMethod: job.trainingMethod,
      error: job.error || null,
      outputModelUrl: job.status === 'completed' ? job.outputModelUrl : null,
      createdAt: job.createdAt,
      completedAt: job.completedAt,
      isSuspended: job.status === 'suspended',
      isProcessing: job.status === 'processing',
    });
  } catch (error) {
    console.error('Error fetching job status:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to query job status.',
    });
  }
});

/**
 * DELETE /api/spatial/job/:sceneId
 * Deletes a scene/job record from the database and wipes all associated files:
 * - Original uploaded video/image
 * - Generated .splat and .ply model files
 * - Cached COLMAP data (images/ and sparse/0/)
 */
router.delete('/:sceneId', async (req, res) => {
  try {
    const { sceneId } = req.params;
    if (!sceneId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required "sceneId" parameter for deletion.',
      });
    }

    const result = await deleteScene(sceneId);
    res.json(result);
  } catch (error) {
    console.error(`Failed to delete scene "${req.params.sceneId}":`, error);
    const isProtected = error.message?.includes('protected');
    res.status(isProtected ? 403 : 500).json({
      success: false,
      error: error.message || 'Failed to delete scene.',
    });
  }
});

export default router;


