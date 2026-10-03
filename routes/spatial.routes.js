import express from 'express';
import multer from 'multer';
import path from 'node:path';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { extractFrames, getVideoMetadata } from '../services/videoService.js';
import { auditVideoQuality } from '../services/gatekeeperService.js';
import { querySpatialRAG } from '../services/ragService.js';
import { identifyObjectFromImage, enrichWithTavily, cropAndIdentifyObject } from '../services/visionRagService.js';

const router = express.Router();

const UPLOADS_DIR = path.resolve('uploads');
const FRAMES_DIR = path.resolve('uploads', 'frames');

// Configure Multer storage
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    try {
      await fs.mkdir(UPLOADS_DIR, { recursive: true });
      cb(null, UPLOADS_DIR);
    } catch (err) {
      cb(err, UPLOADS_DIR);
    }
  },
  filename: (req, file, cb) => {
    const videoId = crypto.randomUUID();
    req.videoId = videoId;
    const ext = path.extname(file.originalname).toLowerCase() || '.mp4';
    cb(null, `${videoId}${ext}`);
  },
});

// Allowed video and image file extensions and MIME types
const allowedExtensions = ['.mp4', '.mov', '.jpg', '.jpeg', '.png'];
const allowedMimeTypes = ['video/mp4', 'video/quicktime', 'image/jpeg', 'image/png', 'image/jpg'];

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  const mime = file.mimetype;

  if (allowedExtensions.includes(ext) || allowedMimeTypes.includes(mime) || mime.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type (${ext || mime}). Only MP4, MOV videos and JPG, PNG images are supported.`), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 1024 * 1024 * 1024, // 1 GB (1024 MB) maximum
  },
});

// Middleware to support either 'video', 'image', or 'file' field name
const handleUpload = (req, res, next) => {
  upload.fields([{ name: 'video', maxCount: 1 }, { name: 'image', maxCount: 1 }, { name: 'file', maxCount: 1 }])(req, res, (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({ success: false, error: 'File size exceeds maximum allowed upload limit of 1GB (1024MB).' });
        }
        return res.status(400).json({ success: false, error: `Upload error: ${err.message}` });
      }
      return res.status(400).json({ success: false, error: err.message });
    }
    next();
  });
};

/**
 * POST /api/spatial/upload
 * Accepts an MP4 or MOV video, saves to uploads/, extracts frames at 2 FPS into uploads/frames/{videoId}/,
 * and returns frame count, paths, and video metadata.
 */
router.post('/upload', handleUpload, async (req, res) => {
  try {
    const uploadedFile = req.files?.video?.[0] || req.files?.image?.[0] || req.files?.file?.[0];

    if (!uploadedFile) {
      return res.status(400).json({
        success: false,
        error: 'No video or image file provided. Please upload an MP4/MOV video or JPG/PNG image.',
      });
    }

    const videoId = req.videoId || path.parse(uploadedFile.filename).name;
    const videoPath = uploadedFile.path;
    const originalName = uploadedFile.originalname || 'Walkthrough.mp4';
    const targetFramesDir = path.join(FRAMES_DIR, videoId);
    await fs.mkdir(targetFramesDir, { recursive: true });

    const ext = path.extname(uploadedFile.filename).toLowerCase();
    const isImage = ['.jpg', '.jpeg', '.png'].includes(ext) || (uploadedFile.mimetype && uploadedFile.mimetype.startsWith('image/'));

    // --- Fast Branch for Instant Object Mode (Single Image) ---
    if (isImage) {
      const targetFrame = path.join(targetFramesDir, 'frame_0001.jpg');
      await fs.copyFile(videoPath, targetFrame);

      const imageMeta = {
        videoId,
        originalFilename: originalName,
        deviceName: 'Instant Object (Single Image)',
        uploadTimestamp: new Date().toISOString(),
        isImage: true,
        durationSeconds: 0,
        totalFrames: 1,
      };

      await fs.writeFile(
        path.join(targetFramesDir, 'video_meta.json'),
        JSON.stringify(imageMeta, null, 2)
      );

      console.log(`[Upload] Ingested single photo for Instant Object Scan (Image2Splat): ${originalName} -> ${videoId}`);
      return res.json({
        success: true,
        videoId,
        originalName,
        totalFrames: 1,
        frames: [`/uploads/frames/${videoId}/frame_0001.jpg`],
        targetFrames: 1,
        fps: 1,
        metadata: imageMeta,
        audit: null,
        isImage: true,
        trainingMethod: 'image2splat',
      });
    }

    // 1. Get video duration and device metadata first
    const metadataForFps = await getVideoMetadata(videoPath, originalName);
    const duration = metadataForFps.durationSeconds || 30;

    // 2. Pre-Flight AI Video Quality Gatekeeper Audit
    console.log(`[Upload] Running Pre-Flight AI Quality Audit for video ${videoId} (${metadataForFps.deviceName})...`);
    const audit = await auditVideoQuality({
      videoPath,
      videoId,
      metadata: metadataForFps,
    });
    console.log(`[Upload] Audit complete: Score=${audit.readiness_score}, Status=${audit.status}`);

    // 3. Dynamic Overlap Frame Extraction:
    // If aspect ratio was rejected (vertical video), skip expensive frame extraction immediately
    let targetFrames = 0;
    let fps = 1;
    let totalFrames = 0;
    let frameFiles = [];

    if (!audit.aspect_ratio_rejected) {
      targetFrames = Math.min(220, Math.max(90, Math.round(duration * 2.2)));
      fps = Math.min(6, Math.max(1, Math.ceil(targetFrames / duration)));
      const extracted = await extractFrames(videoPath, targetFramesDir, fps, targetFrames);
      totalFrames = extracted.totalFrames;
      frameFiles = extracted.frameFiles;
    }

    // 4. Extract video metadata (full, for client)
    const metadata = metadataForFps;

    // Persist upload video metadata sidecar for downstream job association
    await fs.mkdir(targetFramesDir, { recursive: true }).catch(() => {});
    await fs.writeFile(
      path.join(targetFramesDir, 'video_meta.json'),
      JSON.stringify({
        videoId,
        originalFilename: originalName,
        deviceName: metadataForFps.deviceName || 'Mobile Camera',
        uploadTimestamp: new Date().toISOString(),
        metadata: metadataForFps,
      }, null, 2)
    ).catch(() => {});

    // 5. Format relative paths for client consumption
    const relativeVideoPath = path.relative(process.cwd(), videoPath);
    const relativeFramesDir = path.relative(process.cwd(), targetFramesDir);
    const relativeFramePaths = frameFiles.map((f) => path.join(relativeFramesDir, f));

    res.status(200).json({
      success: true,
      videoId,
      originalName,
      deviceName: metadataForFps.deviceName || 'Mobile Camera',
      uploadTimestamp: new Date().toISOString(),
      videoPath: relativeVideoPath,
      fpsExtracted: fps,
      targetFrames,
      totalFrames,
      framesDirectory: relativeFramesDir,
      frames: relativeFramePaths,
      metadata,
      audit,
    });
  } catch (error) {
    console.error('Spatial video upload processing error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'An error occurred during video processing.',
    });
  }
});

/**
 * POST /api/spatial/preflight-audit
 * Standalone endpoint to audit video quality on demand.
 */
router.post('/preflight-audit', async (req, res) => {
  try {
    const { videoId } = req.body || {};
    if (!videoId) {
      return res.status(400).json({ success: false, error: 'Missing required "videoId" parameter.' });
    }

    const uploadsDir = path.resolve('uploads');
    const files = await fs.readdir(uploadsDir);
    const videoFileName = files.find((f) => f.startsWith(videoId) && (f.endsWith('.mp4') || f.endsWith('.mov')));

    if (!videoFileName) {
      return res.status(404).json({ success: false, error: `Raw video file for ID "${videoId}" not found in uploads.` });
    }

    const videoPath = path.join(uploadsDir, videoFileName);
    const metadata = await getVideoMetadata(videoPath);
    const audit = await auditVideoQuality({ videoPath, videoId, metadata });

    res.json({ success: true, videoId, audit });
  } catch (error) {
    console.error('Pre-flight audit error:', error);
    res.status(500).json({ success: false, error: error.message || 'Failed to complete video audit.' });
  }
});

/**
 * POST /api/spatial/rag-query
 * Interactive Spatial Audio RAG query powered by Tavily Search and Nemotron 3 Ultra.
 * Body: { query, spatialContext, videoId, coordinates }
 */
router.post('/rag-query', async (req, res) => {
  try {
    const { query, spatialContext, videoId, coordinates } = req.body || {};

    if (!query || typeof query !== 'string' || query.trim() === '') {
      return res.status(400).json({
        success: false,
        error: 'Missing required "query" parameter in request body.',
      });
    }

    const result = await querySpatialRAG({
      query: query.trim(),
      spatialContext: spatialContext || {},
      videoId: videoId || 'active-scene',
      coordinates: coordinates || { x: 0, y: 1.5, z: 3 },
    });

    res.json(result);
  } catch (error) {
    console.error('Spatial RAG query error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to process spatial RAG query.',
    });
  }
});
/**
 * POST /api/spatial/instant-object/identify
 * Identifies the primary physical object using the Vision model on Nebius Token Factory.
 * If confident, autonomously queries Tavily. If ambiguous, flags needsUserInput: true.
 */
router.post('/instant-object/identify', async (req, res) => {
  try {
    const { videoId, base64Data } = req.body || {};
    let imagePath = null;
    if (videoId) {
      const candidates = [
        path.join(FRAMES_DIR, videoId, 'frame_0001.jpg'),
        path.join(UPLOADS_DIR, `${videoId}.jpg`),
        path.join(UPLOADS_DIR, `${videoId}.png`),
        path.join(UPLOADS_DIR, `${videoId}.jpeg`),
      ];
      for (const c of candidates) {
        try {
          await fs.access(c);
          imagePath = c;
          break;
        } catch {}
      }
    }

    if (!imagePath && !base64Data) {
      return res.status(400).json({ success: false, error: 'Neither valid videoId nor base64Data provided.' });
    }

    const idResult = await identifyObjectFromImage({ imagePath, base64Data });

    if (idResult.isConfident && idResult.objectName) {
      // Confident: automatically enrich with Tavily
      const tavilyData = await enrichWithTavily(idResult.objectName);
      return res.json({
        success: true,
        isConfident: true,
        objectName: idResult.objectName,
        confidence: idResult.confidence,
        category: idResult.category,
        visualDescription: idResult.visualDescription,
        tavilyData,
      });
    } else {
      // Paused: needs user confirmation or manual text override
      return res.json({
        success: true,
        isConfident: false,
        needsUserInput: true,
        reason: 'Could not clearly identify the object. What is this?',
        suggestedName: idResult.objectName || null,
        confidence: idResult.confidence || 0,
      });
    }
  } catch (err) {
    console.error('[VisionRAG] Identification route error:', err);
    res.status(500).json({
      success: false,
      isConfident: false,
      needsUserInput: true,
      reason: 'Could not clearly identify the object. What is this?',
      error: err.message,
    });
  }
});

/**
 * POST /api/spatial/instant-object/tavily-enrich
 * Autonomous agent Tavily enrichment after user enters or confirms the object name.
 */
router.post('/instant-object/tavily-enrich', async (req, res) => {
  try {
    const { objectName, videoId } = req.body || {};
    if (!objectName || !objectName.trim()) {
      return res.status(400).json({ success: false, error: 'Object name is required.' });
    }

    const cleanName = objectName.trim();
    const tavilyData = await enrichWithTavily(cleanName);

    res.json({
      success: true,
      objectName: cleanName,
      tavilyData,
    });
  } catch (err) {
    console.error('[VisionRAG] Tavily enrichment route error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * POST /api/spatial/raycast-object
 * Receives 3D intersection point { x, y, z }, maps to closest training video frame,
 * crops target object bounding box, identifies via Nemotron Vision, and enriches via Tavily.
 */
router.post('/raycast-object', async (req, res) => {
  try {
    const { point, activeModel, cameraPosition, imageBase64, videoId } = req.body || {};
    if (!point || typeof point.x !== 'number') {
      return res.status(400).json({ success: false, error: 'Invalid 3D intersection point coordinates.' });
    }

    console.log(`[SpatialRaycast] Query received for 3D point (${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)}) on model ${activeModel} (has viewport capture: ${Boolean(imageBase64)})`);
    const result = await cropAndIdentifyObject({ point, activeModel, cameraPosition, imageBase64, videoId });
    res.json(result);
  } catch (err) {
    console.error('[SpatialRaycast] Error handling 3D point query:', err);
    res.status(500).json({ success: false, error: err.message || 'Spatial raycast analysis failed.' });
  }
});

/**
 * GET /api/spatial/tts
 * Synthesizes high-fidelity spoken audio MP3 stream for spatial 3D WebXR audio playback.
 * The audio is streamed directly to client Web Audio API PannerNode attached to 3D object coordinates.
 */
router.get('/tts', async (req, res) => {
  try {
    const rawText = (req.query.text || '').trim();
    if (!rawText) {
      return res.status(400).json({ error: 'Text query parameter is required.' });
    }

    // Clean text: strip markdown symbols and limit to a concise narration sentence (max 160 chars for Google TTS)
    const cleanText = rawText
      .replace(/[*_#`~[\]()]/g, ' ')
      .replace(/\s+/g, ' ')
      .slice(0, 160)
      .trim();

    const ttsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodeURIComponent(cleanText)}&tl=en&client=tw-ob`;
    const response = await fetch(ttsUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
    });

    if (!response.ok) {
      throw new Error(`TTS upstream error: ${response.status}`);
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioBuffer.length);
    res.setHeader('Cache-Control', 'public, max-age=3600');
    return res.end(audioBuffer);
  } catch (err) {
    console.warn('[SpatialTTS] Synthesis error, client will use Web Speech fallback:', err.message);
    return res.status(500).json({ error: 'TTS synthesis unavailable' });
  }
});

export default router;

