import express from 'express';
import { testAgent, analyzeFramesWithAgent } from '../services/agentService.js';

const router = express.Router();

/**
 * GET /api/agent/test
 * Tests the agent integration with Nebius Token Factory (nvidia/nemotron-3-ultra).
 * Query params: ?prompt=<string> (optional)
 */
router.get('/test', async (req, res) => {
  try {
    const prompt = req.query.prompt || 'Hello! This is a test prompt from Splatris Spatial AI platform.';
    const response = await testAgent(prompt);
    res.json({
      success: true,
      data: response,
    });
  } catch (error) {
    console.error('Agent test error:', error.message);
    const status = error.status || 500;
    res.status(status).json({
      success: false,
      error: error.message || 'An error occurred while communicating with the agent service.',
    });
  }
});

/**
 * POST /api/agent/analyze-frames
 * Passes video frame metadata to Nemotron 3 Ultra to determine COLMAP and Gaussian Splatting parameters.
 * Expects JSON body: { videoId, totalFrames, metadata, framePaths }
 */
router.post('/analyze-frames', async (req, res) => {
  try {
    const { videoId, totalFrames, metadata, framePaths } = req.body;

    if (totalFrames === undefined && !metadata) {
      return res.status(400).json({
        success: false,
        error: 'Missing required payload. Expected { videoId, totalFrames, metadata, framePaths } in request body.',
      });
    }

    const { decisions, rawResponse } = await analyzeFramesWithAgent({
      videoId,
      totalFrames: totalFrames ?? (Array.isArray(framePaths) ? framePaths.length : 0),
      metadata: metadata || {},
      framePaths: framePaths || [],
    });

    res.json({
      success: true,
      videoId: videoId || null,
      decisions,
      usage: rawResponse?.usage,
    });
  } catch (error) {
    console.error('Agent analyze-frames error:', error.message);
    const status = error.status || 500;
    res.status(status).json({
      success: false,
      error: error.message || 'An error occurred while analyzing frames with the agent.',
    });
  }
});

export default router;
