import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'node:path';
import agentRoutes from './routes/agent.routes.js';
import spatialRoutes from './routes/spatial.routes.js';
import jobRoutes from './routes/job.routes.js';

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Middleware to parse JSON and url-encoded payloads (up to 1GB)
app.use(express.json({ limit: '1024mb' }));
app.use(express.urlencoded({ extended: true, limit: '1024mb' }));

// Serve static frontend assets from public/ with cache-busting headers
app.use(express.static(path.resolve('public'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.html') || filePath.endsWith('.js')) {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
    }
  }
}));

import { getRecentModels, getRecentCategorizedModels, getModelMeta } from './services/jobService.js';

// Serve uploaded assets, videos, frames, and 3D models statically with caching and CORS
app.use('/uploads', express.static(path.resolve('uploads'), {
  setHeaders: (res, filePath) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');
    res.setHeader('Accept-Ranges', 'bytes');
    if (filePath.endsWith('.splat') || filePath.endsWith('.ply') || filePath.endsWith('.ksplat')) {
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Cache-Control', 'public, max-age=86400, immutable');
    }
  }
}));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

// Recent 3DGS models metadata endpoint (categorized by spaces and objects)
app.get('/api/models/recent', async (req, res) => {
  try {
    const { videoModels, imageModels, models } = await getRecentCategorizedModels(3);
    res.json({
      success: true,
      count: models.length,
      videoModels,
      imageModels,
      models,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Individual scene / model metadata endpoint
app.get('/api/models/meta/:id', async (req, res) => {
  try {
    const meta = await getModelMeta(req.params.id);
    if (!meta) {
      return res.status(404).json({ success: false, error: 'Model metadata not found.' });
    }
    res.json({ success: true, meta });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Service metadata endpoint
app.get('/api/info', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Splatris Spatial AI Platform Backend',
    version: '1.0.0',
  });
});

// Mount modular API routes
app.use('/api/agent', agentRoutes);
app.use('/api/spatial/job', jobRoutes);
app.use('/api/spatial', spatialRoutes);

// Start Express server
app.listen(PORT, () => {
  console.log(`Splatris backend listening on http://localhost:${PORT}`);
});

export default app;
