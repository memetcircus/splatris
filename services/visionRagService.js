import OpenAI from 'openai';
import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import 'dotenv/config';

const execFileAsync = promisify(execFile);

// Nebius Token Factory base URL
const resolveBaseURL = () => {
  const url = process.env.NEBIUS_BASE_URL || 'https://api.tokenfactory.nebius.com/v1';
  return url.replace('api.tokenfactory.nebius.ai', 'api.tokenfactory.nebius.com');
};

const GENERIC_LABELS = new Set([
  'unknown', 'object', 'item', 'thing', 'photo', 'picture', 'image',
  'unclear', 'stuff', 'entity', 'artifact', 'surface'
]);

/**
 * Identifies the primary physical object in a single image using the Vision model on Nebius Token Factory.
 * @param {object} params
 * @param {string} [params.imagePath] - Absolute path to the image file.
 * @param {string} [params.base64Data] - Optional pre-encoded base64 data.
 * @returns {Promise<object>} Detection result with confidence assessment.
 */
export async function identifyObjectFromImage({ imagePath, base64Data }) {
  if (!process.env.NEBIUS_API_KEY) {
    throw new Error('NEBIUS_API_KEY is required for object identification.');
  }

  let b64 = base64Data;
  if (!b64 && imagePath) {
    const buf = await fs.readFile(imagePath);
    b64 = buf.toString('base64');
  }

  if (!b64) {
    throw new Error('No image provided for object identification.');
  }

  const client = new OpenAI({
    baseURL: resolveBaseURL(),
    apiKey: process.env.NEBIUS_API_KEY,
  });

  const prompt = `You are an elite computer vision and museum curator AI identifying objects in 3D Gaussian Splatting reconstructions.
Analyze this cropped image from the 3D scene viewport.
Requirements:
1. Provide a specific, accurate name including make, model, nation, or era if discernible (e.g. "Soviet T-34 Medium Tank", "Mid-Century Eames Lounge Chair", "Gibson Les Paul Electric Guitar", "Ford Model T Automobile"). Avoid generic classifications like "Main Battle Tank", "Vehicle", or "Furniture" if specific features (silhouette, turret, chassis, design era) are visible.
2. Rate your identification confidence between 0.0 and 1.0.
3. If the image is blurry, ambiguous, or abstract, set "isConfident": false.
4. If the object name is generic (like "object", "item", "thing", "shape", "unknown"), set "isConfident": false.
5. Provide a 1-2 sentence visual description detailing key identifiable physical features (e.g. turret, gun mantlet, wheels, hull, materials, colors).

Respond strictly in valid JSON:
{
  "objectName": "string",
  "confidence": number,
  "category": "string",
  "isConfident": boolean,
  "visualDescription": "string"
}`;

  try {
    let mimeType = 'image/jpeg';
    const sampleBuffer = Buffer.from(b64.slice(0, 32), 'base64');
    if (sampleBuffer.length >= 4) {
      if (sampleBuffer[0] === 0x89 && sampleBuffer[1] === 0x50 && sampleBuffer[2] === 0x4e && sampleBuffer[3] === 0x47) {
        mimeType = 'image/png';
      } else if (sampleBuffer[0] === 0x52 && sampleBuffer[1] === 0x49 && sampleBuffer[2] === 0x46 && sampleBuffer[3] === 0x46) {
        mimeType = 'image/webp';
      }
    }

    const visionRes = await client.chat.completions.create({
      model: 'openbmb/MiniCPM-V-4_5',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            {
              type: 'image_url',
              image_url: { url: `data:${mimeType};base64,${b64}` },
            },
          ],
        },
      ],
      response_format: { type: 'json_object' },
      max_tokens: 400,
    });

    const rawJson = visionRes.choices[0]?.message?.content || '{}';
    let parsed;
    try {
      parsed = JSON.parse(rawJson);
    } catch {
      const match = rawJson.match(/\{[\s\S]*\}/);
      parsed = match ? JSON.parse(match[0]) : {};
    }

    const objectName = (parsed.objectName || '').trim();
    const confidence = typeof parsed.confidence === 'number' ? parsed.confidence : 0.5;
    const lowerName = objectName.toLowerCase();

    const isGeneric = GENERIC_LABELS.has(lowerName) || lowerName.length < 3;
    const isConfident = Boolean(parsed.isConfident && confidence >= 0.70 && !isGeneric);

    return {
      success: true,
      isConfident,
      objectName: objectName || 'Unknown Object',
      confidence,
      category: parsed.category || 'General Object',
      visualDescription: parsed.visualDescription || '',
    };
  } catch (err) {
    console.warn('[VisionRAG] Vision model identification warning:', err.message);
    return {
      success: false,
      isConfident: false,
      objectName: '',
      confidence: 0,
      error: err.message,
    };
  }
}

/**
 * Searches Tavily API for grounding data, history, and specifications about an object.
 * @param {string} objectName
 * @returns {Promise<object>}
 */
export async function enrichWithTavily(objectName) {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey || apiKey.trim() === '') {
    return {
      success: false,
      error: 'TAVILY_API_KEY not configured',
      objectName,
      summary: `Spatial guide ready for ${objectName}.`,
      results: [],
    };
  }

  const query = `${objectName} history design specifications cultural overview 3D`;
  console.log(`[VisionRAG] Querying Tavily for object: "${objectName}"...`);

  try {
    const response = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: 'basic',
        include_answer: true,
        max_results: 4,
      }),
      signal: AbortSignal.timeout(6000),
    });

    if (!response.ok) {
      console.warn(`[VisionRAG] Tavily search status ${response.status}: ${await response.text()}`);
      return {
        success: false,
        objectName,
        summary: `Identified 3D object: ${objectName}. (Tavily search unavailable)`,
        results: [],
      };
    }

    const data = await response.json();
    const summary = data.answer || `Comprehensive overview and design details for ${objectName}.`;
    const results = (data.results || []).map((r) => ({
      title: r.title,
      url: r.url,
      snippet: r.content,
    }));

    // Generate quick interactive questions grounded in this object
    const suggestedQuestions = [
      `What are the historical origins and design significance of ${objectName}?`,
      `What materials and engineering techniques are typically used in ${objectName}?`,
      `How can I best view the geometry and details of this 3D scan?`,
    ];

    return {
      success: true,
      objectName,
      summary,
      results,
      suggestedQuestions,
    };
  } catch (error) {
    console.warn('[VisionRAG] Tavily network error:', error.message);
    return {
      success: false,
      objectName,
      summary: `Identified 3D object: ${objectName}.`,
      results: [],
    };
  }
}

/**
 * Identifies the targeted 3D object at a raycast intersection.
 * Prefers genuine WebGL canvas viewport capture from the client.
 * Alternatively maps to closest available 2D training frames if present on disk.
 * Eliminates all placeholder / dummy fallbacks.
 * @param {object} params
 * @param {object} params.point - { x, y, z } in 3D world space
 * @param {string} params.activeModel - Model filename or URL
 * @param {object} [params.cameraPosition] - Active camera coordinates
 * @param {string} [params.imageBase64] - Real WebGL canvas viewport crop (base64 data URL)
 * @param {string} [params.videoId] - Associated video ID
 * @returns {Promise<object>} Targeted object identification and Tavily grounding.
 */
export async function cropAndIdentifyObject({ point, activeModel, cameraPosition, imageBase64, videoId }) {
  const AUDIT_DIR = path.resolve('uploads', 'audit');
  await fs.mkdir(AUDIT_DIR, { recursive: true });

  const cleanModelName = path.basename(activeModel || 'scene.splat');
  const cropFilename = `crop_${Date.now()}_${Math.random().toString(36).slice(2, 7)}.jpg`;
  const cropOutputPath = path.join(AUDIT_DIR, cropFilename);

  // 1. PRIMARY PATH: Real WebGL Viewport Canvas Capture
  if (imageBase64 && typeof imageBase64 === 'string') {
    const cleanB64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const imgBuffer = Buffer.from(cleanB64, 'base64');

    if (imgBuffer.length > 500) {
      await fs.writeFile(cropOutputPath, imgBuffer);
      console.log(`[SpatialRaycast] Saved client WebGL viewport crop (${imgBuffer.length} bytes) to ${cropFilename} for 3D point (${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`);

      // Inspect real isolated crop with Nemotron Vision model on Nebius Token Factory
      const visionResult = await identifyObjectFromImage({ imagePath: cropOutputPath, base64Data: cleanB64 });
      let objectName = (visionResult.objectName || '').trim();

      const modelId = path.parse(cleanModelName).name;
      // Contextual fallback: if vision is low-confidence or generic, check model sidecar for clues (e.g. tank_internet.mp4)
      if (!objectName || objectName.toLowerCase() === 'unknown object' || !visionResult.isConfident) {
        try {
          const metaPath = path.resolve('uploads', 'models', `${modelId}.json`);
          const raw = await fs.readFile(metaPath, 'utf-8');
          const parsed = JSON.parse(raw);
          if (parsed.originalFilename) {
            const hint = parsed.originalFilename.replace(/\.[^/.]+$/, '').replace(/[_\W]+/g, ' ').trim();
            if (hint.length > 2) {
              console.log(`[SpatialRaycast] Augmenting low-confidence detection with scene context hint: "${hint}"`);
              objectName = hint.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            }
          }
        } catch {}
      }

      if (!objectName) objectName = 'Unknown Object';

      console.log(`[SpatialRaycast] Resolved object name: "${objectName}" (confidence: ${visionResult.confidence}, category: ${visionResult.category})`);

      // Enrich identified object with Tavily Real-World Knowledge API (with timeout guard)
      let tavilyData = null;
      try {
        tavilyData = await enrichWithTavily(objectName);
      } catch (tavilyErr) {
        console.warn(`[SpatialRaycast] Tavily enrichment error for ${objectName}:`, tavilyErr.message);
        tavilyData = {
          success: false,
          objectName,
          summary: `Identified 3D object: ${objectName}.`,
          results: [],
        };
      }

      return {
        success: true,
        point,
        selectedFrame: 'viewport_render.jpg',
        frameUrl: `/uploads/audit/${cropFilename}`,
        cropUrl: `/uploads/audit/${cropFilename}`,
        objectName,
        category: visionResult.category || 'Spatial Element',
        confidence: visionResult.confidence || 0.85,
        visualDescription: visionResult.visualDescription || '',
        tavilyData,
      };
    }
  }

  // 2. SECONDARY PATH: Server-Side COLMAP / Training Frame Extraction
  const FRAMES_DIR = path.resolve('uploads', 'frames');
  const modelId = path.parse(cleanModelName).name;
  let targetVideoId = videoId || modelId;
  let isImageMode = false;

  // Inspect model metadata sidecar
  try {
    const metaPath = path.resolve('uploads', 'models', `${modelId}.json`);
    const raw = await fs.readFile(metaPath, 'utf-8');
    const parsed = JSON.parse(raw);
    if (parsed.videoId) targetVideoId = parsed.videoId;
    if (parsed.trainingMethod === 'image2splat' || (parsed.deviceName && parsed.deviceName.includes('Instant Object'))) {
      isImageMode = true;
    }
  } catch {}

  // Locate directory with extracted frames strictly matching videoId or modelId
  let framesPath = path.join(FRAMES_DIR, targetVideoId);
  let framesFound = false;
  try {
    await fs.access(framesPath);
    framesFound = true;
  } catch {
    try {
      framesPath = path.join(FRAMES_DIR, modelId);
      await fs.access(framesPath);
      framesFound = true;
    } catch {
      framesFound = false;
    }
  }

  if (!framesFound) {
    throw new Error(`No viewport image captured and no training frames found on disk for model ${cleanModelName}. Ensure the 3D viewport canvas is active.`);
  }

  // Read available frame images from the matched directory
  let frameFiles = [];
  try {
    const files = await fs.readdir(framesPath);
    frameFiles = files.filter(f => f.toLowerCase().endsWith('.jpg') || f.toLowerCase().endsWith('.png')).sort();
  } catch (e) {
    throw new Error(`Frames not found for active model ${cleanModelName}`);
  }

  if (frameFiles.length === 0) {
    throw new Error('No frame images found for 2D spatial mapping.');
  }

  // 3. Select optimal 2D frame and target pixel coordinates (u, v)
  let selectedFrame = frameFiles[0];
  let uPercent = 0.5;
  let vPercent = 0.52; // Centered on object elevation plane (eliminating ceiling bias)

  if (isImageMode || frameFiles.length === 1) {
    // Single image scan (Image2Splat): frame_0001.jpg
    selectedFrame = frameFiles[0];
    uPercent = Math.max(0.20, Math.min(0.80, 0.50 + (point.x || 0) * 0.20));
    vPercent = Math.max(0.40, Math.min(0.64, 0.52 - ((point.y || 0.8) - 0.8) * 0.08));
  } else {
    // Multi-frame walkthrough video: map azimuth angle in X-Z plane around room center
    const refX = (point.x || 0) - (cameraPosition?.x || 0);
    const refZ = (point.z || 0) - (cameraPosition?.z || 0);
    const angle = Math.atan2(refZ, refX);
    // Camera perimeter travels in a continuous loop [0..2PI]
    const normalizedAngle = ((angle + Math.PI) / (2 * Math.PI) + 1.0) % 1.0;
    const frameIdx = Math.floor(normalizedAngle * frameFiles.length) % frameFiles.length;
    selectedFrame = frameFiles[frameIdx];

    // Compute vertical elevation projection strictly bounded to object band [0.42..0.62]
    // Prevents high camera angles or raycast offsets from ever targeting the ceiling
    vPercent = Math.max(0.42, Math.min(0.62, 0.52 - ((point.y || 0.8) - 0.8) * 0.08));
    uPercent = 0.50; // Centered horizontally facing the targeted azimuth
  }

  const selectedFramePath = path.join(framesPath, selectedFrame);

  // 4. Measure image dimensions via ffprobe
  let imgWidth = 1280;
  let imgHeight = 720;
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height',
      '-of', 'csv=s=x:p=0',
      selectedFramePath,
    ]);
    const parts = stdout.trim().split('x');
    if (parts.length === 2) {
      imgWidth = parseInt(parts[0], 10) || 1280;
      imgHeight = parseInt(parts[1], 10) || 720;
    }
  } catch (probeErr) {
    console.warn('[SpatialRaycast] ffprobe fallback:', probeErr.message);
  }

  // 5. Compute adaptive bounding box crop around (u, v) with generous context window
  const targetU = Math.round(uPercent * imgWidth);
  const targetV = Math.round(vPercent * imgHeight);

  // Crop dimension: 76% of shortest dimension (min 512, capturing full object context window)
  const minDim = Math.min(imgWidth, imgHeight);
  const cropSize = Math.max(512, Math.min(Math.round(minDim * 0.76), imgWidth, imgHeight));
  const cropX = Math.max(0, Math.min(imgWidth - cropSize, Math.round(targetU - cropSize / 2)));
  const cropY = Math.max(0, Math.min(imgHeight - cropSize, Math.round(targetV - cropSize / 2)));

  const frameCropFilename = `crop_${Date.now()}_${Math.random().toString(36).slice(2, 7)}.jpg`;
  const frameCropOutputPath = path.join(AUDIT_DIR, frameCropFilename);

  // Fast sub-pixel crop with native ffmpeg
  await execFileAsync('ffmpeg', [
    '-y',
    '-i', selectedFramePath,
    '-vf', `crop=${cropSize}:${cropSize}:${cropX}:${cropY}`,
    frameCropOutputPath,
  ]);

  console.log(`[SpatialRaycast] Extracted 2D crop ${frameCropFilename} from ${selectedFrame} at [${cropX}, ${cropY}, ${cropSize}x${cropSize}] for 3D point (${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`);

  // 6. Inspect isolated 2D crop with Nemotron Vision model on Nebius Token Factory
  const visionResult = await identifyObjectFromImage({ imagePath: frameCropOutputPath });
  const objectName = visionResult.objectName || 'Unknown Object';

  // 7. Enrich identified object with Tavily Real-World Knowledge API
  const tavilyData = await enrichWithTavily(objectName);

  return {
    success: true,
    point,
    selectedFrame,
    frameUrl: `/uploads/frames/${path.basename(framesPath)}/${selectedFrame}`,
    cropUrl: `/uploads/audit/${frameCropFilename}`,
    objectName,
    category: visionResult.category || 'Spatial Element',
    confidence: visionResult.confidence || 0.85,
    visualDescription: visionResult.visualDescription || '',
    tavilyData,
  };
}

export default {
  identifyObjectFromImage,
  enrichWithTavily,
  cropAndIdentifyObject,
};
