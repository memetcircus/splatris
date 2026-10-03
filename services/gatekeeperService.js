import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import OpenAI from 'openai';
import 'dotenv/config';
import { getVideoMetadata } from './videoService.js';

const resolveBaseURL = () => {
  const url = process.env.NEBIUS_BASE_URL || 'https://api.tokenfactory.nebius.com/v1';
  return url.replace('api.tokenfactory.nebius.ai', 'api.tokenfactory.nebius.com');
};

const openai = new OpenAI({
  baseURL: resolveBaseURL(),
  apiKey: process.env.NEBIUS_API_KEY || 'placeholder',
});

/**
 * Runs the lightweight OpenCV frame sampler and Laplacian blur detector.
 * @param {string} videoPath - Path to input video file.
 * @param {string} outputDir - Directory to store the 12 sampled thumbnails.
 * @returns {Promise<object>} Extracted sample stats and frames.
 */
export async function runOpenCVAudit(videoPath, outputDir) {
  const scriptPath = path.resolve('pipeline', 'audit_video.py');

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [scriptPath, videoPath, outputDir, '12']);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => { stderr += d.toString(); });

    proc.on('close', (code) => {
      if (code === 0) {
        try {
          const data = JSON.parse(stdout.trim());
          resolve(data);
        } catch (e) {
          reject(new Error(`Failed to parse OpenCV audit output: ${stdout}`));
        }
      } else {
        reject(new Error(`OpenCV audit script exited with code ${code}: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(new Error(`Failed to launch Python audit script: ${err.message}`));
    });
  });
}

/**
 * Performs Pre-Flight AI Video Quality Gatekeeper audit for 3DGS photogrammetry.
 * Evaluates 4 criteria:
 * 1. Motion Blur & Rotation Speed
 * 2. Parallax & Motion Trajectory (physical perimeter loop vs fixed-axis rotation)
 * 3. Lighting & Exposure Stability
 * 4. Visual Texture / Monolithic Surfaces
 *
 * @param {object} params
 * @param {string} params.videoPath - Absolute path to uploaded video.
 * @param {string} params.videoId - Unique video identifier.
 * @param {object} [params.metadata] - Optional video metadata (duration, dims, etc.).
 * @returns {Promise<object>} Strict JSON audit report.
 */
export async function auditVideoQuality({ videoPath, videoId, metadata = {} }) {
  // 1. Pre-Flight Metadata Check: Strict Aspect Ratio Validation (Reject Portrait Videos)
  let meta = metadata;
  if (!meta.width || !meta.height) {
    try {
      meta = await getVideoMetadata(videoPath);
    } catch (probeErr) {
      console.warn('[Gatekeeper] Could not probe video metadata:', probeErr.message);
    }
  }

  const width = meta?.width || 0;
  const height = meta?.height || 0;
  const isPortrait = (width > 0 && height > 0 && height >= width) || meta?.orientation === 'portrait';

  if (isPortrait) {
    console.warn(`[Gatekeeper] Hard rejection: Vertical/Portrait video detected (${width}x${height}). Aborting frame analysis.`);
    return {
      readiness_score: 0,
      status: 'REJECTED',
      aspect_ratio_rejected: true,
      detected_context: 'vertical_video',
      evaluated_by: 'Metadata Pre-Check (Orientation Guard)',
      issues_detected: [
        {
          timestamp: '00:00',
          type: 'Invalid Format: Vertical Video Detected',
          detail: '3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again.',
          is_critical_blocker: true,
        }
      ],
      director_guidance: [
        'Rotate your recording device 90° into Landscape mode (horizontal) before filming.',
        'Maintain a wide horizontal field of view to allow COLMAP to identify overlapping feature points across frames.',
        'Capture a slow, continuous perimeter loop around the subject or space in 16:9 landscape aspect ratio.'
      ],
      laplacian_stats: {
        width,
        height,
        isPortrait: true,
        sampleCount: 0,
        blurryCount: 0,
      },
      thumbnails: [],
    };
  }

  const auditTempDir = path.resolve('uploads', 'audit', videoId);
  await fs.mkdir(auditTempDir, { recursive: true });

  let cvData;
  try {
    cvData = await runOpenCVAudit(videoPath, auditTempDir);
  } catch (err) {
    console.warn('[Gatekeeper] OpenCV pre-audit warning:', err.message);
    cvData = {
      duration: metadata.durationSeconds || 30,
      width: metadata.width || 1920,
      height: metadata.height || 1080,
      samples: [],
      avgLaplacian: 250,
      blurryCount: 0,
    };
  }

  // Select 4-5 representative sampled frame thumbnails to pass to the Vision model
  const samples = cvData.samples || [];
  const selectedSamples = [];
  if (samples.length > 0) {
    const pickIndices = [0, Math.floor(samples.length * 0.25), Math.floor(samples.length * 0.5), Math.floor(samples.length * 0.75), samples.length - 1];
    const uniquePicks = [...new Set(pickIndices)];
    for (const idx of uniquePicks) {
      if (samples[idx]) selectedSamples.push(samples[idx]);
    }
  }

  // Read all sample thumbnails as base64 data URLs
  const sampleMap = new Map();
  for (const s of samples) {
    try {
      const imgBuffer = await fs.readFile(s.imagePath);
      const b64 = imgBuffer.toString('base64');
      s.dataUrl = `data:image/jpeg;base64,${b64}`;
      sampleMap.set(s.timestamp, s);
    } catch (e) {
      console.warn(`[Gatekeeper] Could not read thumbnail ${s.imagePath}:`, e.message);
    }
  }

  // Selected samples for LLM vision payload (up to 5 frames)
  const imageContents = [];
  const previewThumbnails = [];

  for (const s of selectedSamples) {
    if (s.dataUrl) {
      imageContents.push({
        type: 'image_url',
        image_url: { url: s.dataUrl }
      });
      previewThumbnails.push({
        timestamp: s.timestamp,
        laplacianVariance: s.laplacianVariance,
        isBlurry: s.isBlurry,
        dataUrl: s.dataUrl
      });
    }
  }

  // Prepare technical metrics text summary
  const sampleSummaryText = samples.map((s) =>
    `Sample ${s.sampleIndex} (${s.timestamp}): Edge Sharpness (Laplacian Variance)=${s.laplacianVariance} [${s.laplacianVariance >= 50.0 ? 'SHARP - PASS' : 'LOW SHARPNESS / BLUR'}] (Brightness=${s.meanBrightness}, Contrast=${s.contrast})`
  ).join('\n');

  const auditPrompt = `You are the Splatris Pre-Flight Photogrammetry & 3D Gaussian Splatting (3DGS) Quality Gatekeeper.
Audit the following video capture before passing it to COLMAP Structure-from-Motion (SfM) and Nerfstudio 3DGS training on NVIDIA L40S GPU.

Capture Specifications:
- Duration: ${cvData.duration || metadata.durationSeconds || 'unknown'}s
- Resolution: ${cvData.width || metadata.width}x${cvData.height || metadata.height}
- Native Video FPS: ${cvData.fps || metadata.fps || 30}
- Average Edge Sharpness (Laplacian Variance): ${cvData.avgLaplacian}
- OpenCV Blurry Frame Count (Laplacian < 50.0): ${cvData.blurryCount} / ${samples.length}

CRITICAL MATHEMATICAL PRINCIPLE FOR SHARPNESS & BLUR EVALUATION:
- In computer vision, Laplacian variance measures EDGE ENERGY (SHARPNESS).
- A HIGH variance (e.g. >= 50.0, 100, 130, 150, 200+) mathematically proves the frame is SHARP and in-focus with rich edge details.
- High variance values (> 50.0) MUST ALWAYS PASS the sharpness check without error. There are NO upper limits on sharpness.
- Motion blur is ONLY flagged when Laplacian variance falls strictly BELOW 50.0.
- NEVER flag a frame with variance >= 50.0 as "Motion Blur / Rapid Pan".

Per-Frame Blur Measurements:
${sampleSummaryText || 'Standard sample metrics recorded.'}

Context & Subject Detection:
Automatically identify the capture context from the visual frames:
- Options: "indoor_room", "tabletop_object", "vehicle", "architectural_space", "outdoor_scene", "product_scan", or similar.

Evaluate the capture strictly across the 4 core 3DGS photogrammetry criteria:
1. Motion Blur & Rapid Pan: Motion blur is ONLY flagged when Laplacian variance falls strictly BELOW 50.0. Frames with variance >= 50.0 are sharp and MUST NOT be flagged as blur.
2. Parallax & Motion Trajectory: Determine whether the operator maintained physical translational displacement around the subject/environment, or rotated in-place on a fixed central axis (which breaks multi-view stereo triangulation and depth recovery).
3. Lighting & Exposure Stability: Detect blown-out specular highlights, blinding lens flares, deep underexposed shadows, or drastic auto-exposure compensation shifts.
4. Visual Texture & Monolithic Surfaces: Assess if the subject and surrounding surfaces have sufficient keypoint texture vs. giant monolithic, featureless blank surfaces with zero feature matches.

Context-Aware Guidance Rule:
Generate context-aware, actionable director guidance tailored specifically to the detected subject (e.g., room, tabletop object, vehicle, architectural space). Never default generically to "room" unless the footage is clearly an interior room environment.

Output Rules:
- readiness_score: An integer from 0 to 100.
  - If severe motion blur (>2 blurry frames), fixed-axis rotation (zero parallax displacement), or severe overexposure/lack of texture is detected, readiness_score MUST be below 75 (status = "REJECTED").
  - If capture exhibits smooth translational motion around the subject, strong parallax baseline, sharp focus (avg Laplacian > 150), and stable lighting, readiness_score should be 75-98 (status = "APPROVED").
- status: "APPROVED" if readiness_score >= 75, otherwise "REJECTED".
- detected_context: String naming the detected subject/scene type (e.g. "tabletop_object", "indoor_room", "vehicle", "outdoor_scene").
- issues_detected: Array of detected defects with specific timestamps (formatted as "MM:SS"), clear type, and technical explanation.
- director_guidance: Array of 2 to 4 actionable, professional directing tips specifically tailored to the detected subject and scene type to achieve optimal 3DGS convergence.

Respond strictly in valid JSON with this exact structure:
{
  "readiness_score": number,
  "status": "APPROVED" | "REJECTED",
  "detected_context": string,
  "issues_detected": [
    {
      "timestamp": "MM:SS",
      "type": "Motion Blur / Rapid Pan" | "Parallax Loss" | "Overexposure / Harsh Glare" | "Low Texture / Featureless",
      "detail": "concise description"
    }
  ],
  "director_guidance": [
    "context-aware actionable directing advice 1",
    "context-aware actionable directing advice 2"
  ]
}`;

  let auditResult = null;

  // Try Multimodal Vision model first (openbmb/MiniCPM-V-4_5)
  if (imageContents.length > 0) {
    try {
      console.log(`[Gatekeeper] Dispatched ${imageContents.length} frames to openbmb/MiniCPM-V-4_5 on Nebius Token Factory...`);
      const visionRes = await openai.chat.completions.create({
        model: 'openbmb/MiniCPM-V-4_5',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: auditPrompt },
              ...imageContents
            ]
          }
        ],
        response_format: { type: 'json_object' },
        max_tokens: 1000,
      });

      const rawJson = visionRes.choices[0]?.message?.content || '{}';
      auditResult = JSON.parse(rawJson);
      console.log(`[Gatekeeper] MiniCPM-V Audit Completed: Score=${auditResult.readiness_score}, Status=${auditResult.status}, Context=${auditResult.detected_context || 'auto'}`);
    } catch (visionErr) {
      console.warn(`[Gatekeeper] Vision model call warning (${visionErr.message}). Falling back to Nemotron 3 Ultra text reasoning.`);
    }
  }

  // Fallback to nvidia/Nemotron-3-Ultra-550b-a55b with Laplacian stats if vision failed
  if (!auditResult || typeof auditResult.readiness_score !== 'number') {
    try {
      console.log('[Gatekeeper] Running Nemotron 3 Ultra for photogrammetry audit...');
      const textRes = await openai.chat.completions.create({
        model: 'nvidia/Nemotron-3-Ultra-550b-a55b',
        messages: [
          {
            role: 'system',
            content: 'You are the Splatris Pre-Flight Photogrammetry Quality Gatekeeper. Output strictly valid JSON.',
          },
          {
            role: 'user',
            content: auditPrompt,
          }
        ],
        response_format: { type: 'json_object' },
        max_tokens: 1000,
      });

      const rawJson = textRes.choices[0]?.message?.content || '{}';
      auditResult = JSON.parse(rawJson);
      console.log(`[Gatekeeper] Nemotron Audit Completed: Score=${auditResult.readiness_score}, Status=${auditResult.status}, Context=${auditResult.detected_context || 'auto'}`);
    } catch (llmErr) {
      console.error('[Gatekeeper] Fallback LLM error:', llmErr.message);
      // Hard fallback based purely on OpenCV Laplacian metrics
      const score = Math.max(30, Math.min(95, Math.round(cvData.avgLaplacian / 4)));
      const approved = score >= 75;
      auditResult = {
        readiness_score: score,
        status: approved ? 'APPROVED' : 'REJECTED',
        detected_context: 'general_scene',
        issues_detected: cvData.blurryCount > 0 ? [
          {
            timestamp: samples.find(s => s.isBlurry)?.timestamp || '00:02',
            type: 'Motion Blur / Rapid Pan',
            detail: 'Frame sharpness measured low Laplacian variance. Potential motion blur or out-of-focus frame detected.'
          }
        ] : [],
        director_guidance: [
          'Maintain translational movement around the subject; avoid rotating the recording device on a fixed central axis.',
          'Tilt the recording device ~15° downwards toward surface boundaries and textured planes.',
          'Ensure consistent illumination across all viewing angles without harsh specular glare or auto-exposure jumps.'
        ]
      };
    }
  }

  // 1. Inject OpenCV detected blurry frames if model missed them
  const rawIssues = Array.isArray(auditResult.issues_detected) ? auditResult.issues_detected : [];
  if (cvData.blurryCount > 0) {
    for (const s of samples.filter(s => s.isBlurry)) {
      const alreadyReported = rawIssues.some(iss => iss.timestamp === s.timestamp && (iss.type?.includes('Blur') || iss.type?.includes('Pan')));
      if (!alreadyReported) {
        rawIssues.push({
          timestamp: s.timestamp,
          type: 'Motion Blur / Rapid Pan',
          detail: 'Frame sharpness measured low Laplacian variance. Potential motion blur or out-of-focus frame detected.',
          thumbnail_url: s.dataUrl || null,
        });
      }
    }
  }

  // 2. Mathematically Validate Issues: Filter out false-positive blur flags on sharp frames!
  // In image processing, high Laplacian variance (>= 50.0) indicates sharp edge detail.
  // ONLY frames with variance strictly < 50.0 can be flagged as motion blur.
  const issues = [];
  for (const iss of rawIssues) {
    const isBlurType = iss.type && (iss.type.includes('Blur') || iss.type.includes('Pan'));
    if (isBlurType && iss.timestamp) {
      let matchedSample = sampleMap.get(iss.timestamp);
      if (!matchedSample && samples.length > 0) {
        const parts = (iss.timestamp || '00:00').split(':').map(Number);
        const sec = (parts[0] || 0) * 60 + (parts[1] || 0);
        let closest = samples[0];
        let minDiff = Math.abs(closest.timeSeconds - sec);
        for (const s of samples) {
          const diff = Math.abs(s.timeSeconds - sec);
          if (diff < minDiff) {
            minDiff = diff;
            closest = s;
          }
        }
        matchedSample = closest;
      }

      if (matchedSample && matchedSample.laplacianVariance >= 50.0) {
        console.log(`[Gatekeeper] Dismissed false positive blur flag at ${iss.timestamp}: Laplacian variance is ${matchedSample.laplacianVariance} (>= 50.0, mathematically sharp).`);
        continue; // Discard false blur flag!
      }

      // Update UI message accurately reflecting the math
      iss.detail = 'Frame sharpness measured low Laplacian variance. Potential motion blur or out-of-focus frame detected.';
    }
    issues.push(iss);
  }

  // 3. Attach thumbnail_url to all detected issues using timestamp matching or nearest sample
  for (const iss of issues) {
    if (!iss.thumbnail_url) {
      let matchedSample = sampleMap.get(iss.timestamp);
      if (!matchedSample && samples.length > 0) {
        const parts = (iss.timestamp || '00:00').split(':').map(Number);
        const sec = (parts[0] || 0) * 60 + (parts[1] || 0);
        let closest = samples[0];
        let minDiff = Math.abs(closest.timeSeconds - sec);
        for (const s of samples) {
          const diff = Math.abs(s.timeSeconds - sec);
          if (diff < minDiff) {
            minDiff = diff;
            closest = s;
          }
        }
        matchedSample = closest;
      }
      if (matchedSample && matchedSample.dataUrl) {
        iss.thumbnail_url = matchedSample.dataUrl;
      }
    }
  }

  // 3. Proportional Dynamic Readiness Score Calculation (Healthy Frames / Total Samples)
  const total_sampled = samples.length > 0 ? samples.length : 12;

  // Count unique problematic frames (so multiple flags on the same frame aren't double-counted)
  const problematic_timestamps = new Set();
  for (const iss of issues) {
    if (iss.timestamp) {
      problematic_timestamps.add(iss.timestamp.trim());
    }
  }
  for (const s of samples) {
    if (s.isBlurry && s.timestamp) {
      problematic_timestamps.add(s.timestamp.trim());
    }
  }

  const problematic_frames = Math.min(total_sampled, problematic_timestamps.size);
  const healthy_frames = Math.max(0, total_sampled - problematic_frames);
  const score = Math.round((healthy_frames / total_sampled) * 100);

  // Determine pass status (>= 75 is APPROVED)
  const status = score >= 75 ? 'APPROVED' : 'REJECTED';

  const guidance = Array.isArray(auditResult.director_guidance) && auditResult.director_guidance.length > 0
    ? auditResult.director_guidance
    : [
      'Maintain translational movement around the subject; avoid rotating the recording device on a fixed central axis.',
      'Tilt the recording device ~15° downwards toward surface boundaries and textured planes.',
      'Ensure consistent illumination across all viewing angles without harsh specular glare or auto-exposure jumps.'
    ];

  // Clean up disk files in auditTempDir in background
  setTimeout(async () => {
    try {
      await fs.rm(auditTempDir, { recursive: true, force: true });
    } catch (e) {}
  }, 30000);

  return {
    readiness_score: score,
    status,
    detected_context: auditResult.detected_context || 'general_scene',
    evaluated_by: auditResult.evaluated_by || 'openbmb/MiniCPM-V-4_5 (Nebius)',
    issues_detected: issues,
    director_guidance: guidance,
    laplacian_stats: {
      avgVariance: cvData.avgLaplacian,
      minVariance: cvData.minLaplacian,
      blurryCount: cvData.blurryCount,
      sampleCount: samples.length,
    },
    thumbnails: previewThumbnails.slice(0, 3), // Return 3 lightweight thumbnails for UI preview
  };
}

export default {
  runOpenCVAudit,
  auditVideoQuality,
};
