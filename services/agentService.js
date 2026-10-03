import OpenAI from 'openai';
import 'dotenv/config';

// Nebius Token Factory base URL (.com endpoint, auto-correcting any legacy .ai hostname)
const resolveBaseURL = () => {
  const url = process.env.NEBIUS_BASE_URL || 'https://api.tokenfactory.nebius.com/v1';
  return url.replace('api.tokenfactory.nebius.ai', 'api.tokenfactory.nebius.com');
};

export const openai = new OpenAI({
  baseURL: resolveBaseURL(),
  apiKey: process.env.NEBIUS_API_KEY || 'placeholder',
});

/**
 * Sends a test prompt to the Nemotron Ultra model via Nebius Token Factory.
 * @param {string} [prompt] - The user prompt to send for testing.
 * @param {string} [model] - Optional model identifier override.
 * @returns {Promise<object>} The chat completion response.
 */
export async function testAgent(
  prompt = 'Hello! This is a test prompt from Splatris Spatial AI platform.',
  model = process.env.NEBIUS_MODEL || 'nvidia/Nemotron-3-Ultra-550b-a55b'
) {
  if (!process.env.NEBIUS_API_KEY || process.env.NEBIUS_API_KEY.trim() === '') {
    throw new Error('NEBIUS_API_KEY is not set. Please add your key to the .env file.');
  }

  // Map shorthand model name to Nebius Token Factory catalog ID if needed
  let targetModel = model;
  if (targetModel.toLowerCase() === 'nvidia/nemotron-3-ultra') {
    targetModel = 'nvidia/Nemotron-3-Ultra-550b-a55b';
  }

  const client = new OpenAI({
    baseURL: resolveBaseURL(),
    apiKey: process.env.NEBIUS_API_KEY,
  });

  const response = await client.chat.completions.create({
    model: targetModel,
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
  });

  return response;
}

/**
 * Analyzes video frame metadata and recommends COLMAP matching & Gaussian Splatting parameters.
 * @param {object} params
 * @param {object} [params.metadata] - Video metadata (dimensions, duration, fps, codec, etc.)
 * @param {number} [params.totalFrames] - Extracted frame count
 * @param {string} [params.videoId] - Identifier of the video
 * @param {string[]} [params.framePaths] - Array of frame file paths
 * @returns {Promise<{ decisions: object, rawResponse: object }>}
 */
export async function analyzeFramesWithAgent({
  metadata = {},
  totalFrames = 0,
  videoId = null,
  framePaths = [],
} = {}) {
  // Fast path for Single Image Instant Object mode (bypasses COLMAP SfM)
  if (metadata.isImage || totalFrames <= 1) {
    return {
      decisions: {
        colmapMatcher: 'none (instant inference)',
        isLoopTour: false,
        recommendedIterations: 1,
        filteredFrameIndices: [0],
        reasoning: 'Single image instant object scan detected. Bypassing multi-view COLMAP matching and deploying direct Image2Splat depth unprojection.',
      },
      rawResponse: { usage: { total_tokens: 0 } },
    };
  }

  if (!process.env.NEBIUS_API_KEY || process.env.NEBIUS_API_KEY.trim() === '') {
    throw new Error('NEBIUS_API_KEY is not set. Please add your key to the .env file.');
  }

  const client = new OpenAI({
    baseURL: resolveBaseURL(),
    apiKey: process.env.NEBIUS_API_KEY,
  });

  const prompt = `You are a Spatial AI Pipeline Orchestrator for 3D Gaussian Splatting and photogrammetry reconstruction.

Analyze the following extracted video frame metadata:
- Video ID: ${videoId || 'N/A'}
- Total Extracted Frames: ${totalFrames}
- Resolution: ${metadata.width ? `${metadata.width}x${metadata.height}` : 'unknown'}
- Native Video FPS: ${metadata.fps ?? 'unknown'}
- Duration: ${metadata.durationSeconds ? `${metadata.durationSeconds}s` : 'unknown'}
- Codec: ${metadata.codec || 'unknown'}
- Frame Paths Count: ${Array.isArray(framePaths) ? framePaths.length : totalFrames}

Based on this spatial capture, determine the optimal 3D reconstruction parameters:
1. colmapMatcher:
   - When analyzing a room/space walkthrough, interior scan, or video tour that forms a loop (where the camera returns to or revisits the starting position):
     You MUST recommend a matcher capable of loop closure ("vocab_tree" or "exhaustive") rather than pure linear "sequential".
     Pure sequential matching only matches adjacent consecutive frames and completely fails to compare the final frames with the initial frames, causing duplicate ghost room drift, distorted geometry, and split walls.
     "vocab_tree" is the primary recommended matcher for 360° loops and room tours because it indexes all frames against a visual vocabulary tree, enabling accurate loop closure between start and end frames without quadratic explosion.
   - For open-ended, non-returning linear forward paths with zero camera revisit, "sequential" may be used.
2. recommendedIterations: Recommend Gaussian Splatting training iterations (target 10,000 steps for optimal sharpness and fast convergence).
3. filteredFrameIndices: Return an array of frame indices (0-indexed from 0 to ${Math.max(0, totalFrames - 1)}) to keep for reconstruction. Retain frames evenly across the entire perimeter loop to preserve 70-80% visual overlap and continuous baseline parallax, ensuring start and end frames connect properly.
4. isLoopTour: boolean indicating whether this capture forms a closed loop or room walkthrough requiring loop closure.
5. reasoning: Provide a concise technical justification explaining why these parameters and matcher were chosen (specifically addressing loop closure, start-to-end frame connectivity, and spatial geometry).

You MUST respond strictly with a valid JSON object with the following fields:
{
  "colmapMatcher": "vocab_tree" or "exhaustive" or "sequential",
  "isLoopTour": boolean,
  "recommendedIterations": number,
  "filteredFrameIndices": number[],
  "reasoning": "concise explanation"
}`;

  const completion = await client.chat.completions.create({
    model: 'nvidia/Nemotron-3-Ultra-550b-a55b',
    messages: [
      {
        role: 'system',
        content: 'You are the Splatris Spatial AI Pipeline Orchestrator. Output strictly JSON with no surrounding markdown backticks or commentary.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
    response_format: { type: 'json_object' },
  });

  const rawContent = completion.choices[0]?.message?.content || '{}';
  let decisions;
  try {
    decisions = JSON.parse(rawContent);
  } catch (err) {
    const match = rawContent.match(/\{[\s\S]*\}/);
    if (match) {
      decisions = JSON.parse(match[0]);
    } else {
      throw new Error(`Failed to parse agent response as JSON: ${rawContent}`);
    }
  }

  // Enforce step limit guard: strictly cap iterations to 10,000 for fast & high-fidelity L40S training
  decisions.recommendedIterations = Math.min(Number(decisions.recommendedIterations) || 10000, 10000);

  // Dynamic frame guard: cap filteredFrameIndices to at most 220 frames
  const MAX_FRAMES = 220;
  if (Array.isArray(decisions.filteredFrameIndices) && decisions.filteredFrameIndices.length > MAX_FRAMES) {
    const totalIndices = decisions.filteredFrameIndices.length;
    const keepSet = new Set();
    for (let i = 0; i < MAX_FRAMES; i++) {
      keepSet.add(Math.round((i * (totalIndices - 1)) / (MAX_FRAMES - 1)));
    }
    decisions.filteredFrameIndices = decisions.filteredFrameIndices.filter((_, idx) => keepSet.has(idx));
    console.log(`[AgentService] Decimated agent filteredFrameIndices from ${totalIndices} down to ${decisions.filteredFrameIndices.length}`);
  }

  return {
    decisions,
    rawResponse: completion,
  };
}

export default {
  openai,
  testAgent,
  analyzeFramesWithAgent,
};
