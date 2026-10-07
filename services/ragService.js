import OpenAI from 'openai';
import 'dotenv/config';

// Nebius Token Factory base URL
const resolveBaseURL = () => {
  const url = process.env.NEBIUS_BASE_URL || 'https://api.tokenfactory.nebius.com/v1';
  return url.replace('api.tokenfactory.nebius.ai', 'api.tokenfactory.nebius.com');
};

/**
 * Queries Tavily API for grounding data if TAVILY_API_KEY is available.
 * @param {string} query
 * @returns {Promise<{ searchAnswer: string|null, searchResults: Array }>}
 */
async function searchTavily(query) {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey || apiKey.trim() === '' || apiKey === 'your_tavily_api_key_here') {
    return { searchAnswer: null, searchResults: [] };
  }

  try {
    const response = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: 'advanced',
        include_answer: true,
        max_results: 5,
      }),
    });

    if (!response.ok) {
      console.warn(`Tavily search warning (${response.status}): ${await response.text()}`);
      return { searchAnswer: null, searchResults: [] };
    }

    const data = await response.json();
    return {
      searchAnswer: data.answer || null,
      searchResults: (data.results || []).map((r) => ({ title: r.title, url: r.url, snippet: r.content })),
    };
  } catch (error) {
    console.warn('Tavily search network fallback:', error.message);
    return { searchAnswer: null, searchResults: [] };
  }
}

/**
 * Processes a spatial RAG query with Tavily search and Nemotron 3 Ultra reasoning.
 * @param {object} params
 * @param {string} params.query - User question regarding the 3D scene or spatial concept.
 * @param {object} [params.spatialContext] - Context about the 3D model, camera position, or scene objects.
 * @param {string} [params.videoId] - Video / scene identifier.
 * @param {object} [params.coordinates] - Current 3D camera / cursor position { x, y, z }.
 * @returns {Promise<object>} Grounded answer with spatial audio positioning.
 */
export async function querySpatialRAG({
  query,
  spatialContext = {},
  videoId = 'scene-01',
  coordinates = { x: 0, y: 1.5, z: 3 },
}) {
  if (!process.env.NEBIUS_API_KEY) {
    throw new Error('NEBIUS_API_KEY is required for Spatial RAG reasoning.');
  }

  // 1. Gather web grounding via Tavily (augmented with identified object if present)
  let searchQuery = query;
  if (spatialContext.identifiedObject && !query.toLowerCase().includes(spatialContext.identifiedObject.toLowerCase())) {
    searchQuery = `${spatialContext.identifiedObject} ${query}`;
  }
  const { searchAnswer, searchResults } = await searchTavily(searchQuery);

  // 2. Prepare Nemotron client
  const client = new OpenAI({
    baseURL: resolveBaseURL(),
    apiKey: process.env.NEBIUS_API_KEY,
  });

  const groundingPrompt = searchResults.length > 0
    ? `Grounding Information:\n${searchAnswer ? `Summary: ${searchAnswer}\n` : ''}${searchResults.map((s, i) => `[${i + 1}] ${s.title}: ${s.snippet}`).join('\n')}`
    : 'Grounding Information: Use built-in 3D Gaussian Splatting, computer vision, and spatial scene reasoning knowledge.';

  const objectSection = spatialContext.identifiedObject ? `
- Focus 3D Object: ${spatialContext.identifiedObject}
- Category: ${spatialContext.objectCategory || 'Spatial Element'}
- Visual Description: ${spatialContext.visualDescription || ''}
- Object Grounding Summary: ${spatialContext.objectGrounding || ''}` : '';

  const prompt = `User Query: "${query}"

Spatial Context:
- Scene ID: ${videoId}
- Camera/User 3D Position: (${coordinates.x ?? 0}, ${coordinates.y ?? 1.5}, ${coordinates.z ?? 3})
- Active 3D Representation: 3D Gaussian Splats (.splat scene)${objectSection}
- Spatial Features: ${JSON.stringify(spatialContext)}

${groundingPrompt}

Instructions:
1. Provide a comprehensive, accurate, and complete response tailored to the 3D scene, the identified 3D object, and the user's spatial question. Do NOT truncate or trail off with ellipses (...).
2. Directly answer specific questions about the object's model, era, history, engineering, or visual features using the identified object context and grounding information.
3. Structure the answer so it sounds articulate, thorough, and natural when synthesized through 3D spatial audio.
4. Determine optimal 3D coordinates (x, y, z) for the spatial audio sound emitter in the scene (e.g. at the focal point or object of interest).
5. Output strictly a JSON object with:
{
  "answer": "Spoken-friendly explanation of the spatial query",
  "audioSpatialPosition": { "x": number, "y": number, "z": number },
  "sources": string[],
  "reasoning": "Technical rationale"
}
6. Strict Spatial Guardrail:
You are a spatial AI guide exclusively focused on the currently reconstructed 3D space. If the user asks a question that is unrelated to the active 3D model, its history, or its spatial context, you MUST politely decline and remind them to ask about the current scene. Never answer general knowledge, coding, or off-topic questions.`;

  const completion = await client.chat.completions.create({
    model: 'nvidia/Nemotron-3-Ultra-550b-a55b',
    messages: [
      {
        role: 'system',
        content:
          'You are a spatial AI guide exclusively focused on the currently reconstructed 3D space. If the user asks a question that is unrelated to the active 3D model, its history, or its spatial context, you MUST politely decline and remind them to ask about the current scene. Never answer general knowledge, coding, or off-topic questions. You deliver grounded, immersive explanations of 3D Gaussian Splatting scenes and emit optimal spatial audio positions. Respond strictly in valid JSON.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
    response_format: { type: 'json_object' },
  });

  const rawContent = completion.choices[0]?.message?.content || '{}';
  let parsed;
  try {
    parsed = JSON.parse(rawContent);
  } catch (err) {
    const match = rawContent.match(/\{[\s\S]*\}/);
    if (match) {
      parsed = JSON.parse(match[0]);
    } else {
      parsed = {
        answer: rawContent,
        audioSpatialPosition: { x: 0, y: 1.0, z: -1.0 },
        sources: [],
        reasoning: 'Fallback parse',
      };
    }
  }

  return {
    success: true,
    query,
    answer: parsed.answer || 'Spatial observation registered.',
    audioSpatialPosition: parsed.audioSpatialPosition || { x: 0, y: 1.0, z: -1.0 },
    sources: parsed.sources || searchResults.map((r) => r.url).filter(Boolean),
    reasoning: parsed.reasoning || null,
    groundingUsed: searchResults.length > 0 ? 'Tavily + Nemotron 3 Ultra' : 'Nemotron 3 Ultra Spatial Knowledge',
    usage: completion.usage,
  };
}

export default {
  querySpatialRAG,
};
