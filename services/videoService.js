import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import path from 'node:path';

const execFileAsync = promisify(execFile);

/**
 * Extracts frames from a video at a specified frame rate using ffmpeg.
 * @param {string} videoPath - Absolute path to the source video.
 * @param {string} outputDir - Absolute path to the directory where frames will be saved.
 * @param {number} [fps=2] - Frame rate for extraction (default: 2 fps).
 * @returns {Promise<{ totalFrames: number, frameFiles: string[] }>} Extracted frame information.
 */
export async function extractFrames(videoPath, outputDir, fps = 2, maxFrames = 220) {
  // Validate file existence and size
  const stats = await fs.stat(videoPath);
  if (stats.size < 1024) {
    throw new Error(
      `Uploaded file appears corrupted or empty (${stats.size} bytes). Please ensure you are uploading a valid MP4 or MOV video.`
    );
  }

  // Ensure output directory exists
  await fs.mkdir(outputDir, { recursive: true });

  const outputPattern = path.join(outputDir, 'frame_%04d.jpg');

  // Run ffmpeg with optimal balanced jpeg compression (-q:v 3) and ignore minor container errors
  const args = [
    '-y',
    '-err_detect', 'ignore_err',
    '-i', videoPath,
    '-vf', `fps=${fps}`,
    '-q:v', '3',
    outputPattern
  ];

  try {
    await execFileAsync('ffmpeg', args);
  } catch (error) {
    const errorDetail = error.stderr || error.message;
    // Extract concise error if moov atom or container error
    if (errorDetail.includes('moov atom not found')) {
      throw new Error('Video container error: "moov atom not found". The video file may be incomplete or corrupted.');
    }
    throw new Error(`FFmpeg frame extraction failed: ${errorDetail}`);
  }

  // Read all extracted frame files
  const files = await fs.readdir(outputDir);
  let frameFiles = files
    .filter((file) => file.startsWith('frame_') && file.endsWith('.jpg'))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  if (frameFiles.length === 0) {
    throw new Error('No frames could be extracted from this video. Please ensure the file contains a valid video stream.');
  }

  // --- Strict Frame Decimation: Cap at maxFrames (90) with uniform stride/skip ---
  // A dense consecutive frame sequence (e.g. 200+ frames) creates tiny parallax angles,
  // causing feature blurring and shallow triangulation when matched with vocab_tree.
  // Uniformly decimate to at most maxFrames to enforce a wide baseline between views.
  if (frameFiles.length > maxFrames) {
    const total = frameFiles.length;
    const selectedIndices = new Set();
    for (let i = 0; i < maxFrames; i++) {
      const idx = Math.round((i * (total - 1)) / (maxFrames - 1));
      selectedIndices.add(idx);
    }

    const retainedFiles = [];
    const filesToDelete = [];

    frameFiles.forEach((file, index) => {
      if (selectedIndices.has(index)) {
        retainedFiles.push(file);
      } else {
        filesToDelete.push(file);
      }
    });

    // Remove discarded frames from disk so subsequent pipeline steps never see them
    await Promise.all(
      filesToDelete.map((file) => fs.unlink(path.join(outputDir, file)).catch(() => {}))
    );

    console.log(
      `[VideoService] Decimated frames from ${total} down to ${retainedFiles.length} (cap: ${maxFrames}) with uniform stride to guarantee wide baseline parallax.`
    );
    frameFiles = retainedFiles;
  }

  return {
    totalFrames: frameFiles.length,
    frameFiles,
  };
}

/**
 * Retrieves metadata for a video using ffprobe.
 * Handles iPhone/Android vertical/portrait rotation metadata.
 * @param {string} videoPath - Absolute path to the video file.
 * @returns {Promise<object>} Video metadata including duration, dimensions, orientation, codec, and size.
 */
export async function getVideoMetadata(videoPath, originalName = '') {
  const args = [
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    '-show_streams',
    videoPath
  ];

  try {
    const { stdout } = await execFileAsync('ffprobe', args);
    const probeData = JSON.parse(stdout);

    const videoStream = probeData.streams?.find((s) => s.codec_type === 'video') || {};
    const format = probeData.format || {};

    let width = videoStream.width || null;
    let height = videoStream.height || null;

    // Detect rotation (e.g. iPhone vertical video recorded in portrait)
    let rotation = 0;
    if (videoStream.tags?.rotate) {
      rotation = Math.abs(parseInt(videoStream.tags.rotate, 10));
    } else if (Array.isArray(videoStream.side_data_list)) {
      const displayMatrix = videoStream.side_data_list.find((d) => d.side_data_type === 'Display Matrix');
      if (displayMatrix?.rotation) {
        rotation = Math.abs(parseInt(displayMatrix.rotation, 10));
      }
    }

    // If rotated 90 or 270 degrees, swap display dimensions
    if ((rotation === 90 || rotation === 270) && width && height) {
      const temp = width;
      width = height;
      height = temp;
    }

    // Strict aspect ratio: height >= width is Portrait / Vertical
    const isPortrait = Boolean(width && height && height >= width);

    // Calculate approximate FPS from r_frame_rate
    let fps = null;
    if (videoStream.r_frame_rate) {
      const [num, den] = videoStream.r_frame_rate.split('/').map(Number);
      if (den && !isNaN(num) && !isNaN(den)) {
        fps = Math.round((num / den) * 100) / 100;
      }
    }

    // Extract recording device metadata (e.g., "iPhone 11 Pro", "Insta360 X5", "Meta Quest 3")
    const formatTags = format.tags || {};
    const streamTags = videoStream.tags || {};

    let deviceName = 
      formatTags['com.apple.quicktime.model'] ||
      formatTags['model'] ||
      formatTags['camera_model'] ||
      streamTags['model'] ||
      null;

    const make = formatTags['com.apple.quicktime.make'] || formatTags['make'] || '';
    if (!deviceName && make) {
      if (make.toLowerCase().includes('apple')) deviceName = 'Apple iPhone';
      else if (make.toLowerCase().includes('insta360')) deviceName = 'Insta360 X5';
      else if (make.toLowerCase().includes('gopro')) deviceName = 'GoPro HERO';
      else deviceName = make;
    }

    const desc = formatTags['description'] || '';
    const fn = (originalName || path.basename(videoPath)).toLowerCase();
    if (!deviceName) {
      if (desc.includes('OCULUS') || fn.includes('oculus') || fn.includes('vrshell')) {
        deviceName = 'Meta Quest 3';
      } else if (fn.includes('insta360') || formatTags['encoder']?.includes('Insta360')) {
        deviceName = 'Insta360 X5';
      } else if (fn.includes('gopro') || fn.startsWith('gopr') || fn.startsWith('gh0')) {
        deviceName = 'GoPro HERO';
      } else if (fn.startsWith('img_') || fn.endsWith('.mov')) {
        deviceName = 'Apple iPhone';
      } else if (fn.includes('dji')) {
        deviceName = 'DJI Osmo';
      } else {
        deviceName = 'Mobile Camera';
      }
    }

    return {
      durationSeconds: parseFloat(format.duration || videoStream.duration || '0'),
      width,
      height,
      orientation: isPortrait ? 'portrait' : 'landscape',
      rotation,
      codec: videoStream.codec_name || null,
      fps,
      bitRate: parseInt(format.bit_rate || videoStream.bit_rate || '0', 10),
      sizeBytes: parseInt(format.size || '0', 10),
      formatName: format.format_name || null,
      deviceName,
    };
  } catch (error) {
    console.warn('Could not parse metadata with ffprobe, falling back to file stat:', error.message);
    const stats = await fs.stat(videoPath);
    return {
      durationSeconds: null,
      width: null,
      height: null,
      orientation: 'unknown',
      rotation: 0,
      codec: null,
      fps: null,
      sizeBytes: stats.size,
      formatName: path.extname(videoPath).replace('.', ''),
      deviceName: (originalName || path.basename(videoPath)).toLowerCase().includes('oculus') ? 'Meta Quest 3' : 'Mobile Camera',
    };
  }
}

export default {
  extractFrames,
  getVideoMetadata,
};
