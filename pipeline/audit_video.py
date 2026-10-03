#!/usr/bin/env python3
"""
Splatris Pre-Flight Video Quality Auditor (OpenCV / Python)
Performs lightweight, rapid frame extraction (12 uniform samples)
and calculates Laplacian variance, brightness, and contrast to detect motion blur.
"""

import sys
import os
import json
import cv2

def audit_video(video_path, output_dir, sample_count=12):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    # Strict Aspect Ratio Validation: Reject vertical / portrait video (height >= width)
    if width > 0 and height > 0 and height >= width:
        cap.release()
        result = {
            "duration": round(duration, 2),
            "width": width,
            "height": height,
            "is_portrait": True,
            "aspect_ratio_rejected": True,
            "error": "Invalid Format: Vertical Video Detected",
            "samples": [],
            "avgLaplacian": 0.0,
            "minLaplacian": 0.0,
            "blurryCount": 0,
        }
        print(json.dumps(result))
        return

    os.makedirs(output_dir, exist_ok=True)

    if total_frames <= sample_count:
        indices = list(range(max(1, total_frames)))
    else:
        step = (total_frames - 1) / (sample_count - 1)
        indices = [round(i * step) for i in range(sample_count)]

    samples = []
    laplacian_scores = []

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_val = float(gray.mean())
        contrast_val = float(gray.std())
        laplacian_scores.append(laplacian_var)

        time_sec = float(frame_idx / fps) if fps > 0 else 0.0
        m = int(time_sec // 60)
        s = int(time_sec % 60)
        timestamp = f"{m:02d}:{s:02d}"

        # Resize thumbnail to 480px width max for lightweight AI vision payload
        h, w = frame.shape[:2]
        scale = min(1.0, 480.0 / max(h, w))
        thumb_w = max(32, int(w * scale))
        thumb_h = max(32, int(h * scale))
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        img_name = f"sample_{i:02d}.jpg"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Sharpness / Blur Threshold: 50.0 minimum edge variance for 3DGS photogrammetry.
        # In image processing, high Laplacian variance indicates sharp focus and rich edge details.
        # High variance values (> 50.0, 100+, 150+) ALWAYS pass without upper bound limits.
        # Motion blur is ONLY flagged when variance falls strictly below 50.0.
        BLUR_THRESHOLD = 50.0
        is_blurry = bool(laplacian_var < BLUR_THRESHOLD)
        is_overexposed = bool(mean_val > 220.0)
        is_underexposed = bool(mean_val < 30.0)

        samples.append({
            "sampleIndex": i,
            "frameIndex": frame_idx,
            "timestamp": timestamp,
            "timeSeconds": round(time_sec, 2),
            "laplacianVariance": round(laplacian_var, 1),
            "meanBrightness": round(mean_val, 1),
            "contrast": round(contrast_val, 1),
            "imagePath": img_path,
            "imageFilename": img_name,
            "isBlurry": is_blurry,
            "isOverexposed": is_overexposed,
            "isUnderexposed": is_underexposed,
        })

    cap.release()

    avg_laplacian = float(sum(laplacian_scores) / len(laplacian_scores)) if laplacian_scores else 0.0
    min_laplacian = min(laplacian_scores) if laplacian_scores else 0.0
    blurry_count = sum(1 for s in samples if s["isBlurry"])

    return {
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "totalFrames": total_frames,
        "sampleCount": len(samples),
        "avgLaplacian": round(avg_laplacian, 1),
        "minLaplacian": round(min_laplacian, 1),
        "blurryCount": blurry_count,
        "samples": samples
    }


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python3 audit_video.py <video_path> <output_dir> [sample_count=12]\n")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    sample_count = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    try:
        data = audit_video(video_path, output_dir, sample_count)
        print(json.dumps(data))
    except Exception as e:
        sys.stderr.write(f"Error auditing video: {e}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
