import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
VENV_SITE = BASE_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from src.predictor import predict_image


def process_video(
    video_path,
    output_path=None,
    confidence=0.15,
    iou_threshold=0.45,
    frame_skip=2,
    imgsz=640,
    filter_classes=None,
    pothole_only=False,
    max_frames=None,
    progress_callback=None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    # Prepare VideoWriter if output_path is requested
    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # mp4v is universally supported on Windows OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = video_fps / frame_skip if frame_skip > 0 else video_fps
        writer = cv2.VideoWriter(str(out_p), fourcc, out_fps, (width, height))

    incident_frames = []
    total_detections_count = 0
    processed_count = 0
    frame_idx = 0
    start_time = time.time()

    SEVERITY_ORDER = {"SEVERE": 3, "MODERATE": 2, "MINOR": 1}

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Frame skipping for higher throughput
            if frame_idx % frame_skip != 0:
                continue

            processed_count += 1

            # Convert BGR to RGB for PIL / prediction
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run prediction
            annotated_rgb, detections = predict_image(
                pil_image,
                confidence=confidence,
                iou_threshold=iou_threshold,
                agnostic_nms=True,
                imgsz=imgsz,
                filter_classes=filter_classes,
            )

            # Filter for incident capture
            relevant_dets = detections
            if pothole_only:
                relevant_dets = [
                    d for d in detections
                    if d["class"] in ["D20", "D40", "Pothole"]
                ]

            has_incident = len(relevant_dets) > 0

            if has_incident:
                total_detections_count += len(relevant_dets)

                # Determine highest severity in this frame
                severities = [d.get("severity", "MINOR") for d in relevant_dets]
                highest_severity = max(
                    severities, key=lambda s: SEVERITY_ORDER.get(s, 0)
                )
                max_conf = max(d.get("confidence", 0) for d in relevant_dets)
                classes_present = list(set(d["class"] for d in relevant_dets))

                timestamp_sec = frame_idx / video_fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                millis = int((timestamp_sec * 10) % 10)
                timestamp_str = f"{minutes:02d}:{seconds:02d}.{millis}"

                # Only keep snapshot if it's a new or distinct incident (avoid flooding every consecutive frame)
                incident_frames.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "timestamp_str": timestamp_str,
                    "detections_count": len(relevant_dets),
                    "classes": classes_present,
                    "highest_severity": highest_severity,
                    "max_confidence": round(float(max_conf), 3),
                    "annotated_image": annotated_rgb,
                    "details": relevant_dets,
                })

            # Write annotated frame to output video if writer enabled
            if writer is not None:
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                writer.write(annotated_bgr)

            # Progress callback
            if progress_callback is not None:
                elapsed = time.time() - start_time
                current_fps = processed_count / elapsed if elapsed > 0 else 0
                progress_callback(
                    current_frame=frame_idx,
                    total_frames=frames_to_process,
                    incidents_count=len(incident_frames),
                    fps=round(current_fps, 1),
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    total_elapsed = time.time() - start_time
    final_fps = processed_count / total_elapsed if total_elapsed > 0 else 0

    return {
        "output_path": str(output_path) if output_path else None,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "total_detections": total_detections_count,
        "incident_frames": incident_frames,
        "processing_fps": round(final_fps, 1),
        "duration_sec": round(duration_sec, 2),
        "video_fps": round(video_fps, 1),
        "resolution": f"{width}x{height}",
    }
