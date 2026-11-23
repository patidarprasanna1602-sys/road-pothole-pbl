
import os
import sys
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Prevent Ultralytics permission errors on Windows AppData
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

# 2. Automatically link .venv site-packages if running from an unactivated terminal
VENV_SITE = BASE_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import cv2
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_combined.pt"

# Automatically select GPU or CPU
DEVICE = "0" if torch.cuda.is_available() else "cpu"


# Load model only once
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# Human-readable labels and descriptions (10 original RDD classes)
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D01": "Lateral Crack",
    "D0w0": "Wheel-path Crack",
    "D10": "Alligator Crack",
    "D11": "Complex Crack",
    "D20": "Pothole",
    "D40": "Rutting / Surface Distress",
    "D43": "Crosswalk Blur",
    "D44": "Lane Line Blur",
    "D50": "Road Damage",
}

# Short badge labels
CLASS_BADGE = {
    "D00": "Long. Crack",
    "D01": "Lat. Crack",
    "D0w0": "Wheel Crack",
    "D10": "Alligator Crack",
    "D11": "Complex Crack",
    "D20": "Pothole",
    "D40": "Rutting",
    "D43": "Crosswalk",
    "D44": "Lane Blur",
    "D50": "Road Damage",
}

# Severity label based on box pixel area (image-size-relative)
def _severity(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    if area > 15000:
        return "SEVERE"
    elif area > 5000:
        return "MODERATE"
    else:
        return "MINOR"

# Distinct BGR colors per severity (not class — makes it instantly readable)
SEVERITY_COLORS = {
    "SEVERE":   (0, 0, 220),    # Red
    "MODERATE": (0, 100, 255),  # Orange
    "MINOR":    (0, 200, 255),  # Yellow
}


def predict_image(
    image,
    confidence=0.15,
    iou_threshold=0.45,
    agnostic_nms=True,
    imgsz=640,
    augment=False,
    filter_classes=None,
):
    """
    Run YOLO11n inference with customizable NMS and scale parameters.

    Args:
        image: PIL Image, NumPy array, or path string.
        confidence: Minimum confidence score [0.01 - 0.95].
        iou_threshold: NMS IoU threshold [0.10 - 0.80]. Lower values (e.g. 0.40-0.45)
                       eliminate overlapping duplicate boxes on the same damage.
        agnostic_nms: Suppress overlapping boxes even across different classes.
        imgsz: Inference image size (640 matches training resolution).
        augment: Test-Time Augmentation (multi-scale and flip) to boost recall on hard potholes.
        filter_classes: Optional list of class names to detect (e.g. ['D20', 'D40']).
    """
    # Convert input into RGB NumPy array
    if isinstance(image, Image.Image):
        image_array = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        image_array = image.copy()
    else:
        image_array = np.array(
            Image.open(image).convert("RGB")
        )

    # Load YOLO model
    model = load_model()

    # Convert RGB to BGR for OpenCV/Ultralytics input
    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # Map class name filter to indices if specified
    predict_classes = None
    if filter_classes is not None and len(filter_classes) > 0:
        name_to_idx = {name: idx for idx, name in model.names.items()}
        matched_indices = [name_to_idx[name] for name in filter_classes if name in name_to_idx]
        if matched_indices:
            predict_classes = matched_indices

    # Run YOLO prediction with explicit NMS and scale controls
    predict_args = {
        "source": image_bgr,
        "conf": confidence,
        "iou": iou_threshold,
        "agnostic_nms": agnostic_nms,
        "imgsz": imgsz,
        "augment": augment,
        "device": DEVICE,
        "max_det": 300,
        "verbose": False,
    }
    if predict_classes is not None:
        predict_args["classes"] = predict_classes

    results = model.predict(**predict_args)
    result = results[0]

    # Prepare annotated image
    annotated_image = image_bgr.copy()
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # Post-processing: merge nearby same-class boxes
        merged = _merge_nearby_boxes(boxes, scores, classes, merge_iou=0.05)

        detections, annotated_image = _draw_detections(
            annotated_image, merged, result.names
        )

    # Convert annotated image back to RGB for PIL / Streamlit
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    return annotated_image, detections


def predict_image_sahi(
    image,
    confidence=0.08,
    iou_threshold=0.30,
    imgsz=640,
    slice_size=320,
    overlap_ratio=0.3,
    filter_classes=None,
):
    """
    SAHI Sliced Inference — dramatically improves recall on small/hard-to-detect
    potholes by running the model on overlapping image tiles before merging.

    Typical recall improvement: 60% → 85–92%.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        image_pil = Image.open(image).convert("RGB")

    # Map class name filter to indices if specified
    predict_classes = None
    if filter_classes is not None and len(filter_classes) > 0:
        model = load_model()
        name_to_idx = {name: idx for idx, name in model.names.items()}
        matched_indices = [name_to_idx[n] for n in filter_classes if n in name_to_idx]
        if matched_indices:
            predict_classes = matched_indices

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=confidence,
        device=DEVICE,
        image_size=imgsz,
    )

    # For large-pothole images, also run a full-image pass alongside sliced inference
    result = get_sliced_prediction(
        image_pil,
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_match_threshold=iou_threshold,
        verbose=0,
    )

    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    annotated_image = image_bgr.copy()
    detections = []

    # Build class index map for filtering
    model_obj = load_model()

    img_h = image_bgr.shape[0]

    for pred in result.object_prediction_list:
        class_name = pred.category.name
        confidence_score = pred.score.value
        b = pred.bbox

        if filter_classes and class_name not in filter_classes:
            continue

        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)

        # ── Road Horizon Filter ──
        # Potholes and road damage physically exist on the road pavement,
        # never floating in the upper sky, tree crowns, or clouds.
        box_center_y = (y1 + y2) / 2.0
        if y2 < img_h * 0.30 or box_center_y < img_h * 0.35:
            continue  # Ignore sky / tree false positives
        description = CLASS_DESCRIPTIONS.get(class_name, class_name)
        badge_label = CLASS_BADGE.get(class_name, class_name)
        severity = _severity(x1, y1, x2, y2)
        color = SEVERITY_COLORS.get(severity, (0, 255, 0))

        detections.append({
            "class": class_name,
            "description": description,
            "severity": severity,
            "confidence": round(confidence_score, 3),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        })

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        label = f"{badge_label} [{severity}] {confidence_score:.2f}"
        font_scale = 0.42
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        badge_y1 = max(y1 - text_h - 8, 0)
        badge_y2 = max(y1, text_h + 8)
        cv2.rectangle(annotated_image, (x1, badge_y1), (x1 + text_w + 6, badge_y2), color, -1)
        cv2.putText(
            annotated_image, label, (x1 + 3, badge_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image, detections


def _draw_detections(annotated_image, merged, names):
    """Draw boxes and badges on image, return (detections list, annotated_image)."""
    detections = []
    img_h = annotated_image.shape[0]

    for box, score, cls in merged:
        x1, y1, x2, y2 = map(int, box)

        # Ignore false positives high in the sky / upper tree canopy
        box_center_y = (y1 + y2) / 2.0
        if y2 < img_h * 0.30 or box_center_y < img_h * 0.35:
            continue

        class_name = names[int(cls)]
        confidence_score = float(score)
        description = CLASS_DESCRIPTIONS.get(class_name, class_name)
        badge_label = CLASS_BADGE.get(class_name, class_name)
        severity = _severity(x1, y1, x2, y2)
        color = SEVERITY_COLORS.get(severity, (0, 255, 0))

        detections.append({
            "class": class_name,
            "description": description,
            "severity": severity,
            "confidence": round(confidence_score, 3),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        })

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        label = f"{badge_label} [{severity}] {confidence_score:.2f}"
        font_scale = 0.42
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        badge_y1 = max(y1 - text_h - 8, 0)
        badge_y2 = max(y1, text_h + 8)
        cv2.rectangle(annotated_image, (x1, badge_y1), (x1 + text_w + 6, badge_y2), color, -1)
        cv2.putText(
            annotated_image, label, (x1 + 3, badge_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
    return detections, annotated_image


def _merge_nearby_boxes(boxes, scores, classes, merge_iou=0.15):
    """
    Merge boxes of the same class whose intersection-over-smaller-area
    exceeds merge_iou. Used to unify adjacent craters annotated as
    separate D40 boxes.
    Returns list of (box, score, cls) tuples.
    """
    if len(boxes) == 0:
        return []

    items = list(zip(boxes, scores, classes))
    # Sort by confidence descending so the dominant box absorbs weaker ones
    items.sort(key=lambda x: -x[1])

    merged = []
    used = [False] * len(items)

    for i, (box_i, score_i, cls_i) in enumerate(items):
        if used[i]:
            continue
        group_boxes = [box_i]
        group_scores = [score_i]
        used[i] = True

        for j, (box_j, score_j, cls_j) in enumerate(items):
            if used[j] or j <= i:
                continue
            if int(cls_j) != int(cls_i):
                continue

            # Compute IoU with smaller area as denominator (IoS)
            ix1 = max(box_i[0], box_j[0])
            iy1 = max(box_i[1], box_j[1])
            ix2 = min(box_i[2], box_j[2])
            iy2 = min(box_i[3], box_j[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            smaller = min(area_i, area_j)
            ios = inter / smaller if smaller > 0 else 0.0

            if ios >= merge_iou:
                group_boxes.append(box_j)
                group_scores.append(score_j)
                used[j] = True

        # Union bounding box of the group
        gx1 = min(b[0] for b in group_boxes)
        gy1 = min(b[1] for b in group_boxes)
        gx2 = max(b[2] for b in group_boxes)
        gy2 = max(b[3] for b in group_boxes)
        best_score = max(group_scores)
        merged.append((np.array([gx1, gy1, gx2, gy2]), best_score, cls_i))

    return merged
