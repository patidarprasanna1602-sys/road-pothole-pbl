
import os
import sys
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
VENV_SITE = BASE_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from ultralytics import YOLO

# Available model checkpoints to compare
MODELS = {
    "Baseline_10Class (Old Model)": BASE_DIR / "models" / "best_combined.pt",
    "Retrained_YOLO11s_3Class (New Model)": BASE_DIR / "models" / "best_yolo11s.pt",
}

# Same validation image for both models
IMAGE_PATH = (
    BASE_DIR
    / "data/raw/RDD2022-India/valid/images/"
    / "India_000005_jpg.rf.b25bff8f235e2ce5ff8a883f2d8706c0.jpg"
)

# Prediction settings
CONFIDENCE = 0.10
IMAGE_SIZE = 640

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print("\nTesting image:", IMAGE_PATH.name)
print("Confidence threshold:", CONFIDENCE)

# Test both models
for model_name, model_path in MODELS.items():

    print("\n" + "=" * 50)
    print(f"TESTING {model_name.upper()} MODEL")
    print("=" * 50)

    if not model_path.exists():
        print(f"ERROR: Checkpoint not found: {model_path}")
        continue

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=0,
        save=True,
        project=str(BASE_DIR / "outputs/model_comparison"),
        name=model_name,
        exist_ok=True,
        verbose=False
    )

    result = results[0]

    print("\nPredictions:")

    if len(result.boxes) == 0:
        print("No detections found.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"Class: {class_name} | "
                f"Confidence: {confidence:.3f}"
            )

    print(f"Saved annotated image in: outputs/model_comparison/{model_name}")

print("\nComparison completed!")
