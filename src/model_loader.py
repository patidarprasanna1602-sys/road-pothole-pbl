
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Prevent Ultralytics permission errors on Windows AppData
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

# 2. Automatically link .venv site-packages if running from an unactivated terminal
VENV_SITE = BASE_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

from ultralytics import YOLO

MODEL_PATH = BASE_DIR / "models" / "best_combined.pt"

_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        print("Loading trained YOLO model...")
        _model = YOLO(str(MODEL_PATH))
        print("Model loaded successfully!")

    return _model


if __name__ == "__main__":
    model = load_model()
    print("Model ready:", type(model))
