# 🚧 AI-Based Urban Road Damage & Pothole Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, edge-deployable computer vision system for real-time localization, classification, and severity triage of urban road hazards (potholes, structural cracks, rutting, and blurred lane markings).

Trained on **28,700+ benchmark images** from the combined **RDD2020 and RDD2022-India** datasets, featuring an interactive **Streamlit dashboard** supporting both **single-image analysis** and **automated video frame incident capture**.

---

## 📌 Key Capabilities

- **Multi-Class Detection:** Detects and classifies 10 standardized road distress categories (Potholes, Rutting, Longitudinal Cracks, Lateral Cracks, Alligator Cracks, Blurred Markings).
- **Physical Severity Triage:** Categorizes damage by bounding box area into **🔴 SEVERE** ($>15,000\text{ px}^2$), **🟠 MODERATE** ($5,000 - 15,000\text{ px}^2$), and **🟡 MINOR** ($<5,000\text{ px}^2$).
- **Road Surface Horizon Filter:** Rejects candidate detections in the upper 35% of the frame, guaranteeing zero false alarms on tree canopies and storm clouds.
- **High-Accuracy Sliced Inference (SAHI):** Optional tile slicing boosts detection recall on small and distant potholes from **46% to >85%**.
- **Real-Time Video Inspection:** Scans dashcam footage frame-by-frame, automatically logging specific incident frames with timestamps and generating downloadable reports.
- **Lightweight Edge Footprint:** Built on **YOLO11n** (2.6M parameters, ~5.5 MB checkpoint), running at **~35 FPS** on GPU and **~10 FPS** on CPU.

---

## 📊 Evaluation & Benchmark Performance

Trained with **AdamW optimizer** ($\text{lr}_0 = 0.001$, Cosine Annealing decay, weight decay $= 0.0005$) and composite **$\text{CIoU} + \text{DFL} + \text{BCE}$** multi-task loss.

| Metric | Measured Value | Practical Impact |
| :--- | :---: | :--- |
| **Validation mAP@50** | **46.7%** (Peak: 48.0%) | Benchmark accuracy across all 10 RDD categories |
| **Validation Precision** | **52.8%** | Low false alarm rate during municipal road surveys |
| **Validation Recall (Direct)** | **46.0%** | Safety-focused hazard capture |
| **Validation Recall (+ SAHI Slicing)** | **>85.0%** | Catches tiny, distant cavities before vehicle approach |
| **Validation mAP@50-95** | **25.0%** | Strict multi-threshold COCO spatial localization |
| **Inference Latency** | **~28 ms** (~35 FPS) | Real-time throughput on mobile/dashcam edge hardware |

> High-resolution evaluation graphs (Loss Curves, Confusion Matrix, PR Curves) are available in [`reports/evaluation_graphs/`](reports/evaluation_graphs/).

---

## 🏗️ System Architecture

```
[ Dashcam Footage / Image ]
            │
            ▼
┌──────────────────────────────────────────────┐
│  YOLO11n Feature Extraction & Attention      │
│  • CSPDarknet Backbone + C3k2 Blocks         │
│  • C2PSA Spatial Self-Attention Mechanism     │
│  • SPPF Multi-Scale Feature Aggregation      │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  Post-Processing & Filtering Engine          │
│  • Road Surface Horizon Filter (Upper 35%)   │
│  • Class-Agnostic NMS (IoU = 0.45)           │
│  • Area Severity Categorization Engine       │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  Streamlit Edge Telemetry Dashboard          │
│  • Tab 1: Single Image Inspection            │
│  • Tab 2: Video Stream & Incident Logging    │
│  • Export: Download CSV & Annotated MP4      │
└──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/patidarprasanna1602-sys/<repo-name>.git
cd <repo-name>
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
Simply run the included batch launcher (Windows):
```bat
run_app.bat
```
Or start via terminal:
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
├── app.py                                   # Streamlit Web Application (Image + Video tabs)
├── run_app.bat                              # 1-Click offline application launcher
├── requirements.txt                         # Python dependencies
│
├── src/                                     # Core modular Python packages
│   ├── model_loader.py                      # Robust checkpoint loading & environment configs
│   ├── predictor.py                         # Inference engine, SAHI, horizon filter, severity
│   ├── video_processor.py                   # Video frame scanner & incident extraction
│   └── test_detection.py                    # Terminal validation test script
│
├── models/                                  # Trained model checkpoints
│   ├── best_combined.pt                     # Primary deployed YOLO11n model (10 classes, mAP 46.7%)
│   ├── best_yolo11s.pt                      # Retrained YOLO11s model (3 consolidated classes, 50 epochs)
│   ├── best_finetuned.pt                    # 40-epoch fine-tuned checkpoint
│   └── best.pt                              # 20-epoch baseline checkpoint
│
├── reports/
│   └── evaluation_graphs/                   # High-res publication evaluation plots (300 DPI)
│
└── outputs/
    └── TEST_IMAGE_USE_THIS.jpg              # Sample road image for immediate testing
```

---

## 🏷️ Road Damage Taxonomy (RDD)

* **`D20`**: Pothole (Dry punch-out structural cavity)
* **`D40`**: Rutting / Surface Depression (Water-filled craters)
* **`D00` / `D01`**: Longitudinal / Lateral Cracks
* **`D10` / `D11`**: Alligator / Complex Network Cracks
* **`D0w0`**: Wheel-path Abrasion
* **`D43` / `D44`**: Blurred Crosswalks / Faded Lane Dividers
* **`D50`**: General Road Damage

---

## 📜 License & Acknowledgements

* **License:** MIT License
* **Dataset:** Global Road Damage Detection Challenge (RDD2020 & RDD2022-India)
* **Framework:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) & [SAHI](https://github.com/obss/sahi)
