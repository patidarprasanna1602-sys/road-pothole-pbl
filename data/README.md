# 📁 Dataset Specification: Road Damage Dataset (RDD)

This directory contains the dataset configuration and schema used to train the **YOLO11n** road damage detection model.

---

## 📊 Dataset Overview

* **Benchmark Source:** Combined **RDD2020** (Global Road Damage Detection Challenge) + **RDD2022-India**
* **Total Image Corpus:** **28,709 Images**
  * **Train Split:** 27,535 images (96%)
  * **Validation Split:** 786 images
  * **Test Split:** 388 images
* **Annotation Format:** YOLO Darknet normalized bounding box format (`class_id x_center y_center width height`)
* **Data Hygiene:** 38 cross-dataset duplicate clusters removed via cryptographic **SHA-256 image hashing** to prevent test set data leakage.

---

## 🏷️ 10-Class Damage Taxonomy

| Class ID | Class Code | Description | Physical Characteristics |
| :---: | :---: | :--- | :--- |
| `0` | **`D00`** | Longitudinal Crack | Linear crack running parallel to road direction |
| `1` | **`D01`** | Lateral Crack | Transverse crack running perpendicular to lane |
| `2` | **`D0w0`**| Wheel-path Crack | Surface cracking along vehicle tire paths |
| `3` | **`D10`** | Alligator Crack | Interconnected network/mesh fissures |
| `4` | **`D11`** | Complex Crack | Irregular multi-branching fissures |
| `5` | **`D20`** | **Pothole** | Dry circular punch-out structural cavity |
| `6` | **`D40`** | **Rutting / Depression**| Water-filled craters, surface depressions |
| `7` | **`D43`** | Crosswalk Blur | Worn/faded pedestrian zebra crossings |
| `8` | **`D44`** | Lane Line Blur | Faded white/yellow lane dividing lines |
| `9` | **`D50`** | General Road Damage | Miscellaneous structural surface distress |

---

## 📥 How to Download the Raw Benchmark Images

Due to GitHub's 2 GB repository limit, the multi-gigabyte raw image files are hosted on official academic repositories:

1. **RDD2022 Official Benchmark:**
   - [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://github.com/sekilab/RoadDamageDetector/)
2. **RDD2020 Benchmark:**
   - [IEEE BigData Global Road Damage Detection Challenge 2020](https://github.com/sekilab/RoadDamageDetector/)
3. **Kaggle Mirror:**
   - Search: `RDD2022 Road Damage Dataset` on [Kaggle Datasets](https://www.kaggle.com/)

---

## 🗂️ Expected Directory Structure

When downloading the full dataset locally, arrange the directories as follows:

```text
data/
├── dataset.yaml
├── README.md
├── train/
│   ├── images/   # 27,535 .jpg files
│   └── labels/   # 27,535 .txt annotations
├── valid/
│   ├── images/   # 786 .jpg files
│   └── labels/   # 786 .txt annotations
└── test/
    ├── images/   # 388 .jpg files
    └── labels/   # 388 .txt annotations
```
