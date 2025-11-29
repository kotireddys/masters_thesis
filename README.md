# 🛰️ Radar-Semantic Fusion for Autonomous UAV Safe Landing Zone Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis)

**Master's Thesis Project** | Academic Year 2024/2025  
**Candidate:** Koti Reddy Syamala (Matricola P44000065)  
**Institution:** Università degli Studi di Napoli "Federico II"  
**Supervisors:** Prof. Giancarmine Fasano, Dr. Dmitry Ignatyev

---

## 📋 **Table of Contents**

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Performance Summary](#-performance-summary)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Workflow Execution](#-workflow-execution)
- [Results](#-results)
- [Future Work](#-future-directions)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 **Overview**

This repository contains the **complete simulation code, model training pipelines, and documentation** for my Master's thesis:

> *"A Proof of Concept for Autonomous UAV Safe Landing Zone Detection in GPS-Denied Unstructured Terrain Using Synthetic Radar Altimeter and Semantic Segmentation"*

### **The Problem**
Autonomous Unmanned Aerial Vehicles (UAVs) require reliable perception to ensure safe landings, especially when GPS signals are unavailable (e.g., urban canyons, disaster zones, indoor environments). Existing solutions face critical trade-offs:
- **Vision-only systems** fail in poor weather, fog, or low light
- **LiDAR systems** are accurate but heavy, expensive, and power-hungry
- **Radar altimeters** are weather-resilient but lack semantic context

### **Our Solution**
A **multi-modal fusion framework** that combines:
1. **Synthetic FMCW radar altimeter simulation** (geometric safety via DEMs)
2. **Deep learning semantic segmentation** (terrain classification from RGB imagery)
3. **Bayesian probabilistic fusion** to produce reliable safety maps

This approach achieves **all-weather robustness** with **minimal computational overhead**, making it suitable for deployment on small UAV platforms.

---

## 🚀 **Key Contributions**

### 1. **Synthetic FMCW Radar Simulation**
- Physics-based emulation of Ainstein US-D1 radar behavior
- Models Gaussian noise, quantization, and SNR-dependent dropout
- Achieves **0.12 m RMSE** compared to ground truth LiDAR DEMs

### 2. **Lightweight Semantic Segmentation**
- Trained DeepLabV3+ (ResNet50) and YOLOv11-Seg models on 525 aerial image tiles
- Achieved **67% mean IoU** and **>90% precision** for safe terrain classes
- Fully cloud-native training pipeline via Ultralytics Hub + Roboflow

### 3. **Geometric-Semantic Fusion Logic**
- Bayesian integration: **P_safe = P_geo × P_sem**
- Combines slope/roughness metrics with terrain classification
- Outperforms vision-only baselines by **10–12%**

### 4. **Reproducible Cloud Workflow**
- All experiments run on Google Colab Pro (A100/T4 GPUs)
- No specialized hardware required
- Full experiment tracking and version control

### 5. **Open Science Approach**
- Complete code and documentation publicly available
- Modular architecture allows easy extension to additional sensors
- Supports future Hardware-in-the-Loop (HIL) validation

---

## 📈 **Performance Summary**

| Metric | Value | Baseline (Vision-Only)* |
|--------|-------|-------------------------|
| **Radar Emulation RMSE** | 0.12 m | N/A |
| **Radar-DEM Correlation (R²)** | 0.987 | N/A |
| **Semantic mIoU** | 67% | 58% |
| **Safe-Class Precision** | >90% | 82% |
| **Fusion Precision** | 0.78 | 0.66 |
| **Fusion Recall** | 0.72 | 0.60 |
| **IoS (Intersection/Segmentation)** | 0.84 | 0.72 |
| **Inference Speed** | ~30 FPS (semantic) | ~25 FPS |

*Comparative baselines from Benjwal et al. (2023) and Cho & Jung (2022)

**Hardware:** Google Colab Pro with NVIDIA A100 (40GB) and T4 (16GB) GPUs

---

## 🏗️ **System Architecture**

The framework implements a **five-layer modular pipeline**:
```
┌─────────────────────────────────────────────────────────────┐
│                    1. Sensor-Physics Layer                   │
│         FMCW Radar Altimeter Simulation from DEMs            │
│    (Gaussian noise + quantization + SNR dropout)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    2. Geospatial Layer                       │
│     DEM Preprocessing, Slope/Roughness Extraction → P_geo    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    3. Semantic Layer                         │
│    DeepLabV3+/YOLOv11-Seg Terrain Classification → P_sem     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                 4. Fusion & Decision Layer                   │
│         Bayesian Integration: P_safe = P_geo × P_sem         │
│         Thresholding, Connected Components, Ranking          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    5. Validation Layer                       │
│      Precision, Recall, IoU, IoS/IoM Metrics Evaluation      │
└─────────────────────────────────────────────────────────────┘
```

**Data Sources:**
- **AHN4 LiDAR DEMs** (1m resolution, ±5cm vertical accuracy)
- **PDOK RGB Orthophotos** (10–25cm resolution)

---

## 📁 **Repository Structure**
```
masters_thesis/
│
├── Thesis1_Dataprep.ipynb              # Step 1: Data acquisition & alignment
├── Thesis2_datasetcreation.ipynb       # Step 2: Dataset curation & augmentation
├── Thesis3_Semanticseg.ipynb           # Step 3: Semantic model training
├── Thesis4_Radaraltimeter_sim.ipynb    # Step 4: Radar emulation & P_geo
├── Thesis5_SDF.ipynb                   # Step 5: Fusion logic & validation
├── LICENSE                             # MIT License
└── README.md                           # This file
```

### **Notebook Descriptions**

| File | Step | Description | Open in Colab |
|------|------|-------------|---------------|
| `Thesis1_Dataprep.ipynb` | 1 | Geospatial alignment, cropping AHN4 DEM & PDOK RGB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis/blob/main/Thesis1_Dataprep.ipynb) |
| `Thesis2_datasetcreation.ipynb` | 2 | Augmentation, labeling review, data splits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis/blob/main/Thesis2_datasetcreation.ipynb) |
| `Thesis3_Semanticseg.ipynb` | 3 | Training DeepLabV3+/YOLOv11-Seg models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis/blob/main/Thesis3_Semanticseg.ipynb) |
| `Thesis4_Radaraltimeter_sim.ipynb` | 4 | FMCW radar simulation, P_geo generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis/blob/main/Thesis4_Radaraltimeter_sim.ipynb) |
| `Thesis5_SDF.ipynb` | 5 | Semantic-radar fusion, safety scoring | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kotireddys/masters_thesis/blob/main/Thesis5_SDF.ipynb) |

---

## 🛠️ **Quick Start**

### **Option A: Google Colab (Recommended)**

1. Click any "Open in Colab" badge above
2. Notebooks will auto-install dependencies on first run
3. Mount Google Drive when prompted (for dataset caching)
4. Execute cells sequentially

### **Option B: Local Setup**
```bash
# Clone repository
git clone https://github.com/kotireddys/masters_thesis.git
cd masters_thesis

# Create virtual environment (recommended)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r appendix/requirements.txt

# Launch Jupyter
jupyter notebook
```

### **Required Datasets**

The workflow uses open-access Dutch geospatial data:

1. **AHN4 LiDAR DEM**  
   - Source: [AHN Viewer](https://www.ahn.nl/ahn-viewer)
   - Format: LAZ/LAS point clouds or GeoTIFF DEMs
   - Resolution: 0.5–1 m, vertical accuracy ±5 cm
   - Coverage: All of the Netherlands

2. **PDOK RGB Orthophotos**  
   - Source: [PDOK Services](https://www.pdok.nl/)
   - Format: GeoTIFF via WMTS API
   - Resolution: 10–25 cm/pixel
   - Coordinate System: EPSG:28992 (Amersfoort / RD New)

**Note:** Notebooks 1 and 2 include automated download scripts for sample tiles. Full national coverage requires ~100GB storage.

---

## ▶️ **Workflow Execution**

Execute notebooks in numbered order for full reproducibility:

### **Phase 1: Data Preparation** (Steps 1–2)
```python
# Thesis1_Dataprep.ipynb
- Download AHN4 DEM tiles
- Download corresponding PDOK RGB imagery
- Perform affine reprojection to common CRS
- Crop DEM to exact RGB extent (512×512 px patches)

# Thesis2_datasetcreation.ipynb
- Upload cropped tiles to Roboflow
- Apply augmentations (rotation, flip, brightness)
- Generate 70/20/10 train/val/test splits
- Export dataset YAML for training
```

### **Phase 2: Model Training** (Step 3)
```python
# Thesis3_Semanticseg.ipynb
- Train DeepLabV3+ (ResNet50) via Ultralytics Hub
- Train YOLOv11-Seg for comparison
- Monitor loss curves, mIoU, per-class performance
- Export best checkpoint weights
```

### **Phase 3: Radar Emulation** (Step 4)
```python
# Thesis4_Radaraltimeter_sim.ipynb
- Load DEM patches
- Simulate FMCW radar returns (noise + SNR dropout)
- Compute slope and roughness maps
- Generate P_geo (geometric safety probability)
```

### **Phase 4: Fusion & Validation** (Step 5)
```python
# Thesis5_SDF.ipynb
- Load P_geo and P_sem (from trained model inference)
- Apply fusion: P_safe = P_geo × P_sem
- Threshold and extract connected safe zones
- Rank by area and safety score
- Export GeoJSON polygons
- Compute evaluation metrics (Precision, Recall, IoU, IoS)
```

**Total Execution Time:** ~8 hours on Colab A100 (including training)

---

### **Quantitative Results**

**Confusion Matrix (DeepLabV3+ Test Set):**
- Overall Accuracy: 69.5%
- Misclassification Rate: 30.5%
- Primary confusions: vegetation ↔ grassland, water ↔ wet asphalt

**Case Study Performance:**
- Urban periphery: Precision = 0.82, IoU = 0.74
- Agricultural field: Precision = 0.88, IoU = 0.88
- Mixed terrain: Precision = 0.78, IoU = 0.81

---

## 🔮 **Future Directions**

### **Immediate Next Steps**
- [ ] Hardware-in-the-Loop (HIL) validation with physical Ainstein US-D1 radar
- [ ] Collect real radar-vision paired dataset for domain adaptation
- [ ] Quantize models (INT8) for Jetson Orin Nano deployment

### **System Integration**
- [ ] ROS 2 node implementation for real-time processing
- [ ] PX4 autopilot integration for closed-loop landing control
- [ ] Visual-Inertial Odometry (VIO) for GPS-denied localization

### **Algorithm Enhancements**
- [ ] Multi-path radar scattering simulation
- [ ] Dynamic obstacle detection (moving vehicles, pedestrians)
- [ ] Temporal fusion across multiple descent images
- [ ] Multi-UAV cooperative landing zone sharing

### **Dataset Extensions**
- [ ] Urban, mountainous, and maritime terrain coverage
- [ ] Multi-season data (winter snow, autumn leaves)
- [ ] Adverse weather simulation (fog, rain, dust)

---

## 📚 **Citation**

If you use this work in your research, please cite:
```bibtex
@mastersthesis{syamala2025uavlanding,
  author = {Syamala, Koti Reddy},
  title = {A Proof of Concept for Autonomous UAV Safe Landing Zone Detection 
           in GPS-Denied Unstructured Terrain Using Synthetic Radar Altimeter 
           and Semantic Segmentation},
  school = {Università degli Studi di Napoli "Federico II"},
  year = {2025},
  type = {Master's Thesis},
  address = {Naples, Italy},
  note = {Supervisors: Prof. Giancarmine Fasano, Dr. Dmitry Ignatyev}
}
```

### **Related Publications**

This work builds upon:
- Veneruso et al. (2025) - FMCW radar-aided AAM navigation
- Cho & Jung (2022) - Semantic segmentation for UAM landing
- Benjwal et al. (2023) - Safe landing zone detection via CNNs

---

## 📄 **License**

### **Code License**
This project's **code and scripts** are licensed under the [MIT License](LICENSE).
```
MIT License

Copyright (c) 2025 Koti Reddy Syamala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

### **Thesis Document**
The **thesis document** (PDF) is © 2025 Koti Reddy Syamala and Università degli Studi di Napoli "Federico II". All rights reserved.

### **Dataset Licenses**
- **AHN4 LiDAR:** CC0 1.0 Universal (Public Domain)
- **PDOK Imagery:** CC BY 4.0 (Attribution required)

---

## 📬 **Contact**

**Author:** Koti Reddy Syamala  
**Email:** kotireddy.syamala@studenti.unina.it  
**University:** Università degli Studi di Napoli "Federico II"  
**Program:** Autonomous Vehicle Engineering (LM-MOVE)  

**Thesis Supervisors:**
- Prof. Giancarmine Fasano (Primary Advisor)
- Dr. Dmitry Ignatyev (Co-Advisor)

### **Questions & Issues**
- For **implementation questions**, please open an [issue](https://github.com/kotireddys/masters_thesis/issues)
- For **collaboration inquiries**, email directly
- For **academic correspondence**, contact supervisors via university channels

---

## 🙏 **Acknowledgments**

This work was supported by:
- **Federico II Engineering Faculty** for computational resources
- **Google Colab Pro** for GPU access
- **Roboflow** and **Ultralytics** for dataset/model management platforms
- **AHN** and **PDOK** for open geospatial data

Special thanks to the UAV autonomy research community for foundational work in vision-based landing and radar sensing.

---

## 📊 **Repository Statistics**

![GitHub stars](https://img.shields.io/github/stars/kotireddys/masters_thesis?style=social)
![GitHub forks](https://img.shields.io/github/forks/kotireddys/masters_thesis?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/kotireddys/masters_thesis?style=social)

**Last Updated:** January 2025  
**Status:** ✅ Active Development | Thesis Submitted

---

<div align="center">
  <sub>Built with ❤️ for safer autonomous flight</sub>
</div>
