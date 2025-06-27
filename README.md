# 📘 Cross-Camera Player Re-Identification Documentation

## 📌 Title
**Cross-Camera Player Tracking and Re-Identification in Football Matches**

 

---

## 🧭 Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Input Description](#input-description)
4. [System Architecture](#system-architecture)
5. [Modules and Components](#modules-and-components)
6. [Design Decisions](#design-decisions)
7. [Implementation Steps](#implementation-steps)
8. [Challenges and Edge Cases](#challenges-and-edge-cases)
9. [Evaluation and Metrics](#evaluation-and-metrics)
  10. [Future Work](#conclusion-and-future-work)
  11. [References](#references)

---

## Introduction
This project aims to build a system for tracking and re-identifying football players across two camera angles. It leverages a YOLO-based object detection model along with tracking and appearance-based matching strategies to maintain consistent identities across views.

---

## Project Structure

```html
Cross-Camera-Player-Mapping/
│
├── artifacts/                    # Contains input videos and model weights
│   ├── broadcast.mp4
│   ├── tacticam.mp4
│   └── best.pt                   # YOLO pretrained model
│
├── src/                          # Core logic and low-level processing
│   │                    
│   ├── components/                    # strategy implementations
│   │ 
│   ├── interfaces/                    # Interfaces for design patterns
│   │
│   ├── steps/                         # Pipeline steps
│  
├── utils/                    # Utility functions (logging, helpers)
│
├── Research.md                   # Knowledge Base Used in This Project
├── README.md                     # Project Documentation
|
├── pyproject.toml                # Information about project
|
├── application.py                # Entry Point of the application
├
└── requirements.txt              # Required Python packages
```

---

## 🎥 Input Description
- **Video 1 :** Broadcast video (maybe from side angle)
- **Video 2 :** Tacticam video (top view or higher view)
<!-- - Frame rate, dimensions, and sync assumptions will be discussed. -->


---

## 🧱 System Architecture

- Frame Extraction and Synchronization (SSIM Cross-Correlation)
- Normalization and transformation
- YOLO Detection
- Tracking Module
- Feature Extraction
- Cross View Matcher Re-ID Matching
- Visualization

---

## Components

| Module        | Purpose                                 |
|---------------|------------------------------------------|
| FrameExtractor.py | Frame extraction from videos using `ffmpegcv`
| Synchronizer.py | Synchorization between frames.        |
| ViewTransformer.py     | Calculates Homography Matrix and Warps views.
| PlayerTracker.py      | Use ByteTrack/Torchreid for ID tracking   |
| FeatureExtractor.py    | Extract features (color histograms, etc.) |
| CrossViewMatcher.py       | Match players across views               |
| IDManager.py       | Global ID Manger across views.              |


---

## Design Decisions
- Implmented various LLD (Low Level Design) techniques to make code simple, clean, modular and scalable.
- Chose YOLO for detection due to pretrained model availability.
- Torchreid model - `osnet_x0_25` selected for real-time, ID-preserving tracking.
- Re-ID through combination of bounding box crops and color histograms.
- Used `Docker` for contanerization of the application.

---

## 🔨 Implementation Steps
### ✅ Step 1: Preprocessing
- Frame sampling every X frames
- Video synchronization
- Normalizaton and transformation using homography with `cv2.RANSAC`

### ✅ Step 2: Detection and Tracking
- Detection and tracking of players on each frame with YOLO `.pt` model
- Used `torchreid` model (`osnet_x0_25`) for id-preserving tracking.


### ✅ Step 3: Feature Extraction
- Used various features are like field coordinates, appearance embedding using models, etc.
- Applied feature weight as hyper parameter.

### ✅ Step 4: Cross-Camera Matching
- Used feature based cost matrix and `scipy.optimize.linear_sum_assignment` for optimization.


---

## Challenges faced and Edge Cases

### Homography Inaccuracy

* **Analysis:** Initial attempts to map one camera directly to another resulted in significant warping errors. The core issue was the lack of a stable, independent reference frame.
* **Solution:** Re-architected the entire calibration process to use a **canonical 2D field map** as the "common ground." This immediately stabilized the geometry. Further refined the solution by using a large set of **24 correspondence points** with the RANSAC algorithm, making the transformation resilient to minor human error in point selection.

### Cross-View Matching Ambiguity

* **Analysis:** Relying on a single feature for matching (e.g., only appearance) was fragile and failed during player occlusions or when players with similar appearances were close together.
* **Solution:** Engineered a **multi-feature system** with a weighted cost matrix. This makes the matching logic resilient by design. If the appearance feature is weak for a given pair (e.g., they are far away), a strong match in their on-field coordinates can compensate, leading to a correct overall assignment.

---

## Evaluation and Metrics
- 

---

## Future Work
* **Transition to Tracklet-Level Matching:** Shift from frame-by-frame matching to matching entire tracklets. 

*  **Integrate State-of-the-Art Re-ID Models:** Using the research paper "An enhanced `Swin Transformer` for soccer player re-identification".

* **Real-Time Performance Optimization:** Investigate and implement optimizations for live-streaming analysis, including GPU acceleration at all stages.

---

## References

#### [📚 Knowledge Base Used in This Project ](Research.md) 
– Detailed list of research papers, tools, and documentation reviewed and applied during development.


## 👤 Author
Lakshay 