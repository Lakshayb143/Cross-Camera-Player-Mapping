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
  10. [Conclusion and Future Work](#conclusion-and-future-work)
  11. [References](#references)

---

## Introduction
This project aims to build a system for tracking and re-identifying football players across two camera angles. It leverages a YOLOv11-based object detection model along with tracking and appearance-based matching strategies to maintain consistent identities across views.

---

## Project Structure

```text
Cross-Camera-Player-Mapping/
│
├── artifacts/                    # Contains input videos and model weights
│   ├── broadcast.mp4
│   ├── tacticam.mp4
│   └── best.pt                   # YOLOv11 pretrained model
│
├── core/                         # Core logic and low-level processing
│   ├── components/               # strategy implementations
│   │   ├── ExtractionStrategies.py
│   │   ├── ModelStrategies.py
│   │   ├── SynchronizationStrategies.py
│   │   └── __init__.py
│   │
│   ├── interfaces/               # Interfaces for design patterns
│   │   ├── IExtractor.py
│   │   ├── ISynchronizer.py
│   │   └── IModel.py
│   │
│   ├── steps/                    # Pipeline steps
│   │   ├── FrameExtractor.py
│   │   ├── Synchronizer.py
│   │   ├── Detector.py
│   │   └── Tracker.py
│   │
│   └── utils/                    # Utility functions (logging, helpers)
│       ├── logger.py
│
├── Research.md                   # Knowledge Base Used in This Project
├── README.md                     
└── requirements.txt              # Required Python packages
```

---

## 🎥 Input Description
- **Video 1 :** Overview of the full field.
- **Video 2 :** Closer or different angle of the field.
<!-- - Frame rate, dimensions, and sync assumptions will be discussed. -->


---

## 🧱 System Architecture

- Preprocessing
- YOLOv11 Detection
- Tracking Module
- Feature Extraction
- Cross-Camera Re-ID Matching
- Visualization

---

## Modules and Components

| Module        | Purpose                                 |
|---------------|------------------------------------------|
| Preprocessing | Frame extraction, resizing, etc.         |
| Detection     | Run YOLOv11 to detect players            |
| Tracking      | Use DeepSORT/ByteTrack for ID tracking   |
| Features      | Extract ReID features (color histograms, etc.) |
| Matcher       | Match players across views               |
| Visualizer    | Overlay IDs on video and generate output |

---

## Design Decisions
- Implmented various LLD (Low Level Design) techniques to make code simple, clean, modular and scalable.
- Chose YOLOv11 for detection due to pretrained model availability.
- DeepSORT selected for real-time, ID-preserving tracking.
- Re-ID through combination of bounding box crops and color histograms.
- Trade-offs between accuracy and speed discussed here.

---

## 🔨 Implementation Steps
### ✅ Step 1: Preprocessing
- Frame sampling every X frames
- Video synchronization

### ✅ Step 2: Detection
- Run inference on each frame with YOLOv11 `.pt` model
- 

### ✅ Step 3: Tracking
- 
- 

### ✅ Step 4: Feature Extraction
- 
- 

### ✅ Step 5: Cross-Camera Matching
- 
- 

### ✅ Step 6: Visualization
- 
- 

---

## Challenges faced and Edge Cases
- **Occlusions from other players** - Players often overlap or disappear briefly. We can use appearance embeddings
help to reacquire identity.
- **Incomplete Field Overlap** - If a player appears in one view but is out of sight in the other, we can leave it
unmapped 
- **Similar Uniforms** - Teammates look almost identical.
- **Temporary ID Swaps** - If one player overtakes another, trackers sometimes swap IDs.  Mitigate this by
using strong appearance constraints

---

## Evaluation and Metrics
- 

---

## Conclusion and Future Work
- 

---

## References

#### [📚 Knowledge Base Used in This Project ](Research.md) 
– Detailed list of research papers, tools, and documentation reviewed and applied during development.


## 👤 Author
Lakshay 