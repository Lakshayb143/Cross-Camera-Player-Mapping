# 📘 Cross-Camera Player Re-Identification Documentation

## 📌 Title
**Cross-Camera Player Tracking and Re-Identification in Football Matches**

## 👤 Author
Lakshay  
AI Internship Assignment

---

## 🧭 Table of Contents
1. [Introduction](#introduction)
2. [Input Description](#input-description)
3. [System Architecture](#system-architecture)
4. [Modules and Components](#modules-and-components)
5. [Design Decisions](#design-decisions)
6. [Implementation Steps](#implementation-steps)
7. [Challenges and Edge Cases](#challenges-and-edge-cases)
8. [Evaluation and Metrics](#evaluation-and-metrics)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## 🧠 Introduction
This project aims to build a system for tracking and re-identifying football players across two camera angles. It leverages a YOLOv11-based object detection model along with tracking and appearance-based matching strategies to maintain consistent identities across views.

---

## 🎥 Input Description
- **Video 1 (Main Camera):** Overview of the full field.
- **Video 2 (Side Camera):** Closer or different angle of the field.
- Frame rate, dimensions, and sync assumptions will be discussed.
- Screenshots to illustrate scene differences will be added.

---

## 🧱 System Architecture
![Architecture Diagram](docs/architecture.png)

- Preprocessing
- YOLOv11 Detection
- Tracking Module
- Feature Extraction
- Cross-Camera Re-ID Matching
- Visualization

---

## ⚙️ Modules and Components

| Module        | Purpose                                 |
|---------------|------------------------------------------|
| Preprocessing | Frame extraction, resizing, etc.         |
| Detection     | Run YOLOv11 to detect players            |
| Tracking      | Use DeepSORT/ByteTrack for ID tracking   |
| Features      | Extract ReID features (color histograms, etc.) |
| Matcher       | Match players across views               |
| Visualizer    | Overlay IDs on video and generate output |

---

## 🧠 Design Decisions
- Chose YOLOv11 for detection due to pretrained model availability.
- DeepSORT selected for real-time, ID-preserving tracking.
- Re-ID through combination of bounding box crops and color histograms.
- Trade-offs between accuracy and speed discussed here.

---

## 🔨 Implementation Steps
### ✅ Step 1: Preprocessing
- Frame sampling every X frames
- Resizing to YOLOv11 input dimensions

### ✅ Step 2: Detection
- Run inference on each frame with YOLOv11 `.pt` model
- Filter non-player classes

### ✅ Step 3: Tracking
- Apply DeepSORT per video
- Generate unique track IDs

### ✅ Step 4: Feature Extraction
- Crop player images
- Compute color histograms / simple embeddings

### ✅ Step 5: Cross-Camera Matching
- Compare features using cosine similarity or color histogram distances
- Generate ID mapping

### ✅ Step 6: Visualization
- Assign consistent IDs
- Output annotated video clips

---

## ⚠️ Challenges and Edge Cases
- Occlusions from other players
- Camera view changes
- False positives in detection
- Mismatched bounding boxes due to motion blur

---

## 📈 Evaluation and Metrics
- Total players detected and tracked
- Number of correct cross-view re-IDs
- Screenshots and before-after comparisons

---

## 📌 Conclusion and Future Work
- What worked well?
- What would be improved with more time?
- Potential extensions (3-camera setup, fine-tuned ReID model)

---

## 🔗 References
1. YOLOv11 Official Documentation
2. DeepSORT Paper and GitHub Repo
3. OpenCV Python Tutorials
4. SoccerNet Dataset and Benchmarks