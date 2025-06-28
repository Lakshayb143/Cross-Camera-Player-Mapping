# Research & Technical References



## 🧭 Table of Contents
1. [Documentations & Articles](#documentations)
2. [Research Papers](#research-papers)
3. [Tools](#tools)
4. [Theory](#theory)

---

## Documentations & Articles

- [Ultralytics Docs - Yolo](https://docs.ultralytics.com/)

- [An enhanced Swin Transformer for soccer player reidentification](https://www.nature.com/articles/s41598-024-51767-4#:~:text=exist%20between%20sport%20player%20ReID,identify%20players.%20These%20problems%20are)

- [Swin Transformer Based on Two-Fold Loss and Background Adaptation Re-Ranking for Person Re-Identification](https://www.mdpi.com/2079-9292/11/13/1941)


- [Camera Calibration in Sports with Keypoints - Roboflow](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)

- [Homography examples using OpenCV](https://learnopencv.com/homography-examples-using-opencv-python-c/)

- [Tracking multiple objects with OpenCV](https://pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/)

- [Object Tracking with YOLOv8 and Python](https://pyimagesearch.com/2024/06/17/object-tracking-with-yolov8-and-python/)

- [Soccer Analytics Handbook](https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb)
---

## Research Papers


- [Structural Similarity Index (SSIM): Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf)

- [YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID](https://arxiv.org/abs/2501.13710)

- [Look, Listen and Learn](https://arxiv.org/abs/1705.08168#)
    
    

---

## Tools

- **OpenCV**: Library used for computer vision.

- **FFmpeg**: The command-line tool that powers most video processing. [website](https://ffmpeg.org/)

    #### Github Repos
    - [Repo 1](https://github.com/ML-KULeuven/)
    - [Repo 2](https://github.com/Friends-of-Tracking-Data-FoTD)

- **Torch Reid** -Torchreid is a library for deep-learning person re-identification, written in PyTorch [webiste](https://github.com/KaiyangZhou/deep-person-reid)
    


---
## Theory




### Synchronization

#### Strategies for Synchronization

Different strategies for Video Synchronization

#### 1\. Audio Cross-Correlation

This is one of the fastest and most effective methods if the videos have usable audio.

  * **How it Works:** It extracts the audio waveforms from both videos as 1D numerical arrays. It then performs a cross-correlation on these arrays. The result is a new array where the peak's position indicates the time offset that best aligns the two audio signals. It's especially effective with sharp, distinct sounds like a referee's whistle, a starting pistol, or a ball kick.
  * **Pros:**
      * **Extremely Fast:** 1D correlation is computationally much cheaper than 2D image comparison.
      * **Highly Accurate:** Can often achieve sub-frame accuracy.
      * **Robust to Visual Changes:** Completely unaffected by camera movement, lighting changes, or visual obstructions.
  * **Cons:**
      * Requires both videos to have clear, synchronized audio tracks.
      * Fails with very noisy/ambient audio or if one track is missing.
      * Can be confused by repetitive, non-unique sounds.
  * **Best For:** Events where a distinct, shared audio event occurs (e.g., most field sports, races).
  * **Implementation:**
    1.  Use `ffmpeg-python` or a command-line call to extract audio as `.wav` files.
    2.  Use `scipy.io.wavfile.read` to load the audio into NumPy arrays.
    3.  Use `scipy.signal.correlate` to find the cross-correlation and `numpy.argmax` to find the peak, which gives you the offset in audio samples.
    4.  Convert the sample offset to a frame offset using the audio sample rate and video FPS.

#### 2\. Visual Phase Correlation

This is a more advanced visual technique that works in the frequency domain.

  * **How it Works:** Instead of comparing pixels directly, it converts two frames to the frequency domain using a Fast Fourier Transform (FFT). Phase correlation is exceptionally good at finding translational shifts (x, y offsets) between two images, and its confidence peak is a strong indicator of similarity. We can use this confidence peak to find the best temporal offset.
  * **Pros:**
      * Highly robust to changes in lighting, brightness, and contrast.
      * Computationally efficient and built directly into OpenCV.
  * **Cons:**
      * Primarily designed for translational shifts, so it's less effective if there are significant rotation or scaling differences between the two camera views.
  * **Best For:** Scenarios with challenging lighting conditions or where cameras are relatively stable.
  * **Implementation:**
      * Use `cv2.phaseCorrelate` on pairs of preprocessed (grayscale, resized) frames for different time offsets.
      * The offset that returns the highest confidence peak from `cv2.phaseCorrelate` is the best match.

#### 3\. Feature-Based Matching (e.g., Scoreboard/Clock OCR)

This is an "intelligent" content-aware method.

  * **How it Works:** You train a model (or use a pre-trained one) to first detect a specific, reliable object in both views, like the game clock or scoreboard. Then, you apply Optical Character Recognition (OCR) to read the time from the detected region. You can then map the frame number in each video to the actual game time, creating a perfect alignment.
  * **Pros:**
      * Extremely accurate and unambiguous if the OCR is reliable. It aligns based on the actual "ground truth" of the game time.
  * **Cons:**
      * Much more complex to implement. Requires a reliable object detection pipeline (e.g., YOLO to find the clock) *and* a robust OCR pipeline (e.g., `pytesseract`).
      * Fails if the clock/scoreboard is not visible in both views, is obscured, or if the OCR fails.
  * **Best For:** Broadcast videos where a digital clock is consistently visible.

#### 4\. Manual Synchronization Interface (Human-in-the-Loop)

Sometimes, the best solution is to let a human do it.

  * **How it Works:** Create a simple tool (using OpenCV's UI functions, or a GUI library like PyQt/Tkinter) that plays both videos side-by-side. The user can pause the videos and adjust an offset slider until a key event (like a ball being kicked) is perfectly aligned. They then save the offset.
  * **Pros:**
      * Guaranteed to be accurate from a human perspective.
      * A perfect fallback when all automated methods fail due to strange conditions.
  * **Cons:**
      * Not automated and does not scale. It requires manual labor for every pair of videos.
  * **Best For:** A small number of very important videos, or as a fallback/validation tool for your automated systems.

### Summary Table of Strategies

| Strategy | How It Works | Pros | Cons | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Visual SSIM Correlation** | Slides frame sequences, finds max similarity using SSIM. | Good general-purpose, purely visual, handles content changes well. | Can be slow, sensitive to drastic lighting/perspective shifts. | A robust default choice when audio is unavailable. |
| **Audio Cross-Correlation** | Finds the peak alignment of 1D audio waveforms. | Very fast, very accurate, immune to visual issues. | Useless without clear, synchronized audio and distinct sound events. | Sports with whistles, races with starting guns, interviews with clappers. |
| **Visual Phase Correlation**| Finds the best frame alignment using frequency-domain analysis. | Fast, extremely robust to lighting/contrast changes. | Less effective with large rotation/scale changes between views. | Videos with difficult or inconsistent lighting. |
| **Feature-Based (OCR)** | Reads a game clock in both views and aligns based on the time. | Potentially the most accurate, uses ground truth time. | Complex to implement, requires reliable detection and OCR. | Professional broadcasts with a consistent on-screen clock. |
| **Manual Interface** | A human user visually aligns the videos and sets the offset. | Perfect accuracy (human-validated), great fallback. | Not automated, does not scale. | Final validation, or when all automated methods fail. |


---
---

### 🧠 What Is Homography — In Simple Terms?

Homography is a **mathematical transformation** that maps points from one plane to another. Think of it as a way to "warp" the view from one camera so it looks like it's from another angle.



* **Broadcast video**: side-view of football match.
* **Tacticam video**: top-down (or elevated) view of same match.

You use homography to **project both views onto a common 2D plane**, usually the football field. Once aligned, players in both videos can be spatially compared or even matched based on their position on the field.

---

### 🏁 Why Do You Need It?

For **player re-identification**:

* Tracking ID `42` in broadcast and ID `17` in tacticam: you want to know they are the same person.
* If both views are aligned (via homography), comparing player **positions, motion, or appearance** becomes easier.
* Homography is the bridge to align different camera angles.

---

### ✅ What Points Should You Mark?

You must select **corresponding points** in both videos. These should be **fixed field features** that are visible in both views.

#### Good examples:

| Real-world feature         | Why it's good                  |
| -------------------------- | ------------------------------ |
| 4 corners of penalty box   | Well-defined & always present  |
| Corners of center circle   | Symmetric & fixed              |
| Sideline–goal intersection | Sharp corners, visible in both |

* ⚠️ Do **not** click on players! They move — not reliable.
* Choose **lines** or **points** that are:

  * Flat (on the ground),
  * Clearly visible in both videos,
  * Far enough apart (not too close).



---

### 👨‍🔬 How This Helps Player Re-ID

* Instead of matching by just visual appearance (which is hard across angles), you can now:

  * Match by position.
  * Combine appearance + spatial location + motion.

This is called **cross-view player tracking** and is a common technique in multi-camera sports analytics.

---



## 🎯 So, What Are Destination Points?

Think of destination points as the **reference view** or the **"ideal" layout** you're mapping to. The choice depends on:

### 🎥 Your Two Video Types

* **Broadcast video**: perspective from the side, often at an angle.
* **Tacticam**: top-down or tactical overhead view (more like a 2D map of the field).

### 👇 Practical Choices for Destination Points:

#### ✅ Option 1: Select Corresponding Points from Tacticam Video

* Pause a frame from the **tacticam video**.
* Select the **same 4 logical points** (e.g., corners of penalty box, center circle, field corners).
* Use these as `dst_pts`.

That gives you:

```python
H, status = cv2.findHomography(src_pts_broadcast, dst_pts_tacticam)
```

Then you can align broadcast frames to match the tacticam's view using:

```python
warped = cv2.warpPerspective(broadcast_frame, H, (width, height))
```

---

#### ✅ Option 2: Map to a Predefined Field Layout

If tacticam is not usable or noisy, you can:

* Use a **fixed top-down layout** of the soccer field (e.g., 105x68 meters).
* Define destination points manually as pixel locations in a synthetic image.

Example:

```python
# Map top-left, top-right, bottom-right, bottom-left of field
dst_pts = np.float32([
    [0, 0],           # top-left
    [600, 0],         # top-right
    [600, 400],       # bottom-right
    [0, 400]          # bottom-left
])
```

---

## 📸 Visualization Tip

To verify if your homography is good:

```python
# Draw points on original frame
for pt in src_pts:
    cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)

# Warp frame using H
warped = cv2.warpPerspective(frame, H, (600, 400))

# Display both
cv2.imshow("Original", frame)
cv2.imshow("Warped", warped)
```

---

## ✅ Summary

| What?                      | How to Choose Destination Points                                     |
| -------------------------- | -------------------------------------------------------------------- |
| Matching Tacticam          | Click same logical points from that view                             |
| Mapping to 2D Field Layout | Use fixed pixel coordinates for field corners                        |
| Synthetic Layout           | Good if you want a bird’s eye transformation for all broadcast clips |



