# 🚗 Advanced Lane Detection from Scratch (OpenCV + NumPy)

This project implements an advanced lane detection system **without using any machine learning models**, relying solely on classical computer vision techniques using **OpenCV** and **NumPy**. The pipeline processes video frames (or static road images) and detects lane boundaries, overlays them back on the road, and calculates lane curvature and vehicle position relative to the center.

Note : The parameters are based on the images I used for testing
---

## 📽️ Demo

> 📹 *Check out the lane detection in action below!*  
[![Watch Demo](readme/readme_image.PNG)](https://drive.google.com/file/d/1vWD7798FkGMFnbIDs0_xnnO9QUf1SkcO/view?usp=sharing)

---

## ▶️ How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/Haseebasif7/Advance-Lane-Detection.git
cd Advance-Lane-Detection
```

2. **Install dependencies:**

Make sure you have Python installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

3. **Run the pipeline:**

```bash
python lane_pipeline/main.py
```


---

## 🧩 Pipeline Overview

The lane detection pipeline involves several key steps:

### 1️⃣ Camera Calibration

We start by calibrating the camera using multiple chessboard images. This helps us compute the **camera matrix** and **distortion coefficients** which are used to correct lens distortion. Without this step, road lines can appear curved or warped due to the fisheye effect.

### 2️⃣ Distortion Correction

Using the calibration data, we correct any distortions in the input road image. This ensures the lane lines are geometrically accurate, which is critical for the rest of the pipeline.

### 3️⃣ Perspective Transform (Bird's-Eye View)

We apply a perspective transformation to get a **top-down view** of the road. This rectified view simplifies lane detection by making the lanes appear parallel and consistent in shape and width.

### 4️⃣ Binary Thresholding

We use **color filtering** (e.g., R-Binary) to highlight lane line pixels. The result is a binary image where potential lane pixels are white, and the rest is black.

### 5️⃣ Lane Pixel Detection

We use a **histogram-based sliding window technique** to find pixels belonging to the left and right lanes. A polynomial curve is then fitted through these points to estimate the lane boundary lines.

### 6️⃣ Lane Curvature & Vehicle Offset

We calculate:
- **Radius of curvature** of the lane (how sharply it's turning).
- **Vehicle's offset from the center** of the lane using pixel-to-meter conversion.

These metrics are crucial for real-world driving applications like ADAS.

Finally, the detected lane lines are **warped back** to the original image perspective, and both visual and numerical information is overlayed on the frame.

---
