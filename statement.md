# Problem Statement

## Project Title
**Stereo Depth Estimation and 3D Reconstruction using Classical Computer Vision**

---

## Problem Description

Estimating the three-dimensional structure of a scene from two-dimensional images is one of the
fundamental challenges in computer vision. Humans perceive depth effortlessly using binocular
vision — our two eyes capture slightly different views of the same scene, and our brain fuses
them to infer distances. This project replicates that process computationally using a **stereo
camera pair** and entirely **classical (non-learning) algorithms**.

Given two photographs of the same scene taken simultaneously from cameras separated by a known
horizontal distance (the *baseline*), the system computes:

1. **Disparity Map** — for each pixel in the left image, how many pixels to the left is the
   matching feature in the right image? Disparity is inversely proportional to depth.
2. **Depth Map** — converting disparity to metric depth (in metres) via the relation
   `depth = (focal_length × baseline) / disparity`.
3. **3D Point Cloud** — back-projecting every valid depth pixel into 3D space using the pinhole
   camera model, producing a coloured `.ply` file viewable in MeshLab, CloudCompare, or Blender.

---

## Scope

| In Scope | Out of Scope |
|----------|-------------|
| Classical stereo matching (BM, SGBM) | Deep learning / neural disparity networks |
| Pinhole camera model | Fisheye / omnidirectional cameras |
| ASCII PLY export | Binary PLY, OBJ, PCD formats |
| Checkerboard stereo calibration | IMU / LiDAR fusion |
| Synthetic scene generation for testing | Real-time video streaming |

---

## Target Users

- **Computer Vision students** learning epipolar geometry and 3D reconstruction
- **Robotics engineers** prototyping obstacle-distance estimation on a budget
- **Researchers** who need a reproducible baseline without GPU infrastructure

---

## High-Level Features

### 1 — Classical CV Only
All algorithms are deterministic, interpretable, and GPU-free.  No PyTorch, TensorFlow,
ONNX, or any pretrained model is used.  The complete computational graph is expressed in
OpenCV and NumPy primitives.

### 2 — Stereo Block Matching
Implements both `StereoBM` (fast, sparse) and `StereoSGBM` (Semi-Global Block Matching —
slower but denser) from OpenCV.  SGBM applies a path-aggregation cost that enforces
smoothness along multiple scan-line directions, substantially reducing stripe artefacts on
weakly-textured surfaces.

### 3 — Epipolar Geometry & Rectification
For uncalibrated pairs the images are returned as-is (assuming pre-rectification).
For calibrated pairs the full `cv2.stereoRectify` + `cv2.initUndistortRectifyMap` +
`cv2.remap` pipeline aligns each row of the left and right images to the same epipolar line,
which is a prerequisite for horizontal-only disparity search.

### 4 — Pinhole Back-Projection
Metric 3D coordinates are recovered as:
```
X = (u - cx) * depth / focal_length
Y = (v - cy) * depth / focal_length
Z = depth
```
where `(cx, cy)` is the principal point and `focal_length` is in pixels.

### 5 — Synthetic Test Scene
A built-in scene generator renders four textured rectangles at depths 1.2 m, 1.8 m, 2.5 m,
and 4.0 m, shifts them horizontally between left/right frames to simulate true parallax, and
writes a `calib.json` so the pipeline can be validated end-to-end without physical hardware.

### 6 — No External Point-Cloud Libraries
The `.ply` file is written by hand (ASCII header + one `x y z r g b` line per point).
This makes the output inspectable with any text editor and removes any dependency on Open3D
or libpcl.
