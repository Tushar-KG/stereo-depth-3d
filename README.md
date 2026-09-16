# Stereo Depth Estimation + 3D Reconstruction

> **Classical Computer Vision only — no deep learning, no ML models.**  
> OpenCV · NumPy · Python 3.8+

![pipeline](https://img.shields.io/badge/pipeline-classical_CV-blue)
![deps](https://img.shields.io/badge/deps-opencv%20%7C%20numpy%20%7C%20pytest-green)

---

## Overview

This project implements the full **passive stereo depth estimation** pipeline from scratch
using only classical computer vision techniques.  Given two photographs taken from cameras
separated by a known horizontal distance (a *stereo pair*), it produces:

| Output | File | Description |
|--------|------|-------------|
| Disparity map | `output/disparity_maps/disparity.png` | Grayscale; brighter = closer |
| Depth colormap | `output/disparity_maps/depth_colormap.png` | JET; red = close, blue = far |
| 3-D point cloud | `output/point_clouds/reconstruction.ply` | ASCII PLY; open in MeshLab/CloudCompare/Blender |

---

## Features

- **Stereo Block Matching** — both `StereoBM` (fast) and `StereoSGBM` (denser, recommended)
- **Epipolar rectification** — full `cv2.stereoRectify` / `cv2.remap` pipeline for real cameras
- **Metric depth** — `Z = (f × B) / d` with median-blur cleaning and far-clip
- **Pinhole back-projection** — `X = (u − cx)·Z/f`, `Y = (v − cy)·Z/f`
- **Hand-written ASCII PLY** — no Open3D, no libpcl; pure Python + NumPy
- **Built-in synthetic scene** — works out-of-the-box with `--generate-sample`, no camera needed
- **Checkerboard calibration** — `calibrate_from_checkerboard()` for real hardware
- **26 pytest tests** — covers every module and the full pipeline end-to-end

---

## Tech Stack

| Library | Role |
|---------|------|
| `opencv-python-headless` | Image I/O, stereo matching, rectification, colormaps |
| `numpy` | Array math, back-projection, PLY formatting |
| `pytest` | Unit & integration tests |

---

## Project Structure

```
stereo-depth-3d/
├── README.md
├── statement.md                    ← Problem statement (course deliverable)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── main.py                     ← CLI entry point (argparse)
│   ├── calibration.py              ← load_calibration_json, calibrate_from_checkerboard
│   ├── rectification.py            ← rectify_pair (epipolar rectification)
│   ├── disparity.py                ← compute_disparity (BM / SGBM)
│   ├── depth_estimation.py         ← disparity_to_depth, clean_depth_map
│   ├── reconstruction.py           ← depth_to_pointcloud, save_ply
│   └── generate_sample.py          ← synthetic stereo scene generator
│
├── data/
│   └── sample_stereo_pairs/        ← generated at runtime (gitignored)
│       ├── left.png
│       ├── right.png
│       └── calib.json
│
├── output/                         ← generated at runtime (gitignored)
│   ├── disparity_maps/
│   │   ├── disparity.png
│   │   └── depth_colormap.png
│   └── point_clouds/
│       └── reconstruction.ply
│
└── tests/
    └── test_pipeline.py
```

---

## Setup

### 1 — Clone / download the project

```bash
git clone <repo-url>
cd stereo-depth-3d
```

### 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Demo (Synthetic Scene — No Camera Needed)

```bash
python src/main.py --generate-sample
```

This:
1. Renders a synthetic scene with 5 textured rectangles at depths 1.2 m, 1.8 m, 2.5 m, 4.0 m, 6.0 m.
2. Runs the full pipeline (rectify → disparity → depth → point cloud).
3. Saves all outputs to `output/`.

Expected output:
```
────────────────────────────────────────────────────
  Step 1 / 6 — Loading images
────────────────────────────────────────────────────
[generate_sample] Written to: data/sample_stereo_pairs
  focal_length_px = 700.0
  baseline_m      = 0.1000
  ...

════════════════════════════════════════════════════
  Pipeline complete in X.XX s
  Output directory : stereo-depth-3d/output
════════════════════════════════════════════════════

  Outputs:
    Disparity map  : output/disparity_maps/disparity.png
    Depth colormap : output/disparity_maps/depth_colormap.png
    Point cloud    : output/point_clouds/reconstruction.ply
```

---

## Running on Real Stereo Photos

### Step 1 — Capture images

Take two photos of the same scene from two camera positions separated by a known horizontal
distance (the *baseline*).  The cameras must be looking in the same direction and the baseline
must be purely horizontal for the pre-rectification assumption to hold.

Alternatively, use a factory-calibrated stereo camera (e.g. Intel RealSense, ZED) and
export the rectified frames as PNG/JPG.

### Step 2 — (Optional) Run checkerboard calibration

If you have a printed checkerboard pattern and want metric depth in real units:

```python
from src.calibration import calibrate_from_checkerboard

result = calibrate_from_checkerboard(
    image_glob   = "data/calib/*.jpg",
    pattern_size = (9, 6),       # inner corners per row / column
    square_size_m= 0.025,        # physical square size in metres
)
print(result)
```

Then save the result as `calib.json`:

```json
{
    "focal_length_px": 712.4,
    "baseline_m":       0.12,
    "cx":             319.5,
    "cy":             239.5
}
```

### Step 3 — Run the pipeline

```bash
python src/main.py \
    --left  data/left.jpg \
    --right data/right.jpg \
    --calib data/calib.json
```

---

## CLI Flag Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--generate-sample` | off | Generate & use a synthetic stereo pair |
| `--left PATH` | — | Path to the left image |
| `--right PATH` | — | Path to the right image |
| `--calib PATH` | — | Path to `calib.json` |
| `--method bm\|sgbm` | `sgbm` | Stereo matching algorithm |
| `--num-disparities N` | `96` | Disparity search range (multiple of 16) |
| `--block-size N` | `7` | Matching window size (odd integer) |
| `--max-depth M` | `20.0` | Far-clip threshold in metres |
| `--output DIR` | `output` | Root directory for all output files |

---

## Running Tests

```bash
pytest tests/ -v
```

All 26 tests should pass.  Tests use `tmp_path` fixtures — no hardcoded paths,
fully isolated.

To run a specific class:
```bash
pytest tests/test_pipeline.py::TestEndToEnd -v
```

---

## Viewing the Point Cloud (`.ply`)

| Viewer | How to open |
|--------|-------------|
| **MeshLab** | File → Import Mesh → `reconstruction.ply` |
| **CloudCompare** | File → Open → `reconstruction.ply` |
| **Blender** | File → Import → Stanford PLY (`.ply`) |
| **Text editor** | It's ASCII — open directly to inspect the header and data |

---

## Known Limitations

### 1 — Textureless regions produce noisy disparity
Block matching relies on local texture contrast.  Flat-coloured walls, white ceilings, and
smooth surfaces produce ambiguous matches → the disparity map will have holes or noise in
those areas.  Remedy: use SGBM (larger smoothness penalty) or increase `--block-size`.

### 2 — Rectification is a no-op without full stereo calibration
`rectify_pair()` requires a full stereo calibration dict (K_left, K_right, D_left, D_right,
R, T).  Without it, the images are returned unchanged.  If your stereo pair is not already
rectified, depth estimates will be incorrect because the epipolar lines won't be horizontal.

### 3 — Accuracy depends heavily on calibration quality
The depth formula `Z = (f × B) / d` is only as accurate as `f` (focal length) and `B`
(baseline).  A 5% error in focal length → 5% error in all depth values.
Always calibrate with a proper checkerboard and many images.

### 4 — Depth degrades quadratically at long range
At large depths the disparity becomes very small (e.g. < 1 px), and a 1-pixel error
translates to a large depth error.  Passive stereo is most accurate at short range
(< 5 × baseline distance, i.e. < 0.5 m for a 10 cm baseline).

### 5 — The synthetic scene tests rectification and geometry, not real-world robustness
The generator uses known-good geometry, so the point cloud from the synthetic scene looks
clean.  Real images will have lens distortion, lighting variation, and occlusion — all of
which degrade results.

---

## Future Enhancements

- **WLS filter** (Weighted Least Squares) post-processing for disparity refinement
- **Binary PLY** export for much smaller file sizes
- **Stereo video** support: process frame-by-frame, output depth video
- **Full stereo calibration wizard**: capture → detect → calibrate → save, all in one command
- **Confidence map**: export a per-pixel reliability score alongside depth
- **ROS2 node wrapper**: publish depth and point cloud to ROS topics for robotics integration
- **CUDA acceleration**: swap `cv2.StereoSGBM_create` for `cv2.cuda.createStereoSGM` on GPU

---

## Mathematical Background

### Disparity → Depth

```
Z = (focal_length_px × baseline_m) / disparity_px
```

Derived from similar triangles in the stereo rig geometry (epipolar geometry).

### Pinhole Back-Projection (Depth → 3D)

```
X = (u - cx) * Z / focal_length_px
Y = (v - cy) * Z / focal_length_px
Z = Z
```

where `(u, v)` is the pixel position and `(cx, cy)` is the principal point.

### Stereo Rectification

Applies rotation matrices R1, R2 (from `cv2.stereoRectify`) to both images so that
corresponding points lie on the same horizontal scan line, enabling efficient 1-D
disparity search.

---

by Tushar Kant Gupta
