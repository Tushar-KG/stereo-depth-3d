# Sample Output — Pre-generated Results

This directory contains pre-generated output from running:

```bash
python src/main.py --generate-sample
```

It is committed to the repository so evaluators can verify results immediately
**without running the pipeline themselves**.

---

## Contents

| File | Description |
|------|-------------|
| `stereo_pairs/left.png` | Synthetic left-eye image (640×480) |
| `stereo_pairs/right.png` | Synthetic right-eye image (640×480, shifted by parallax) |
| `stereo_pairs/calib.json` | Simple intrinsics: focal_length_px=700, baseline=0.1m, cx=320, cy=240 |
| `stereo_pairs/stereo_calib.json` | Full stereo calibration (K, D, R, T) used to drive `cv2.stereoRectify` |
| `disparity_maps/disparity.png` | Grayscale disparity map (brighter = closer) |
| `disparity_maps/depth_colormap.png` | JET-colourised depth map (red = near, blue = far) |
| `point_clouds/reconstruction.ply` | ASCII PLY point cloud — 10,000 sampled points |

---

## Scene Description

The synthetic scene contains **5 textured rectangles** at known depths:

| Depth | Expected Disparity |
|-------|--------------------|
| 1.2 m | ~58.3 px |
| 1.8 m | ~38.9 px |
| 2.5 m | ~28.0 px |
| 4.0 m | ~17.5 px |
| 6.0 m | ~11.7 px |

Disparity computed with **SGBM** (Semi-Global Block Matching), num_disparities=96, block_size=7.

Rectification was performed using the **full `cv2.stereoRectify` + `cv2.remap` path**
with the synthetic rig's known K, D, R, T matrices (zero distortion, identity rotation,
horizontal translation = −0.1 m).

---

## Viewing the Point Cloud

Open `point_clouds/reconstruction.ply` in:
- **MeshLab** — File → Import Mesh
- **CloudCompare** — File → Open
- **Blender** — File → Import → Stanford PLY
- **Text editor** — it's plain ASCII
