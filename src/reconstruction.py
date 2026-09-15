"""
reconstruction.py
=================
3-D point cloud generation and ASCII PLY export.

Classical CV background -- pinhole back-projection
-------------------------------------------------
The pinhole camera model defines the forward projection:

    u = f * (X / Z) + cx          (column in image)
    v = f * (Y / Z) + cy          (row in image)

Inverting this, given a pixel (u, v) and its metric depth Z, the 3-D
coordinates in *camera space* are:

    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    Z = Z  (depth along the optical axis, positive into the scene)

This is called *back-projection* and is the fundamental operation that takes
a 2-D depth image and lifts it into 3-D space.

PLY file format (ASCII variant)
--------------------------------
PLY (Polygon File Format / Stanford Triangle Format) is the simplest standard
for point clouds.  An ASCII PLY file has:
  1. A plain-text header declaring element types and properties.
  2. One data row per element (here, one per point: x y z r g b).

Example header for an N-point coloured cloud:

    ply
    format ascii 1.0
    element vertex N
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    x0 y0 z0 r0 g0 b0
    ...

No external library is required -- we write the file line-by-line.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth: np.ndarray,
    color_img: np.ndarray,
    focal_length_px: float,
    cx: float,
    cy: float,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map into a 3-D point cloud.

    Implements the pinhole back-projection formula:
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        Z = Z

    where (u, v) is the pixel position in the left (reference) image.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W) in metres.  NaN marks invalid pixels.
    color_img : np.ndarray
        Left image used to sample point colours (H, W, 3) BGR or (H, W) gray.
    focal_length_px : float
        Camera focal length in pixels.
    cx : float
        Principal point x-coordinate (usually ~ image_width / 2).
    cy : float
        Principal point y-coordinate (usually ~ image_height / 2).
    max_points : int or None
        If set, randomly subsample to at most *max_points* points (useful to
        keep PLY files manageable for large images).

    Returns
    -------
    points : np.ndarray
        Shape (N, 3), float32 -- (X, Y, Z) coordinates in metres.
    colors : np.ndarray
        Shape (N, 3), uint8  -- (R, G, B) colours sampled from *color_img*.
    """
    h, w = depth.shape

    # Build pixel-coordinate grids
    # u_grid[r, c] = column index c  (horizontal pixel position)
    # v_grid[r, c] = row    index r  (vertical   pixel position)
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))

    valid = np.isfinite(depth) & (depth > 0)

    Z = depth[valid]
    u = u_grid[valid]
    v = v_grid[valid]

    # Pinhole back-projection
    X = (u - cx) * Z / focal_length_px
    Y = (v - cy) * Z / focal_length_px

    points = np.stack([X, Y, Z], axis=1)   # (N, 3)

    # Sample colours from the left image
    rows = v_grid[valid].astype(int)
    cols = u_grid[valid].astype(int)

    if color_img.ndim == 3:
        # BGR -> RGB for correct PLY colour output
        bgr = color_img[rows, cols]             # (N, 3)
        colors = bgr[:, ::-1].astype(np.uint8)  # flip to RGB
    else:
        gray = color_img[rows, cols].astype(np.uint8)
        colors = np.stack([gray, gray, gray], axis=1)

    # Optional downsampling to keep files manageable
    if max_points is not None and len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a coloured 3-D point cloud to an ASCII PLY file.

    The PLY format is written by hand -- no external library required.
    Each vertex line contains:  x y z red green blue

    Parameters
    ----------
    path : str
        Output file path (will be created; parent directories must exist).
    points : np.ndarray
        Shape (N, 3), float  -- (X, Y, Z) in metres.
    colors : np.ndarray
        Shape (N, 3), uint8  -- (R, G, B) values in [0, 255].

    Notes
    -----
    The resulting file can be opened with:
      • MeshLab   (File -> Import Mesh)
      • CloudCompare (File -> Open)
      • Blender   (File -> Import -> Stanford PLY)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)

    with path.open("w", encoding="ascii") as fh:
        # ---- ASCII PLY header ----
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {n}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")

        # ---- Data rows ----
        # Use numpy to format efficiently rather than a Python for-loop
        # which would be extremely slow for millions of points.
        data = np.hstack([
            points.astype(np.float32),
            colors.astype(np.float32),
        ])
        fmt = ["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"]
        np.savetxt(fh, data, fmt=fmt)
