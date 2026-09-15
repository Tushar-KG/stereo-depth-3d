"""
calibration.py
==============
Camera calibration utilities for the stereo depth pipeline.

Two modes are supported:
  1. Load pre-computed intrinsics from a JSON file (most common for this project).
  2. Run full OpenCV checkerboard calibration from a glob of images (for real cameras).

Classical CV background
-----------------------
A pinhole camera maps a 3-D point P = (X, Y, Z) to image pixel (u, v) via:

    u = fx * (X/Z) + cx
    v = fy * (Y/Z) + cy

where fx, fy are the focal lengths in pixels and (cx, cy) is the principal point
(where the optical axis pierces the image plane).

For a stereo rig the additional parameter is the *baseline* B -- the horizontal
distance (in metres) between the two camera optical centres.  Combined, these
four values are enough to convert a measured pixel disparity d into metric depth Z:

    Z = (fx * B) / d          (see depth_estimation.py for full derivation)
"""

import json
import glob
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_calibration_json(path: str) -> dict:
    """Load camera intrinsics from a JSON file.

    Expected JSON structure
    -----------------------
    {
        "focal_length_px": 700.0,   // focal length in pixels (fx = fy assumed)
        "baseline_m":       0.1,    // stereo baseline in metres
        "cx":             319.5,    // principal point x  (~ image_width  / 2)
        "cy":             239.5     // principal point y  (~ image_height / 2)
    }

    Parameters
    ----------
    path : str
        Filesystem path to the JSON calibration file.

    Returns
    -------
    dict
        Keys: focal_length_px (float), baseline_m (float), cx (float), cy (float).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If any required key is missing from the JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with path.open("r") as fh:
        data = json.load(fh)

    required_keys = {"focal_length_px", "baseline_m", "cx", "cy"}
    missing = required_keys - data.keys()
    if missing:
        raise KeyError(f"Calibration JSON is missing keys: {missing}")

    return {
        "focal_length_px": float(data["focal_length_px"]),
        "baseline_m":      float(data["baseline_m"]),
        "cx":              float(data["cx"]),
        "cy":              float(data["cy"]),
    }


def calibrate_from_checkerboard(
    image_glob: str,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> dict:
    """Calibrate a single camera from a glob of checkerboard images.

    Uses the standard OpenCV pipeline:
      1. cv2.findChessboardCorners  -- locates inner corners in each image.
      2. cv2.cornerSubPix           -- refines corners to sub-pixel accuracy.
      3. cv2.calibrateCamera        -- solves for intrinsic matrix K and
                                      distortion coefficients D using
                                      Zhang's method (homography-based).

    Parameters
    ----------
    image_glob : str
        Glob pattern matching the calibration images, e.g. 'data/calib/*.jpg'.
    pattern_size : tuple of (int, int)
        Number of *inner* corners per row and column of the checkerboard.
        A standard 9x6 board has 10 squares per row, 7 per column.
    square_size_m : float
        Physical size of one square in metres.  Only affects the scale of
        the returned translation vectors; the intrinsics are unaffected.

    Returns
    -------
    dict
        focal_length_px : float   (mean of fx, fy from K)
        cx, cy          : float   (principal point from K)
        camera_matrix   : ndarray (3x3)
        dist_coeffs     : ndarray (1x5)
        rms_error       : float   (reprojection RMS in pixels)

    Raises
    ------
    RuntimeError
        If fewer than 3 usable images are found.
    """
    image_paths = sorted(glob.glob(image_glob))
    if len(image_paths) < 3:
        raise RuntimeError(
            f"Need at least 3 checkerboard images; found {len(image_paths)} matching '{image_glob}'"
        )

    # World-space coordinates of the checkerboard corners (Z = 0 for flat board)
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    object_points = []   # 3-D points in world space
    image_points  = []   # 2-D points in image space
    img_shape     = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]   # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not ret:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(corners_refined)

    if len(object_points) < 3:
        raise RuntimeError(
            "Could not detect the checkerboard pattern in enough images.  "
            "Make sure pattern_size matches the number of *inner* corners."
        )

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None
    )

    fx, fy = K[0, 0], K[1, 1]
    return {
        "focal_length_px": float((fx + fy) / 2),
        "cx":              float(K[0, 2]),
        "cy":              float(K[1, 2]),
        "camera_matrix":   K,
        "dist_coeffs":     D,
        "rms_error":       float(rms),
    }
