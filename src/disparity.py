"""
disparity.py
============
Stereo disparity computation using classical block-matching algorithms.

Classical CV background -- disparity & block matching
-----------------------------------------------------
After rectification, every pixel (u, v) in the left image has its
corresponding match somewhere on the *same row* in the right image, at
column u - d, where d >= 0 is the *disparity*.

Disparity is related to depth Z by:
    d = (focal_length_px x baseline_m) / Z

Block matching finds d for each pixel by sliding a small window (block) of
size (block_size x block_size) along the epipolar line and finding the
horizontal shift that minimises a similarity cost (e.g. Sum of Absolute
Differences, SAD).

Two algorithms are provided:

  StereoBM (Block Matching)
  -------------------------
  Fast, runs left-to-right, computes a 1-D horizontal search.  Produces a
  sparse, noisy disparity map, especially on textureless or occluded regions.

  StereoSGBM (Semi-Global Block Matching, Hirschmüller 2008)
  ----------------------------------------------------------
  Aggregates matching costs along multiple scan-line directions (usually 5 or
  8 paths) using a dynamic-programming smoothness penalty.  This enforces
  spatial consistency, producing a much denser and smoother disparity map at
  the cost of higher CPU usage.

Invalid pixels (disparity <= 0, i.e. unmatched) are set to NaN so downstream
code can detect and mask them rather than treating them as real measurements.
"""

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_disparity(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    method: str = "sgbm",
    num_disparities: int = 96,
    block_size: int = 7,
) -> np.ndarray:
    """Compute a dense disparity map from a rectified stereo pair.

    Parameters
    ----------
    rect_left : np.ndarray
        Rectified left image (HxW or HxWx3, uint8).
    rect_right : np.ndarray
        Rectified right image -- must be the same size and type as *rect_left*.
    method : {'sgbm', 'bm'}
        Stereo matching algorithm.  'sgbm' (default) gives denser, smoother
        results; 'bm' is faster but noisier.
    num_disparities : int
        Search range: the algorithm looks for matches within
        [0, num_disparities] pixels to the left of each pixel.
        Must be a positive multiple of 16.
    block_size : int
        Width/height of the matching window in pixels.
        Must be odd and >= 1.  Larger blocks -> smoother but less detailed.

    Returns
    -------
    disparity : np.ndarray
        Float32 disparity map of shape (H, W).  Invalid pixels (no match
        found) are set to NaN.  Valid pixels have disparity > 0.
    """
    # Ensure num_disparities is a multiple of 16 (OpenCV requirement)
    num_disparities = max(16, (num_disparities // 16) * 16)

    # Ensure block_size is odd and at least 1
    block_size = max(1, block_size)
    if block_size % 2 == 0:
        block_size += 1

    method = method.lower()

    if method == "bm":
        disparity_map = _compute_bm(rect_left, rect_right, num_disparities, block_size)
    elif method == "sgbm":
        disparity_map = _compute_sgbm(rect_left, rect_right, num_disparities, block_size)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'bm' or 'sgbm'.")

    return disparity_map


def disparity_to_visualizable(disparity: np.ndarray) -> np.ndarray:
    """Normalise a float32 disparity map to an 8-bit grayscale image for saving.

    Invalid (NaN) pixels are mapped to black (0).  Valid pixels are linearly
    scaled to [1, 255] so that the image is not entirely black when all values
    are small.

    Parameters
    ----------
    disparity : np.ndarray
        Float32 disparity map, shape (H, W), with NaN for invalid pixels.

    Returns
    -------
    np.ndarray
        uint8 grayscale image, shape (H, W).
    """
    valid_mask = np.isfinite(disparity)
    out = np.zeros_like(disparity, dtype=np.uint8)

    if valid_mask.any():
        d_min = disparity[valid_mask].min()
        d_max = disparity[valid_mask].max()
        denom = max(d_max - d_min, 1e-6)
        normalised = (disparity - d_min) / denom           # 0..1
        out[valid_mask] = (normalised[valid_mask] * 254 + 1).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to grayscale."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _raw_to_float(raw: np.ndarray) -> np.ndarray:
    """Convert OpenCV's fixed-point disparity (x16) to float32 with NaN masking.

    OpenCV StereoMatcher returns disparity multiplied by 16 (fixed-point Q12.4).
    The value -16 (== -1 x 16) indicates an invalid/unmatched pixel.
    """
    disp_float = raw.astype(np.float32) / 16.0
    disp_float[disp_float <= 0] = np.nan
    return disp_float


def _compute_bm(
    left: np.ndarray,
    right: np.ndarray,
    num_disparities: int,
    block_size: int,
) -> np.ndarray:
    """Run cv2.StereoBM on grayscale images."""
    gray_L = _to_gray(left)
    gray_R = _to_gray(right)

    matcher = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size,
    )
    raw = matcher.compute(gray_L, gray_R)
    return _raw_to_float(raw)


def _compute_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    num_disparities: int,
    block_size: int,
) -> np.ndarray:
    """Run cv2.StereoSGBM with 5-direction path aggregation.

    P1 / P2 are smoothness penalty parameters.  Higher P2 enforces stronger
    smoothness; the recommended heuristic is P1 = 8*ch*bs² and P2 = 32*ch*bs².
    """
    # SGBM can work on colour images (use 3 channels)
    num_channels = 1 if left.ndim == 2 else left.shape[2]
    bs2 = block_size * block_size

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8  * num_channels * bs2,
        P2=32 * num_channels * bs2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(left, right)
    return _raw_to_float(raw)
