"""
depth_estimation.py
===================
Convert disparity maps to metric depth maps and colourised visualisations.

Classical CV background -- disparity-to-depth
--------------------------------------------
For an ideal rectified stereo rig the relationship between pixel disparity d
and metric depth Z is derived from similar triangles:

    Z / B = f / d    ⟹    Z = (f x B) / d

where:
    f = focal length in pixels
    B = stereo baseline in metres (horizontal distance between camera centres)
    d = disparity in pixels (how many pixels the same world point shifts
        between the left and right image)

The formula shows:
  • Large disparity  -> close object (small Z).
  • Small disparity  -> far object   (large Z).
  • d = 0 is undefined (point at infinity), which is why we masked it to NaN.

Depth noise is amplified quadratically at large distances because a 1-pixel
disparity error at d ~ 1 px translates to a huge depth error, while the same
error at d = 100 px is negligible.  This is the fundamental limitation of
passive stereo -- it is accurate at close range and degrades at long range.

Depth cleaning
--------------
Two passes are applied after the raw conversion:
  1. Median blur (configurable kernel size) -- removes salt-and-pepper artefacts
     caused by isolated mismatches without blurring edges as strongly as a
     Gaussian (because the median is robust to outliers).
  2. Hard clip at max_depth_m -- discards pixels at implausibly large depths
     which are almost always noise or sky.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
) -> np.ndarray:
    """Convert a disparity map to a metric depth map.

    Uses the standard stereo depth formula: Z = (f x B) / d.

    Parameters
    ----------
    disparity : np.ndarray
        Float32 disparity map (H, W).  NaN marks invalid pixels.
    focal_length_px : float
        Camera focal length in pixels.
    baseline_m : float
        Stereo baseline in metres.

    Returns
    -------
    np.ndarray
        Float32 depth map (H, W) in metres.  NaN where disparity was NaN or <= 0.
    """
    depth = np.full_like(disparity, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    return depth


def clean_depth_map(
    depth: np.ndarray,
    max_depth_m: float = 20.0,
    median_ksize: int = 5,
) -> np.ndarray:
    """Apply median smoothing and far-clip to remove depth artefacts.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W) in metres, NaN for invalid pixels.
    max_depth_m : float
        Pixels with depth > max_depth_m are set to NaN (implausibly far).
    median_ksize : int
        Kernel size for median blur.  Must be odd >= 1.
        A value of 1 effectively disables the blur.
        NOTE: cv2.medianBlur does not handle NaN directly; we temporarily
        replace NaN with 0, blur, then restore the mask.

    Returns
    -------
    np.ndarray
        Cleaned float32 depth map (H, W) in metres.
    """
    if median_ksize % 2 == 0:
        median_ksize += 1
    median_ksize = max(1, median_ksize)

    # Work on a copy, keeping the NaN mask
    invalid_mask = ~np.isfinite(depth)
    cleaned = depth.copy()

    # Replace NaN with 0 so medianBlur can operate on the array
    cleaned[invalid_mask] = 0.0

    if median_ksize > 1:
        cleaned = cv2.medianBlur(cleaned, median_ksize)

    # Restore NaN for originally-invalid pixels and far-clipped pixels
    cleaned[invalid_mask] = np.nan
    cleaned[cleaned > max_depth_m] = np.nan
    cleaned[cleaned <= 0] = np.nan

    return cleaned


def depth_to_visualizable(depth: np.ndarray) -> np.ndarray:
    """Produce a colourised depth image (JET colormap, closer = warmer).

    The JET colourmap encodes depth as hue:
        low depth  (close) -> red/orange (warm)
        high depth (far)   -> blue       (cool)

    Invalid (NaN) pixels are rendered as black.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W) in metres.

    Returns
    -------
    np.ndarray
        BGR uint8 colourised depth image (H, W, 3).
    """
    valid_mask = np.isfinite(depth)
    out_gray = np.zeros(depth.shape, dtype=np.uint8)

    if valid_mask.any():
        d_min = depth[valid_mask].min()
        d_max = depth[valid_mask].max()
        denom = max(d_max - d_min, 1e-6)

        # Invert so that close (small depth) -> high value -> warm colour in JET
        # After normalisation: 0 = far, 255 = close
        normalised = 1.0 - (depth - d_min) / denom   # 0..1, inverted
        normalised = np.clip(normalised, 0.0, 1.0)
        out_gray[valid_mask] = (normalised[valid_mask] * 255).astype(np.uint8)

    # Apply JET colormap (works on uint8 grayscale)
    color_img = cv2.applyColorMap(out_gray, cv2.COLORMAP_JET)

    # Black out invalid pixels
    color_img[~valid_mask] = 0

    return color_img
