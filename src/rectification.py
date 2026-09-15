"""
rectification.py
================
Stereo image rectification utilities.

Classical CV background -- epipolar geometry & rectification
-----------------------------------------------------------
Two cameras looking at the same scene share an *epipolar constraint*: the
projection of a 3-D point P into the left image, the optical centre of the
left camera O_L, and O_R all lie in the same plane (the *epipolar plane*).
Its intersection with the right image is a line called the *epipolar line*.

For a horizontally-aligned stereo rig the epipolar lines are horizontal, so
matching a pixel in the left image only requires a 1-D horizontal search in
the right image -- this is the key simplification exploited by block matching.

If the cameras are not perfectly aligned (they almost never are in practice),
*rectification* applies a projective warp to both images such that:
  • All epipolar lines become horizontal (same row).
  • The rectified images have the same vertical field of view.

OpenCV implements this via:
  1. cv2.stereoRectify     -- computes rectification rotation matrices R1, R2
                             and new projection matrices P1, P2.
  2. cv2.initUndistortRectifyMap -- builds pixel-remapping look-up tables.
  3. cv2.remap             -- applies the warp to each image.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rectify_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    stereo_calib: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectify a stereo image pair.

    If *stereo_calib* is None the function assumes the images are already
    rectified (rows are already aligned epipolar lines) and returns them
    unchanged.  This is the expected mode when using the synthetic test pair
    or images from a factory-rectified stereo camera.

    If *stereo_calib* is provided, full stereo rectification is performed
    using the OpenCV pipeline described in the module docstring.

    Parameters
    ----------
    left_img : np.ndarray
        Left camera image (HxW or HxWx3, uint8).
    right_img : np.ndarray
        Right camera image -- must be the same size as *left_img*.
    stereo_calib : dict or None
        If None: no-op, assume pre-rectified images.
        If dict, must contain:
            K_left   (3x3) -- left camera intrinsic matrix
            K_right  (3x3) -- right camera intrinsic matrix
            D_left   (1x5) -- left distortion coefficients
            D_right  (1x5) -- right distortion coefficients
            R        (3x3) -- rotation of right camera relative to left
            T        (3x1) -- translation of right camera relative to left (metres)

    Returns
    -------
    (rect_left, rect_right) : tuple of np.ndarray
        Rectified left and right images, same dtype as input.
    """
    if stereo_calib is None:
        # Assume pre-rectified; skip warping to avoid introducing any artefacts.
        return left_img.copy(), right_img.copy()

    h, w = left_img.shape[:2]
    image_size = (w, h)

    K_L = np.asarray(stereo_calib["K_left"],  dtype=np.float64)
    K_R = np.asarray(stereo_calib["K_right"], dtype=np.float64)
    D_L = np.asarray(stereo_calib["D_left"],  dtype=np.float64)
    D_R = np.asarray(stereo_calib["D_right"], dtype=np.float64)
    R   = np.asarray(stereo_calib["R"],        dtype=np.float64)
    T   = np.asarray(stereo_calib["T"],        dtype=np.float64)

    # Compute rectification transforms:
    #   R1, R2 : rotation matrices that bring each camera into the rectified frame
    #   P1, P2 : new projection matrices (3x4) in the rectified coordinate system
    #   Q      : disparity-to-depth mapping matrix (used by cv2.reprojectImageTo3D)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_L, D_L, K_R, D_R,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,    # crop to only valid pixels (no black borders)
    )

    # Build pixel-remapping look-up tables for each camera
    map1_L, map2_L = cv2.initUndistortRectifyMap(
        K_L, D_L, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_R, map2_R = cv2.initUndistortRectifyMap(
        K_R, D_R, R2, P2, image_size, cv2.CV_16SC2
    )

    # Apply the warp
    rect_left  = cv2.remap(left_img,  map1_L, map2_L, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map1_R, map2_R, cv2.INTER_LINEAR)

    return rect_left, rect_right
