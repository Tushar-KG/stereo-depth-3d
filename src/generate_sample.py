"""
generate_sample.py
==================
Synthetic stereo pair generator for end-to-end pipeline testing.

Why synthetic data?
-------------------
Real stereo cameras require physical hardware and lab-quality calibration.
For coursework testing, we render a simple scene that respects the same
geometric constraints as real stereo imaging:

  1. Several flat rectangles are placed at known depths in front of a
     "virtual camera".
  2. The LEFT image is rendered from the origin (0, 0, Z).
  3. The RIGHT image is rendered from a camera displaced B metres to the
     right: (B, 0, Z).
  4. A rectangle at depth Z casts a different column position in the two
     images: its horizontal shift (disparity) equals (f * B) / Z pixels,
     exactly matching the theoretical formula.

Texture requirement
-------------------
Block-matching algorithms compute similarity by comparing small image
patches.  If the patch contains a uniform colour, *every* shift produces the
same cost, making the optimal disparity ambiguous.  We therefore fill each
rectangle with speckled random noise -- this provides the local contrast
needed for reliable matching.

Output
------
Writes to a chosen directory:
  left.png   -- left-eye view
  right.png  -- right-eye view (rectangles shifted left by the parallax amount)
  calib.json -- known-correct calibration for the synthetic rig
"""

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

# Each rectangle: (depth_m, color_bgr_mean, left_frac, top_frac, width_frac, height_frac)
# Positions are specified as fractions of the image dimensions for resolution-independence.
_SCENE_RECTS = [
    # depth_m   hue_tint (BGR)          x0     y0    w      h
    (1.2,       (100,  60,  30),        0.05,  0.15, 0.25,  0.60),   # left foreground
    (1.8,       ( 30,  90,  40),        0.35,  0.20, 0.20,  0.55),   # left-centre
    (2.5,       ( 60,  30, 110),        0.58,  0.10, 0.18,  0.70),   # right-centre
    (4.0,       ( 20,  80, 140),        0.78,  0.25, 0.18,  0.50),   # far right
    (6.0,       ( 80,  50,  20),        0.15,  0.65, 0.65,  0.30),   # background strip
]

_BACKGROUND_NOISE_STD = 25   # noise amplitude for the background (grey)
_RECT_NOISE_STD       = 40   # noise amplitude inside each rectangle


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sample_pair(
    output_dir: str,
    image_width:  int   = 640,
    image_height: int   = 480,
    focal_length_px: float = 700.0,
    baseline_m:      float = 0.1,
) -> dict:
    """Render a synthetic left+right stereo pair and write a calib.json.

    The scene consists of several textured rectangles at different known depths.
    Each rectangle is shifted horizontally between the two views by exactly
    (focal_length_px x baseline_m) / depth pixels, faithfully simulating
    real stereo parallax.

    Parameters
    ----------
    output_dir : str
        Directory to write left.png, right.png, and calib.json.
    image_width : int
        Rendered image width in pixels.
    image_height : int
        Rendered image height in pixels.
    focal_length_px : float
        Simulated camera focal length in pixels.
    baseline_m : float
        Simulated stereo baseline in metres.

    Returns
    -------
    dict
        Calibration parameters (same structure as calib.json):
        focal_length_px, baseline_m, cx, cy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)   # reproducible output

    cx = image_width  / 2.0
    cy = image_height / 2.0

    # Start with a noisy grey background (gives BM something to match on)
    left_img  = _make_background(image_height, image_width, rng)
    right_img = _make_background(image_height, image_width, rng)

    # Paint rectangles from far to near (painter's algorithm -- no Z-buffering)
    for depth_m, color_bgr, xf, yf, wf, hf in sorted(_SCENE_RECTS, key=lambda r: -r[0]):
        parallax_px = (focal_length_px * baseline_m) / depth_m  # disparity

        x0 = int(xf * image_width)
        y0 = int(yf * image_height)
        x1 = int((xf + wf) * image_width)
        y1 = int((yf + hf) * image_height)

        # Clamp to image boundaries
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, image_width - 1), min(y1, image_height - 1)

        # Generate a single speckled texture and share it between both views
        # (same texture, different horizontal position -- authentic parallax)
        texture = _make_speckle_texture(y1 - y0, x1 - x0, color_bgr, rng)

        # LEFT image: paint at original position
        left_img[y0:y1, x0:x1] = texture

        # RIGHT image: shift left by parallax_px (object appears to move left
        # in the right view for a positive depth)
        shift = int(round(parallax_px))
        rx0 = max(x0 - shift, 0)
        rx1 = max(x1 - shift, 0)
        rx1 = min(rx1, image_width)

        # Crop texture horizontally if the shift pushes part of it off-screen
        tex_col_start = max(0, rx0 - (x0 - shift))
        tex_w = rx1 - rx0
        if tex_w > 0 and tex_col_start < texture.shape[1]:
            right_img[y0:y1, rx0:rx1] = texture[:, tex_col_start:tex_col_start + tex_w]

    # Save images
    cv2.imwrite(str(output_dir / "left.png"),  left_img)
    cv2.imwrite(str(output_dir / "right.png"), right_img)

    # Write calibration JSON
    calib = {
        "focal_length_px": focal_length_px,
        "baseline_m":      baseline_m,
        "cx":              cx,
        "cy":              cy,
    }
    with (output_dir / "calib.json").open("w") as fh:
        json.dump(calib, fh, indent=2)

    print(f"[generate_sample] Written to: {output_dir}")
    print(f"  focal_length_px = {focal_length_px}")
    print(f"  baseline_m      = {baseline_m}")
    print(f"  cx, cy          = {cx}, {cy}")
    for depth_m, _, xf, yf, wf, hf in _SCENE_RECTS:
        disp = (focal_length_px * baseline_m) / depth_m
        print(f"  depth {depth_m:4.1f} m -> expected disparity ~ {disp:.1f} px")

    return calib


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Create a grey background with Gaussian noise (gives BM texture to match)."""
    base = np.full((h, w, 3), 128, dtype=np.float32)
    noise = rng.normal(0, _BACKGROUND_NOISE_STD, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _make_speckle_texture(
    h: int,
    w: int,
    color_bgr: tuple,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a speckled texture with a colour tint.

    The texture is a sum of:
      • The mean colour (constant, provides the hue identity of the rectangle)
      • Gaussian noise (provides local contrast for block matching)

    Parameters
    ----------
    h, w : int
        Height and width of the texture patch.
    color_bgr : tuple
        Mean (B, G, R) colour for the rectangle.
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    np.ndarray
        uint8 BGR texture of shape (h, w, 3).
    """
    base  = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    base  = np.broadcast_to(base, (h, w, 3)).copy()
    noise = rng.normal(0, _RECT_NOISE_STD, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)
