"""
main.py
=======
Command-line entrypoint for the Stereo Depth Estimation + 3D Reconstruction pipeline.

Usage examples
--------------
# Quick demo with a synthetic scene (no real camera needed):
    python src/main.py --generate-sample

# Run on your own images with a calibration file:
    python src/main.py --left data/left.png --right data/right.png --calib data/calib.json

# Use StereoBM instead of SGBM:
    python src/main.py --generate-sample --method bm

# Custom output directory and disparity range:
    python src/main.py --generate-sample --output my_results --num-disparities 128 --block-size 9

Pipeline stages
---------------
  1. (Optional) Generate a synthetic stereo pair.
  2. Load stereo images and calibration parameters.
  3. Rectify the image pair (no-op if no full stereo calibration provided).
  4. Compute the disparity map (BM or SGBM).
  5. Convert disparity -> metric depth.
  6. Clean the depth map (median blur + far clip).
  7. Back-project depth -> 3-D point cloud.
  8. Save all outputs to the chosen output directory.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Make src importable when running as "python src/main.py"
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration      import load_calibration_json
from rectification    import rectify_pair
from disparity        import compute_disparity, disparity_to_visualizable
from depth_estimation import disparity_to_depth, clean_depth_map, depth_to_visualizable
from reconstruction   import depth_to_pointcloud, save_ply
from generate_sample  import generate_sample_pair


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stereo-depth",
        description=(
            "Stereo Depth Estimation + 3D Reconstruction (Classical CV - no deep learning)\n"
            "\n"
            "Example (synthetic demo, no camera needed):\n"
            "  python src/main.py --generate-sample\n"
            "\n"
            "Example (real photos):\n"
            "  python src/main.py --left left.png --right right.png --calib calib.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Input ----
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a synthetic stereo pair for testing and run the full pipeline on it.",
    )
    inp.add_argument("--left",  type=str, default=None, help="Path to the left image.")
    inp.add_argument("--right", type=str, default=None, help="Path to the right image.")
    inp.add_argument(
        "--calib", type=str, default=None,
        help="Path to calib.json (focal_length_px, baseline_m, cx, cy).",
    )

    # ---- Algorithm ----
    alg = parser.add_argument_group("Algorithm")
    alg.add_argument(
        "--method", choices=["bm", "sgbm"], default="sgbm",
        help="Stereo matching algorithm: 'bm' (fast) or 'sgbm' (denser). Default: sgbm.",
    )
    alg.add_argument(
        "--num-disparities", type=int, default=96,
        help="Disparity search range (multiple of 16). Default: 96.",
    )
    alg.add_argument(
        "--block-size", type=int, default=7,
        help="Matching window size (odd integer >= 1). Default: 7.",
    )
    alg.add_argument(
        "--max-depth", type=float, default=20.0,
        help="Clip depth values beyond this distance (metres). Default: 20.0.",
    )

    # ---- Output ----
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output", type=str, default="output",
        help="Root directory for all outputs. Default: 'output'.",
    )

    return parser


# ---------------------------------------------------------------------------
# Pipeline steps (each prints a progress banner)
# ---------------------------------------------------------------------------

def _banner(step: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {step}")
    print(f"{'-' * 60}")


def _load_image(path: str, name: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot read {name} image: {path}", file=sys.stderr)
        sys.exit(1)
    return img


def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()

    output_root = Path(args.output)
    disp_dir    = output_root / "disparity_maps"
    cloud_dir   = output_root / "point_clouds"
    sample_dir  = Path("data") / "sample_stereo_pairs"

    disp_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 - Generate or load images
    # ------------------------------------------------------------------
    _banner("Step 1 / 6 - Loading images")

    if args.generate_sample:
        print("[info] Generating synthetic stereo pair …")
        calib = generate_sample_pair(str(sample_dir))
        left_path  = str(sample_dir / "left.png")
        right_path = str(sample_dir / "right.png")
        calib_path = str(sample_dir / "calib.json")
    else:
        if not args.left or not args.right:
            print(
                "[ERROR] Provide --left and --right, or use --generate-sample.",
                file=sys.stderr,
            )
            sys.exit(1)
        left_path  = args.left
        right_path = args.right
        calib_path = args.calib

    left_img  = _load_image(left_path,  "left")
    right_img = _load_image(right_path, "right")
    print(f"  Left  : {left_path}  ({left_img.shape[1]}x{left_img.shape[0]})")
    print(f"  Right : {right_path}  ({right_img.shape[1]}x{right_img.shape[0]})")

    # ------------------------------------------------------------------
    # Step 2 - Load calibration
    # ------------------------------------------------------------------
    _banner("Step 2 / 6 - Loading calibration")
    if calib_path:
        calib = load_calibration_json(calib_path)
        print(f"  Loaded: {calib_path}")
    else:
        # Fallback: estimate focal length from image width (rough heuristic)
        h, w = left_img.shape[:2]
        calib = {
            "focal_length_px": float(w),      # common rule of thumb: f ~ image_width
            "baseline_m":      0.1,
            "cx":              w / 2.0,
            "cy":              h / 2.0,
        }
        print("  [warn] No calib.json provided; using rough default intrinsics.")

    print(f"  focal_length_px = {calib['focal_length_px']:.1f} px")
    print(f"  baseline_m      = {calib['baseline_m']:.4f} m")
    print(f"  cx, cy          = {calib['cx']:.1f}, {calib['cy']:.1f}")

    # ------------------------------------------------------------------
    # Step 3 - Rectification
    # ------------------------------------------------------------------
    _banner("Step 3 / 6 - Rectification")
    rect_left, rect_right = rectify_pair(left_img, right_img, stereo_calib=None)
    print("  No full stereo calibration provided -> assuming pre-rectified images (no-op).")

    # ------------------------------------------------------------------
    # Step 4 - Disparity
    # ------------------------------------------------------------------
    _banner("Step 4 / 6 - Disparity computation")
    print(f"  Method        : {args.method.upper()}")
    print(f"  num_disparities: {args.num_disparities}")
    print(f"  block_size    : {args.block_size}")

    t0 = time.perf_counter()
    disparity = compute_disparity(
        rect_left, rect_right,
        method=args.method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
    )
    print(f"  Done in {time.perf_counter() - t0:.2f} s")

    valid_pct = 100 * np.isfinite(disparity).sum() / disparity.size
    print(f"  Valid pixels  : {valid_pct:.1f}%")
    if np.any(np.isfinite(disparity)):
        vd = disparity[np.isfinite(disparity)]
        print(f"  Disparity range: [{vd.min():.1f}, {vd.max():.1f}] px")

    disp_vis = disparity_to_visualizable(disparity)
    disp_out = disp_dir / "disparity.png"
    cv2.imwrite(str(disp_out), disp_vis)
    print(f"  Saved -> {disp_out}")

    # ------------------------------------------------------------------
    # Step 5 - Depth
    # ------------------------------------------------------------------
    _banner("Step 5 / 6 - Depth estimation")
    depth_raw = disparity_to_depth(
        disparity,
        focal_length_px=calib["focal_length_px"],
        baseline_m=calib["baseline_m"],
    )
    depth = clean_depth_map(depth_raw, max_depth_m=args.max_depth, median_ksize=5)

    valid_depth_pct = 100 * np.isfinite(depth).sum() / depth.size
    print(f"  Valid depth pixels: {valid_depth_pct:.1f}%")
    if np.any(np.isfinite(depth)):
        vdp = depth[np.isfinite(depth)]
        print(f"  Depth range: [{vdp.min():.2f}, {vdp.max():.2f}] m")

    depth_color = depth_to_visualizable(depth)
    depth_out   = disp_dir / "depth_colormap.png"
    cv2.imwrite(str(depth_out), depth_color)
    print(f"  Saved -> {depth_out}")

    # ------------------------------------------------------------------
    # Step 6 - 3-D Reconstruction
    # ------------------------------------------------------------------
    _banner("Step 6 / 6 - 3D point cloud reconstruction")
    points, colors = depth_to_pointcloud(
        depth, rect_left,
        focal_length_px=calib["focal_length_px"],
        cx=calib["cx"],
        cy=calib["cy"],
        max_points=500_000,
    )
    print(f"  Points generated: {len(points):,}")

    ply_out = cloud_dir / "reconstruction.ply"
    save_ply(str(ply_out), points, colors)
    print(f"  Saved -> {ply_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {time.perf_counter() - t_start:.2f} s")
    print(f"  Output directory : {output_root.resolve()}")
    print(f"{'=' * 60}")
    print(f"\n  Outputs:")
    print(f"    Disparity map  : {disp_out}")
    print(f"    Depth colormap : {depth_out}")
    print(f"    Point cloud    : {ply_out}")
    print(f"\n  View the .ply file in MeshLab, CloudCompare, or Blender.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if not args.generate_sample and (args.left is None or args.right is None):
        parser.print_help()
        print(
            "\n[ERROR] Either --generate-sample or both --left and --right are required.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_pipeline(args)
