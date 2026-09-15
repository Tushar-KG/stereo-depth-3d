"""
tests/test_pipeline.py
======================
End-to-end pytest suite for the Stereo Depth Estimation pipeline.

All tests use tmp_path (a pytest fixture that provides a fresh temporary
directory per test), so no hardcoded paths are used and tests are fully
isolated.

Test coverage:
  1. generate_sample          -- files exist, images are readable, calib is valid
  2. compute_disparity        -- output shape, dtype, NaN masking
  3. disparity_to_depth       -- positive values, NaN propagation
  4. depth_to_pointcloud      -- output shapes, no negative Z
  5. save_ply + round-trip    -- valid ASCII PLY format, parseable back
  6. full pipeline            -- run end-to-end, check output files exist
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make src/ importable when pytest is run from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC          = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generate_sample  import generate_sample_pair
from calibration      import load_calibration_json
from rectification    import rectify_pair
from disparity        import compute_disparity, disparity_to_visualizable
from depth_estimation import disparity_to_depth, clean_depth_map, depth_to_visualizable
from reconstruction   import depth_to_pointcloud, save_ply


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    """Generate the synthetic stereo pair once; reuse across the whole module."""
    d = tmp_path_factory.mktemp("sample")
    generate_sample_pair(str(d))
    return d


@pytest.fixture(scope="module")
def sample_images(sample_dir):
    """Load the generated left/right images."""
    left  = cv2.imread(str(sample_dir / "left.png"))
    right = cv2.imread(str(sample_dir / "right.png"))
    return left, right


@pytest.fixture(scope="module")
def sample_calib(sample_dir):
    """Load the generated calibration dict."""
    return load_calibration_json(str(sample_dir / "calib.json"))


@pytest.fixture(scope="module")
def disparity_map(sample_images):
    """Compute SGBM disparity once; reuse in depth and cloud tests."""
    left, right = sample_images
    return compute_disparity(left, right, method="sgbm",
                             num_disparities=96, block_size=7)


@pytest.fixture(scope="module")
def depth_map(disparity_map, sample_calib):
    """Convert disparity to cleaned depth once."""
    calib = sample_calib
    raw   = disparity_to_depth(disparity_map,
                                focal_length_px=calib["focal_length_px"],
                                baseline_m=calib["baseline_m"])
    return clean_depth_map(raw, max_depth_m=20.0, median_ksize=5)


# ===========================================================================
# 1 -- Sample generation
# ===========================================================================

class TestGenerateSample:

    def test_files_exist(self, sample_dir):
        """All expected files should be present after generation."""
        assert (sample_dir / "left.png").exists(),  "left.png missing"
        assert (sample_dir / "right.png").exists(), "right.png missing"
        assert (sample_dir / "calib.json").exists(), "calib.json missing"

    def test_images_readable(self, sample_images):
        """Images must load as non-empty uint8 BGR arrays."""
        left, right = sample_images
        assert left  is not None, "left image failed to load"
        assert right is not None, "right image failed to load"
        assert left.dtype  == np.uint8
        assert right.dtype == np.uint8
        assert left.ndim   == 3 and left.shape[2]  == 3
        assert right.ndim  == 3 and right.shape[2] == 3

    def test_images_same_size(self, sample_images):
        """Both images must be the same resolution."""
        left, right = sample_images
        assert left.shape == right.shape, (
            f"Shape mismatch: left={left.shape}, right={right.shape}"
        )

    def test_calib_keys(self, sample_calib):
        """Calibration dict must contain all required keys with positive values."""
        calib = sample_calib
        for key in ("focal_length_px", "baseline_m", "cx", "cy"):
            assert key in calib, f"Missing key '{key}' in calib"
            assert calib[key] > 0, f"Expected positive value for '{key}', got {calib[key]}"

    def test_calib_json_roundtrip(self, sample_dir):
        """JSON can be re-loaded and values survive a round-trip."""
        with (sample_dir / "calib.json").open() as f:
            data = json.load(f)
        calib2 = load_calibration_json(str(sample_dir / "calib.json"))
        assert abs(data["focal_length_px"] - calib2["focal_length_px"]) < 1e-6


# ===========================================================================
# 2 -- Disparity computation
# ===========================================================================

class TestDisparity:

    def test_output_shape_matches_input(self, sample_images, disparity_map):
        """Disparity map must have the same (H, W) as the input images."""
        left, _ = sample_images
        h, w    = left.shape[:2]
        assert disparity_map.shape == (h, w), (
            f"Expected ({h}, {w}), got {disparity_map.shape}"
        )

    def test_output_dtype_float32(self, disparity_map):
        """Disparity map must be float32."""
        assert disparity_map.dtype == np.float32

    def test_has_valid_pixels(self, disparity_map):
        """At least some pixels should have valid (non-NaN) disparity."""
        assert np.any(np.isfinite(disparity_map)), "All disparity pixels are NaN"

    def test_invalid_pixels_are_nan(self, disparity_map):
        """Invalid pixels must be NaN, never <= 0."""
        invalid = ~np.isfinite(disparity_map)
        # All non-NaN values must be > 0
        valid_vals = disparity_map[~invalid]
        assert np.all(valid_vals > 0), "Some valid disparity values are <= 0"

    def test_bm_output_shape(self, sample_images):
        """StereoBM should also produce the correct output shape."""
        left, right = sample_images
        disp_bm = compute_disparity(left, right, method="bm",
                                    num_disparities=64, block_size=15)
        h, w = left.shape[:2]
        assert disp_bm.shape == (h, w)
        assert disp_bm.dtype == np.float32

    def test_visualizable_is_uint8(self, disparity_map):
        """disparity_to_visualizable must return a uint8 image."""
        vis = disparity_to_visualizable(disparity_map)
        assert vis.dtype == np.uint8
        assert vis.shape == disparity_map.shape

    def test_unknown_method_raises(self, sample_images):
        """Unknown method name should raise ValueError."""
        left, right = sample_images
        with pytest.raises(ValueError, match="Unknown method"):
            compute_disparity(left, right, method="magic_algorithm")


# ===========================================================================
# 3 -- Depth estimation
# ===========================================================================

class TestDepthEstimation:

    def test_depth_values_positive(self, depth_map):
        """All valid (non-NaN) depth values must be strictly positive."""
        valid = depth_map[np.isfinite(depth_map)]
        assert len(valid) > 0, "No valid depth pixels at all"
        assert np.all(valid > 0), "Some depth values are <= 0"

    def test_depth_shape_matches_disparity(self, disparity_map, depth_map):
        """Depth map must match disparity map shape."""
        assert depth_map.shape == disparity_map.shape

    def test_depth_nan_where_disparity_nan(self, disparity_map, sample_calib):
        """Wherever disparity is NaN, depth should also be NaN."""
        calib = sample_calib
        raw   = disparity_to_depth(disparity_map,
                                    focal_length_px=calib["focal_length_px"],
                                    baseline_m=calib["baseline_m"])
        disp_nan  = ~np.isfinite(disparity_map)
        depth_nan = ~np.isfinite(raw)
        # All NaN disparity pixels must produce NaN depth
        assert np.all(depth_nan[disp_nan]), "NaN disparity did not produce NaN depth"

    def test_depth_clipped_at_max(self, disparity_map, sample_calib):
        """clean_depth_map must not return any value > max_depth_m."""
        calib    = sample_calib
        raw      = disparity_to_depth(disparity_map,
                                       focal_length_px=calib["focal_length_px"],
                                       baseline_m=calib["baseline_m"])
        max_d    = 5.0
        cleaned  = clean_depth_map(raw, max_depth_m=max_d, median_ksize=1)
        valid    = cleaned[np.isfinite(cleaned)]
        assert np.all(valid <= max_d + 1e-3), "Depth exceeds max_depth_m after clipping"

    def test_depth_colormap_shape_and_dtype(self, depth_map):
        """depth_to_visualizable must return a BGR uint8 image."""
        color = depth_to_visualizable(depth_map)
        assert color.dtype == np.uint8
        assert color.ndim  == 3
        assert color.shape[2] == 3


# ===========================================================================
# 4 -- Point cloud generation
# ===========================================================================

class TestPointCloud:

    def test_pointcloud_shapes(self, depth_map, sample_images, sample_calib):
        """Points and colors must be (N, 3) with matching N."""
        left, _ = sample_images
        calib   = sample_calib
        pts, col = depth_to_pointcloud(
            depth_map, left,
            focal_length_px=calib["focal_length_px"],
            cx=calib["cx"], cy=calib["cy"],
        )
        assert pts.shape[1] == 3, "Points should have 3 columns (X, Y, Z)"
        assert col.shape[1] == 3, "Colors should have 3 columns (R, G, B)"
        assert pts.shape[0] == col.shape[0], "Points and colors must have same N"

    def test_pointcloud_positive_z(self, depth_map, sample_images, sample_calib):
        """All Z coordinates in the point cloud must be > 0 (in front of camera)."""
        left, _ = sample_images
        calib   = sample_calib
        pts, _  = depth_to_pointcloud(
            depth_map, left,
            focal_length_px=calib["focal_length_px"],
            cx=calib["cx"], cy=calib["cy"],
        )
        assert np.all(pts[:, 2] > 0), "Some Z values are non-positive"

    def test_max_points_respected(self, depth_map, sample_images, sample_calib):
        """max_points should cap the returned cloud size."""
        left, _ = sample_images
        calib   = sample_calib
        limit   = 1000
        pts, col = depth_to_pointcloud(
            depth_map, left,
            focal_length_px=calib["focal_length_px"],
            cx=calib["cx"], cy=calib["cy"],
            max_points=limit,
        )
        assert pts.shape[0] <= limit, f"Expected <= {limit} points, got {pts.shape[0]}"

    def test_colors_in_valid_range(self, depth_map, sample_images, sample_calib):
        """Color values must be in [0, 255] uint8."""
        left, _ = sample_images
        calib   = sample_calib
        _, col  = depth_to_pointcloud(
            depth_map, left,
            focal_length_px=calib["focal_length_px"],
            cx=calib["cx"], cy=calib["cy"],
        )
        assert col.dtype  == np.uint8
        assert col.min()  >= 0
        assert col.max()  <= 255


# ===========================================================================
# 5 -- PLY export & format validation
# ===========================================================================

class TestPlyExport:

    def _make_small_cloud(self):
        """Tiny 4-point cloud for format-checking tests."""
        pts = np.array([[0.1, 0.2, 1.0],
                        [0.5, 0.0, 2.0],
                        [-0.3, 0.4, 1.5],
                        [0.0, 0.0, 3.0]], dtype=np.float32)
        col = np.array([[255,   0,   0],
                        [  0, 255,   0],
                        [  0,   0, 255],
                        [128, 128, 128]], dtype=np.uint8)
        return pts, col

    def test_ply_file_created(self, tmp_path):
        """save_ply must create the file."""
        pts, col = self._make_small_cloud()
        out_path = tmp_path / "test.ply"
        save_ply(str(out_path), pts, col)
        assert out_path.exists(), "PLY file was not created"
        assert out_path.stat().st_size > 0, "PLY file is empty"

    def test_ply_header_format(self, tmp_path):
        """PLY header must match the expected ASCII format."""
        pts, col = self._make_small_cloud()
        out_path = tmp_path / "test.ply"
        save_ply(str(out_path), pts, col)

        lines = out_path.read_text(encoding="ascii").splitlines()

        assert lines[0] == "ply",              "First line must be 'ply'"
        assert lines[1] == "format ascii 1.0","Second line must be 'format ascii 1.0'"
        assert "element vertex 4" in lines,    "Header must declare correct vertex count"
        assert "property float x" in lines,   "Header missing 'property float x'"
        assert "property uchar red" in lines,  "Header missing 'property uchar red'"
        assert "end_header" in lines,          "Header must end with 'end_header'"

    def test_ply_vertex_count(self, tmp_path):
        """PLY data rows must equal declared vertex count."""
        pts, col = self._make_small_cloud()
        out_path = tmp_path / "test.ply"
        save_ply(str(out_path), pts, col)

        content  = out_path.read_text(encoding="ascii")
        header_end = content.index("end_header\n") + len("end_header\n")
        data_lines = [l for l in content[header_end:].splitlines() if l.strip()]

        assert len(data_lines) == 4, f"Expected 4 data lines, got {len(data_lines)}"

    def test_ply_coordinate_values(self, tmp_path):
        """Parsed X,Y,Z values must match what was saved (within float precision)."""
        pts, col = self._make_small_cloud()
        out_path = tmp_path / "test.ply"
        save_ply(str(out_path), pts, col)

        content    = out_path.read_text(encoding="ascii")
        header_end = content.index("end_header\n") + len("end_header\n")
        data_lines = content[header_end:].strip().splitlines()

        for i, line in enumerate(data_lines):
            vals = line.split()
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            assert abs(x - pts[i, 0]) < 1e-4, f"Row {i}: X mismatch"
            assert abs(y - pts[i, 1]) < 1e-4, f"Row {i}: Y mismatch"
            assert abs(z - pts[i, 2]) < 1e-4, f"Row {i}: Z mismatch"


# ===========================================================================
# 6 -- Full end-to-end pipeline
# ===========================================================================

class TestEndToEnd:

    def test_full_pipeline_outputs_exist(self, tmp_path):
        """Running the full pipeline must produce disparity.png, depth_colormap.png, and .ply."""
        # Generate synthetic pair in tmp_path
        sample = tmp_path / "sample"
        generate_sample_pair(str(sample))

        left  = cv2.imread(str(sample / "left.png"))
        right = cv2.imread(str(sample / "right.png"))
        calib = load_calibration_json(str(sample / "calib.json"))

        # Rectify (no-op)
        rl, rr = rectify_pair(left, right, stereo_calib=None)

        # Disparity
        disp = compute_disparity(rl, rr, method="sgbm",
                                  num_disparities=96, block_size=7)

        # Depth
        depth_raw = disparity_to_depth(disp,
                                        focal_length_px=calib["focal_length_px"],
                                        baseline_m=calib["baseline_m"])
        depth = clean_depth_map(depth_raw, max_depth_m=20.0, median_ksize=5)

        # Outputs
        out_dir = tmp_path / "output"
        disp_dir  = out_dir / "disparity_maps"
        cloud_dir = out_dir / "point_clouds"
        disp_dir.mkdir(parents=True)
        cloud_dir.mkdir(parents=True)

        disp_vis = disparity_to_visualizable(disp)
        cv2.imwrite(str(disp_dir / "disparity.png"), disp_vis)

        depth_color = depth_to_visualizable(depth)
        cv2.imwrite(str(disp_dir / "depth_colormap.png"), depth_color)

        pts, col = depth_to_pointcloud(depth, rl,
                                        focal_length_px=calib["focal_length_px"],
                                        cx=calib["cx"], cy=calib["cy"])
        ply_path = cloud_dir / "reconstruction.ply"
        save_ply(str(ply_path), pts, col)

        # Assertions
        assert (disp_dir / "disparity.png").exists(),    "disparity.png not produced"
        assert (disp_dir / "depth_colormap.png").exists(), "depth_colormap.png not produced"
        assert ply_path.exists(),                         "reconstruction.ply not produced"
        assert ply_path.stat().st_size > 100,             "PLY file is suspiciously small"

    def test_pipeline_produces_positive_depth(self, tmp_path):
        """The pipeline should yield at least some positive depth values."""
        sample = tmp_path / "sample2"
        generate_sample_pair(str(sample))
        left  = cv2.imread(str(sample / "left.png"))
        right = cv2.imread(str(sample / "right.png"))
        calib = load_calibration_json(str(sample / "calib.json"))

        disp  = compute_disparity(left, right, method="sgbm",
                                   num_disparities=96, block_size=7)
        depth = disparity_to_depth(disp,
                                    focal_length_px=calib["focal_length_px"],
                                    baseline_m=calib["baseline_m"])
        depth = clean_depth_map(depth, max_depth_m=20.0, median_ksize=5)

        valid = depth[np.isfinite(depth)]
        assert len(valid) > 0,         "No valid depth pixels after full pipeline"
        assert np.all(valid > 0),      "Non-positive depth values in pipeline output"

    def test_rectification_noop_preserves_shape(self, sample_images):
        """rectify_pair with stereo_calib=None must return same-shape images."""
        left, right = sample_images
        rl, rr = rectify_pair(left, right, stereo_calib=None)
        assert rl.shape == left.shape
        assert rr.shape == right.shape
