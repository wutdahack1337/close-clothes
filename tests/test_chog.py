"""
Tests for Color Histogram + HOG (CHOG) feature extraction module.
"""

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_chog():
    path = os.path.join(PROJECT_ROOT, "color_histogram_and_hog", "main.py")
    spec = importlib.util.spec_from_file_location("chog_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def chog():
    return _load_chog()


@pytest.fixture
def solid_image():
    """A simple 128x128 solid red RGB image."""
    return Image.fromarray(np.full((128, 128, 3), [200, 50, 50], dtype=np.uint8))


@pytest.fixture
def random_image():
    """A random noise RGB image."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── extract_features ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_output_shape(self, chog, solid_image):
        """Feature vector must be 8196-dimensional (96 color + 8100 HOG)."""
        feat = chog.extract_features(solid_image)
        assert feat.shape == (8196,), f"Expected (8196,), got {feat.shape}"

    def test_output_dtype(self, chog, solid_image):
        """Feature vector must be float32."""
        feat = chog.extract_features(solid_image)
        assert feat.dtype == np.float32

    def test_no_nan_or_inf(self, chog, solid_image):
        """Feature vector must not contain NaN or Inf."""
        feat = chog.extract_features(solid_image)
        assert np.all(np.isfinite(feat)), "Feature vector contains NaN or Inf"

    def test_different_images_differ(self, chog, solid_image, random_image):
        """Two visually different images must produce different features."""
        feat1 = chog.extract_features(solid_image)
        feat2 = chog.extract_features(random_image)
        assert not np.allclose(feat1, feat2), "Different images produced identical features"

    def test_same_image_deterministic(self, chog, solid_image):
        """Same image queried twice must produce identical features."""
        feat1 = chog.extract_features(solid_image)
        feat2 = chog.extract_features(solid_image)
        np.testing.assert_array_equal(feat1, feat2)

    def test_accepts_non_square_image(self, chog):
        """Non-square images should be resized and processed without error."""
        img = Image.fromarray(np.zeros((64, 200, 3), dtype=np.uint8))
        feat = chog.extract_features(img)
        assert feat.shape == (8196,)

    def test_accepts_small_image(self, chog):
        """Very small images should be upscaled and processed without error."""
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        feat = chog.extract_features(img)
        assert feat.shape == (8196,)

    def test_color_hist_part_normalized(self, chog, solid_image):
        """The color histogram part (first 96 dims) should be L2-normalized."""
        feat = chog.extract_features(solid_image)
        color_part = feat[:96]
        norm = np.linalg.norm(color_part)
        assert abs(norm - 1.0) < 1e-5, f"Color hist norm={norm:.6f}, expected ~1.0"

    def test_hog_part_normalized(self, chog, random_image):
        """The HOG part (last 8100 dims) should be L2-normalized."""
        feat = chog.extract_features(random_image)
        hog_part = feat[96:]
        norm = np.linalg.norm(hog_part)
        assert abs(norm - 1.0) < 1e-5, f"HOG norm={norm:.6f}, expected ~1.0"


# ── _l2 helper ───────────────────────────────────────────────────────────────

class TestL2Normalization:
    def test_unit_vector(self, chog):
        """_l2 should return a unit vector."""
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = chog._l2(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector_no_error(self, chog):
        """_l2 on zero vector must not raise ZeroDivisionError."""
        v = np.zeros(10, dtype=np.float32)
        result = chog._l2(v)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))
