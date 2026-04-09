"""
Tests for VGG19 feature extraction module.
Skips model-loading tests if CUDA/CPU inference is too slow for CI.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vgg():
    path = os.path.join(PROJECT_ROOT, "vgg_net_19", "main.py")
    spec = importlib.util.spec_from_file_location("vgg_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vgg():
    return _load_vgg()


@pytest.fixture(scope="module")
def vgg_model(vgg):
    """Load the VGG19 model once per test session."""
    model = vgg.build_model()
    model.eval()
    return model


# ── build_model ───────────────────────────────────────────────────────────────

class TestBuildModel:
    def test_returns_nn_module(self, vgg, vgg_model):
        """build_model() must return a torch.nn.Module."""
        import torch.nn as nn
        assert isinstance(vgg_model, nn.Module)

    def test_output_dim_is_4096(self, vgg, vgg_model):
        """Model output must be 4096-dimensional."""
        dummy = torch.zeros(1, 3, 224, 224).to(vgg.DEVICE)
        with torch.no_grad():
            out = vgg_model(dummy)
        assert out.shape == (1, 4096), f"Expected (1, 4096), got {out.shape}"

    def test_model_in_eval_mode(self, vgg, vgg_model):
        """Model must be in eval (not training) mode."""
        assert not vgg_model.training, "Model should be in eval mode"


# ── transform ────────────────────────────────────────────────────────────────

class TestTransform:
    def test_transform_output_shape(self, vgg):
        """Transform must produce a (3, 224, 224) tensor."""
        img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
        tensor = vgg.transform(img)
        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_transform_output_dtype(self, vgg):
        """Transform must produce a float32 tensor."""
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        tensor = vgg.transform(img)
        assert tensor.dtype == torch.float32

    def test_transform_normalizes_values(self, vgg):
        """After ImageNet normalization, pixel values should not be in [0, 1]."""
        # A white image (all 255) normalized by ImageNet stats shifts values
        arr = np.full((224, 224, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        tensor = vgg.transform(img)
        # ImageNet normalization of white: (1 - mean) / std > 1 for most channels
        assert tensor.max().item() > 1.0 or tensor.min().item() < 0.0


# ── full inference ────────────────────────────────────────────────────────────

class TestInference:
    def test_feature_vector_shape(self, vgg, vgg_model):
        """Full pipeline must produce a (4096,) feature vector."""
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        tensor = vgg.transform(img).unsqueeze(0).to(vgg.DEVICE)
        with torch.no_grad():
            feat = vgg_model(tensor).cpu().numpy().squeeze()
        assert feat.shape == (4096,)

    def test_feature_l2_norm(self, vgg, vgg_model):
        """After L2 normalization, feature vector must have unit norm."""
        img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
        tensor = vgg.transform(img).unsqueeze(0).to(vgg.DEVICE)
        with torch.no_grad():
            feat = vgg_model(tensor).cpu().numpy().squeeze()
        feat = feat / np.clip(np.linalg.norm(feat), 1e-8, None)
        assert abs(np.linalg.norm(feat) - 1.0) < 1e-5

    def test_different_images_differ(self, vgg, vgg_model):
        """Two different images must produce different feature vectors."""
        img1 = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8))

        def get_feat(img):
            t = vgg.transform(img).unsqueeze(0).to(vgg.DEVICE)
            with torch.no_grad():
                f = vgg_model(t).cpu().numpy().squeeze()
            return f / np.clip(np.linalg.norm(f), 1e-8, None)

        assert not np.allclose(get_feat(img1), get_feat(img2))

    def test_same_image_deterministic(self, vgg, vgg_model):
        """Same image must produce identical features when queried twice."""
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        def get_feat(img):
            t = vgg.transform(img).unsqueeze(0).to(vgg.DEVICE)
            with torch.no_grad():
                f = vgg_model(t).cpu().numpy().squeeze()
            return f

        np.testing.assert_array_equal(get_feat(img), get_feat(img))
