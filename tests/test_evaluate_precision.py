"""
Tests for the evaluate_precision script (Precision@K logic).
"""

import hashlib
import importlib.util
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_eval():
    path = os.path.join(PROJECT_ROOT, "scripts", "evaluate_precision.py")
    spec = importlib.util.spec_from_file_location("eval_prec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _load_eval()


# ── file_md5 ──────────────────────────────────────────────────────────────────

class TestFileMd5:
    def test_same_file_same_hash(self, ev, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"hello")
        assert ev.file_md5(str(f)) == ev.file_md5(str(f))

    def test_different_content_different_hash(self, ev, tmp_path):
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert ev.file_md5(str(f1)) != ev.file_md5(str(f2))

    def test_returns_hex_string(self, ev, tmp_path):
        f = tmp_path / "x.jpg"
        f.write_bytes(b"data")
        result = ev.file_md5(str(f))
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest is always 32 chars


# ── load_ground_truth_hashes ──────────────────────────────────────────────────

class TestLoadGroundTruthHashes:
    def test_excludes_query_jpg(self, ev, tmp_path):
        (tmp_path / "query.jpg").write_bytes(b"query")
        (tmp_path / "001.jpg").write_bytes(b"gt1")
        hashes = ev.load_ground_truth_hashes(str(tmp_path))
        query_hash = hashlib.md5(b"query").hexdigest()
        assert query_hash not in hashes

    def test_includes_other_jpgs(self, ev, tmp_path):
        (tmp_path / "query.jpg").write_bytes(b"query")
        (tmp_path / "001.jpg").write_bytes(b"gt1")
        (tmp_path / "002.jpg").write_bytes(b"gt2")
        hashes = ev.load_ground_truth_hashes(str(tmp_path))
        assert len(hashes) == 2

    def test_empty_folder_returns_empty_set(self, ev, tmp_path):
        (tmp_path / "query.jpg").write_bytes(b"query")
        hashes = ev.load_ground_truth_hashes(str(tmp_path))
        assert hashes == set()


# ── precision_at_k ────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def _setup_images(self, tmp_path, contents):
        """Write files and return their paths and MD5 hashes."""
        paths = {}
        hashes = {}
        for name, data in contents.items():
            p = tmp_path / name
            p.write_bytes(data)
            paths[name] = str(p)
            hashes[name] = hashlib.md5(data).hexdigest()
        return paths, hashes

    def test_all_hits(self, ev, tmp_path):
        """All K results match ground truth → Precision = 1.0."""
        paths, hashes = self._setup_images(tmp_path, {
            "a.jpg": b"aaa",
            "b.jpg": b"bbb",
        })
        gt_hashes = {hashes["a.jpg"], hashes["b.jpg"]}
        result_paths = ["a.jpg", "b.jpg"]
        p = ev.precision_at_k(result_paths, gt_hashes, str(tmp_path))
        assert p == pytest.approx(1.0)

    def test_no_hits(self, ev, tmp_path):
        """No results match ground truth → Precision = 0.0."""
        paths, hashes = self._setup_images(tmp_path, {
            "a.jpg": b"aaa",
            "b.jpg": b"bbb",
        })
        gt_hashes = {hashlib.md5(b"other").hexdigest()}
        result_paths = ["a.jpg", "b.jpg"]
        p = ev.precision_at_k(result_paths, gt_hashes, str(tmp_path))
        assert p == pytest.approx(0.0)

    def test_partial_hits(self, ev, tmp_path):
        """1 of 4 results matches → Precision = 0.25."""
        paths, hashes = self._setup_images(tmp_path, {
            "a.jpg": b"aaa",
            "b.jpg": b"bbb",
            "c.jpg": b"ccc",
            "d.jpg": b"ddd",
        })
        gt_hashes = {hashes["b.jpg"]}
        result_paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        p = ev.precision_at_k(result_paths, gt_hashes, str(tmp_path))
        assert p == pytest.approx(0.25)

    def test_empty_results(self, ev, tmp_path):
        """Empty result list → Precision = 0.0 (no division error)."""
        p = ev.precision_at_k([], {hashlib.md5(b"x").hexdigest()}, str(tmp_path))
        assert p == pytest.approx(0.0)

    def test_missing_file_skipped(self, ev, tmp_path):
        """Result path pointing to non-existent file is skipped gracefully."""
        (tmp_path / "real.jpg").write_bytes(b"real")
        gt_hashes = {hashlib.md5(b"real").hexdigest()}
        result_paths = ["real.jpg", "ghost.jpg"]
        p = ev.precision_at_k(result_paths, gt_hashes, str(tmp_path))
        # Only "real.jpg" counts, ghost is skipped
        assert p == pytest.approx(0.5)
