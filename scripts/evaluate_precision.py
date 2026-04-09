"""
Evaluate Precision@K for both VGG19 and CHOG models.

Each test folder under test_dataset/ contains:
  - query.jpg          : the query image
  - *.jpg (others)     : ground truth images that should appear in top-K results

A result image is considered a match if it is pixel-identical (MD5 hash)
to any ground truth image.

Usage:
    python scripts/evaluate_precision.py [--k K] [--test-dir path]
"""

import argparse
import hashlib
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_module(name, rel_path):
    path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_ground_truth_hashes(test_folder: str) -> set:
    """Return MD5 hashes of every file in test_folder except query.jpg."""
    hashes = set()
    for fname in os.listdir(test_folder):
        if fname == "query.jpg":
            continue
        fpath = os.path.join(test_folder, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(".jpg"):
            hashes.add(file_md5(fpath))
    return hashes


def precision_at_k(result_paths: list[str], gt_hashes: set, clothes_root: str) -> float:
    """
    result_paths : relative paths stored in features.npz (e.g. 'Blazer/001.jpg')
    gt_hashes    : MD5 hashes of ground-truth images
    clothes_root : absolute path to clothes_dataset/
    Returns Precision@K in [0, 1].
    """
    hits = 0
    for rel in result_paths:
        abs_path = os.path.join(clothes_root, rel)
        if not os.path.isfile(abs_path):
            continue
        if file_md5(abs_path) in gt_hashes:
            hits += 1
    return hits / len(result_paths) if result_paths else 0.0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Precision@K")
    parser.add_argument("--k",        type=int, default=5,  help="Number of results to retrieve (default: 5)")
    parser.add_argument("--test-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "test_dataset"),
                        help="Path to test_dataset folder")
    args = parser.parse_args()

    K = args.k
    TEST_DIR = args.test_dir
    CLOTHES_ROOT = os.path.join(PROJECT_ROOT, "clothes_dataset")

    # ── Load models & indices ────────────────────────────────────────────────
    print("Loading modules...")
    vgg_mod  = _load_module("vgg_main",  "vgg_net_19/main.py")
    chog_mod = _load_module("chog_main", "color_histogram_and_hog/main.py")

    print("Loading VGG19 model...")
    vgg_model = vgg_mod.build_model()
    vgg_model.eval()

    print("Loading VGG19 index...")
    _d = np.load(os.path.join(PROJECT_ROOT, "vgg_net_19", "features.npz"))
    features_vgg = _d["features"].astype(np.float32)
    paths_vgg    = _d["paths"]
    nn_vgg = NearestNeighbors(metric="euclidean", algorithm="brute").fit(features_vgg)

    print("Loading CHOG index...")
    _d = np.load(os.path.join(PROJECT_ROOT, "color_histogram_and_hog", "features.npz"))
    features_chog = _d["features"].astype(np.float32)
    paths_chog    = _d["paths"]
    nn_chog = NearestNeighbors(metric="euclidean", algorithm="brute").fit(features_chog)

    print(f"\nEvaluating Precision@{K}\n{'='*60}")

    test_cases = sorted([
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ])

    results_vgg  = []
    results_chog = []

    for tc in test_cases:
        folder = os.path.join(TEST_DIR, tc)
        query_path = os.path.join(folder, "query.jpg")
        if not os.path.isfile(query_path):
            print(f"  [{tc}] SKIP — no query.jpg")
            continue

        gt_hashes = load_ground_truth_hashes(folder)
        if not gt_hashes:
            print(f"  [{tc}] SKIP — no ground truth images")
            continue

        img = Image.open(query_path).convert("RGB")

        # VGG19
        tensor = vgg_mod.transform(img).unsqueeze(0).to(vgg_mod.DEVICE)
        with torch.no_grad():
            feat = vgg_model(tensor).cpu().numpy().squeeze()
        feat = feat / max(np.linalg.norm(feat), 1e-8)
        dists_v, idxs_v = nn_vgg.kneighbors(feat.reshape(1, -1), n_neighbors=K)
        top_paths_vgg = [str(paths_vgg[i]) for i in idxs_v[0]]
        p_vgg = precision_at_k(top_paths_vgg, gt_hashes, CLOTHES_ROOT)
        results_vgg.append(p_vgg)

        # CHOG
        feat_c = chog_mod.extract_features(img)
        dists_c, idxs_c = nn_chog.kneighbors(feat_c.reshape(1, -1), n_neighbors=K)
        top_paths_chog = [str(paths_chog[i]) for i in idxs_c[0]]
        p_chog = precision_at_k(top_paths_chog, gt_hashes, CLOTHES_ROOT)
        results_chog.append(p_chog)

        print(f"  [{tc}]  GT={len(gt_hashes)}  VGG19={p_vgg:.2f}  CHOG={p_chog:.2f}")
        print(f"    VGG19 top-{K}: {[os.path.basename(p) for p in top_paths_vgg]}")
        print(f"    CHOG  top-{K}: {[os.path.basename(p) for p in top_paths_chog]}")

    print(f"\n{'='*60}")
    mean_vgg  = np.mean(results_vgg)  if results_vgg  else 0.0
    mean_chog = np.mean(results_chog) if results_chog else 0.0
    print(f"  Mean Precision@{K}  VGG19 : {mean_vgg:.4f}")
    print(f"  Mean Precision@{K}  CHOG  : {mean_chog:.4f}")
    print(f"  Winner: {'VGG19' if mean_vgg >= mean_chog else 'CHOG'}")


if __name__ == "__main__":
    main()
