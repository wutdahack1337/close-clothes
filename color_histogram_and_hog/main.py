import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "clothes_dataset")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "features.npz")
IMG_SIZE = 128
N_BINS = 32
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def _l2(v):
    return v / np.clip(np.linalg.norm(v), 1e-8, None)


def extract_features(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]

    # Color Histogram on HSV
    hsv = rgb2hsv(arr)  # [H, W, 3]
    color_hist = np.concatenate([
        np.histogram(hsv[:, :, c], bins=N_BINS, range=(0.0, 1.0))[0].astype(np.float32)
        for c in range(3)
    ])  # [96]
    color_hist = _l2(color_hist)

    # HOG on grayscale
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    hog_feat = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    ).astype(np.float32)  # [8100]
    hog_feat = _l2(hog_feat)

    return np.concatenate([color_hist, hog_feat])  # [8196]


def build_index():
    label_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "*")))
    if not label_dirs:
        raise FileNotFoundError(f"No label directories found in {DATASET_ROOT}")

    all_paths, all_labels, all_features = [], [], []

    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        image_paths = sorted(glob.glob(os.path.join(label_dir, "*.jpg")))
        indexed_count = 0

        for p in image_paths:
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    feat = extract_features(img)
                all_features.append(feat)
                all_paths.append(os.path.relpath(p, DATASET_ROOT))
                all_labels.append(label)
                indexed_count += 1
            except Exception as e:
                print(f"  [WARN] skipping {p}: {e}")

        print(f"  Done: {label} (found {len(image_paths)} images, indexed {indexed_count})")

    if not all_features:
        raise ValueError(f"No features were extracted from any images under {DATASET_ROOT}.")

    features_matrix = np.vstack(all_features)
    np.savez(
        INDEX_PATH,
        features=features_matrix,
        paths=np.array(all_paths),
        labels=np.array(all_labels),
    )
    print(f"\nIndex saved -> {INDEX_PATH}")
    print(f"Total: {features_matrix.shape[0]} images, {features_matrix.shape[1]} dims")


def query_index(query_path, k):
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}. Run --build first.")

    data = np.load(INDEX_PATH)
    features = data["features"]
    paths = data["paths"]
    labels = data["labels"]

    num_samples = len(features)
    if num_samples == 0:
        raise ValueError("The index is empty. Build the index with at least one image before querying.")
    if k <= 0:
        raise ValueError(f"Parameter k must be a positive integer, but got k={k}.")
    if k > num_samples:
        print(f"Requested k={k} is larger than the number of indexed samples ({num_samples}). Clamping k to {num_samples}")
        k = num_samples

    with Image.open(query_path) as img:
        img = img.convert("RGB")
        feat = extract_features(img)

    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nn_model.fit(features)
    distances, indices = nn_model.kneighbors(feat.reshape(1, -1))

    print(f"\nTop-{k} results for: {query_path}\n")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        rel_path = paths[idx] if isinstance(paths[idx], str) else paths[idx].decode('utf-8')
        print(f"  [{rank}] dist={dist:.4f}  label={labels[idx]}  path={rel_path}")


def main():
    parser = argparse.ArgumentParser(description="Clothing Image Similarity Search (Color Histogram + HOG + KNN)")
    parser.add_argument("--build",           action="store_true",  help="Build feature index from dataset")
    parser.add_argument("--query", type=str, metavar="IMAGE_PATH", help="Path to query image")
    parser.add_argument("--k",     type=int, default=5,            help="Number of nearest neighbors (default: 5)")
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        return

    if args.build:
        build_index()

    if args.query:
        query_index(args.query, args.k)


if __name__ == "__main__":
    main()
