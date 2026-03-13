import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "clothes_dataset")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "features.npz")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model():
    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights)
    # Remove the final Linear(4096, 1000), keep 4096-dim output
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.eval()
    model.to(DEVICE)
    return model


def build_index(model):
    label_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "*")))
    if not label_dirs:
        raise FileNotFoundError(f"No label directories found in {DATASET_ROOT}")

    all_paths, all_labels, all_features = [], [], []

    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        image_paths = sorted(glob.glob(os.path.join(label_dir, "*.jpg")))

        for batch_start in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[batch_start: batch_start + BATCH_SIZE]
            batch_tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(TRANSFORM(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"  [WARN] skipping {p}: {e}")

            if not batch_tensors:
                continue

            tensor_batch = torch.stack(batch_tensors).to(DEVICE)  # [B, 3, 224, 224]
            with torch.no_grad():
                feats = model(tensor_batch).cpu().numpy()  # [B, 4096]

            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(norms, 1e-8, None)

            all_features.append(feats)
            all_paths.extend(valid_paths)
            all_labels.extend([label] * len(valid_paths))

        print(f"  Done: {label} ({len(image_paths)} images)")

    features_matrix = np.vstack(all_features)  # [N, 4096]
    np.savez(
        INDEX_PATH,
        features=features_matrix,
        paths=np.array(all_paths),
        labels=np.array(all_labels),
    )
    print(f"\nIndex saved -> {INDEX_PATH}")
    print(f"Total: {features_matrix.shape[0]} images, {features_matrix.shape[1]} dims")


def query_index(model, query_path, k):
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run --build first."
        )

    data = np.load(INDEX_PATH, allow_pickle=True)
    features = data["features"]  # [N, 4096]
    paths = data["paths"]
    labels = data["labels"]

    img = Image.open(query_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().squeeze()  # [4096]
    feat = feat / np.clip(np.linalg.norm(feat), 1e-8, None)

    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nn_model.fit(features)
    distances, indices = nn_model.kneighbors(feat.reshape(1, -1))

    print(f"\nTop-{k} results for: {query_path}\n")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        print(f"  [{rank}] dist={dist:.4f}  label={labels[idx]}  path={paths[idx]}")


def main():
    parser = argparse.ArgumentParser(description="Clothing Image Similarity Search (VGG19 + KNN)")
    parser.add_argument("--build",           action="store_true",  help="Build feature index from dataset")
    parser.add_argument("--query", type=str, metavar="IMAGE_PATH", help="Path to query image")
    parser.add_argument("--k",     type=int, default=5,            help="Number of nearest neighbors (default: 5)")
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        return

    print(f"Using device: {DEVICE}")
    model = build_model()

    if args.build:
        build_index(model)

    if args.query:
        query_index(model, args.query, args.k)


if __name__ == "__main__":
    main()
