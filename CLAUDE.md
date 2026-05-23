# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Always activate the virtual environment first:**
```bash
source .venv/bin/activate
```

This must be done before running any Python commands or scripts.

## Project Overview

**close-clothes** is a clothing similarity search application that uses two feature extraction methods:

1. **VGG19 (Deep Learning)**: Uses a pre-trained ImageNet model to extract 4096-dim feature vectors from images
2. **Color Histogram + HOG**: Combines HSV color histograms (96 dims) with HOG features (8100 dims) for a 8196-dim vector

Both methods build searchable feature indices from a dataset of clothing images. The Flask web app (`app/app.py`) allows users to upload query images and find k-nearest neighbors using either method.

### Key Components

- **`app/app.py`**: Flask web server (runs on localhost:1337)
  - Loads both feature models and indices
  - Provides HTTP endpoints for image search
  - Uses thread locks for concurrent VGG19 queries
  
- **`vgg_net_19/main.py`**: VGG19 feature extraction
  - `build_model()`: Creates VGG19 with final 1000-class layer removed
  - `build_index()`: Iterates dataset, extracts 4096-dim features, saves to `features.npz`
  
- **`color_histogram_and_hog/main.py`**: CHOG feature extraction
  - `extract_features()`: Computes 8196-dim feature vector from single image
  - `build_index()`: Batch processes dataset, saves to `features.npz`
  
- **`scripts/find_cross_label_duplicates.py`**: Dataset cleaning utility

### Data Files

- `clothes_dataset/`: Root directory with label subdirectories (Jaket/, Polo/, etc.), each containing .jpg images
- `vgg_net_19/features.npz`: Pre-built VGG19 index (~120MB) with keys: `features`, `paths`, `labels`
- `color_histogram_and_hog/features.npz`: Pre-built CHOG index (~240MB) with same structure
- `app/static/uploads/`: Temporary storage for uploaded query images

## Common Commands

### Quick Start
```bash
source .venv/bin/activate
source run.sh
# Opens web interface at http://localhost:1337
```

### Build Feature Indices (if missing)
```bash
source .venv/bin/activate
python3 vgg_net_19/main.py --build
python3 color_histogram_and_hog/main.py --build
```

### Run Flask App Directly
```bash
source .venv/bin/activate
python3 app/app.py
# Runs on http://localhost:1337
```

### Extract Features from Single Image
```bash
source .venv/bin/activate
# VGG19 only
python3 vgg_net_19/main.py --query path/to/image.jpg

# CHOG only
python3 color_histogram_and_hog/main.py --query path/to/image.jpg
```

### Dataset Duplicate Detection
```bash
source .venv/bin/activate

# Report only (safe)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action report

# Move duplicates to quarantine
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action move --quarantine-dir duplicate_quarantine

# Delete duplicates (requires --yes flag)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action delete --yes

# Delete from manual list with dry-run first
python scripts/find_cross_label_duplicates.py --root clothes_dataset --delete-list need_delete.txt --dry-run
python scripts/find_cross_label_duplicates.py --root clothes_dataset --delete-list need_delete.txt --yes
```

## Git Workflow

Do not commit directly to `main`. Always create a feature branch and open a pull request:
```bash
git checkout -b feature/your-feature-name
# ... make changes ...
git push origin feature/your-feature-name
# Open PR on GitHub
```

## Key Architectural Notes

- **Path handling**: Both feature modules use relative paths from `PROJECT_ROOT` (parent of `vgg_net_19/` and `color_histogram_and_hog/`)
- **GPU support**: VGG19 uses CUDA if available, falls back to CPU
- **Thread safety**: Flask app uses a lock (`_vgg_lock`) for concurrent VGG19 inference
- **Feature normalization**: Both methods L2-normalize features before indexing
- **Nearest neighbors**: Uses scikit-learn's brute-force NearestNeighbors with Euclidean distance
