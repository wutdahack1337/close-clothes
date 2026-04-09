# close-clothes

Clothing similarity search using VGG19 and Color Histogram + HOG.

## Requirements

- Python 3.12+
- `clothes_dataset/` — download from [kaggle.com/datasets/ryanbadai/clothes-dataset](https://kaggle.com/datasets/ryanbadai/clothes-dataset)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 color_histogram_and_hog/main.py --build
python3 vgg_net_19/main.py --build

python3 app/app.py
# Open http://localhost:1337
```

Or just:

```bash
source run.sh
```

## Evaluate Precision@K

```bash
python scripts/evaluate_precision.py --k 5
```

## Dataset Duplicate Filter

```bash
# Report only
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action report

# Move duplicates to quarantine
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action move --quarantine-dir duplicate_quarantine

# Delete duplicates
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action delete --yes

# Delete from a manual list (dry-run first)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --delete-list need_delete.txt --dry-run
python scripts/find_cross_label_duplicates.py --root clothes_dataset --delete-list need_delete.txt --yes
```

## Tests

```bash
python -m pytest tests/ -v
```
