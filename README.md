# close-clothes
data mining project

You wanna find k similar clothes to a query image.

## Requirement
- clothes_dataset folder: download from kaggle.com/datasets/ryanbadai/clothes-dataset

## Quick Start
```
source run.sh
browser http://localhost:1337
```

## Details
```
python3 -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt

python3 color_histogram_and_hog/main.py --build
python3 vgg_net_19/main.py --build

python3 app/app.py
browser http://localhost:1337
```

## Dataset Duplicate Filter
Detect duplicated images that appear in different labels:

```bash
source .venv/bin/activate

# 1) Report only (safe)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action report

# 2) Move duplicated files (except 1 keeper per exact-duplicate group)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action move --quarantine-dir duplicate_quarantine

# 3) Delete duplicated files (dangerous)
python scripts/find_cross_label_duplicates.py --root clothes_dataset --action delete --yes
```

Reports are written to:
- `duplicate_reports/duplicate_report.json`

Delete images from a manual list file:

```bash
# need_delete.txt example:
# Jaket/075.jpg
# Jaket/078.jpg
# Polo/251.jpg

source .venv/bin/activate

# 1) Preview only
python scripts/find_cross_label_duplicates.py \
	--root clothes_dataset \
	--delete-list need_delete.txt \
	--dry-run

# 2) Delete for real (requires --yes)
python scripts/find_cross_label_duplicates.py \
	--root clothes_dataset \
	--delete-list need_delete.txt \
	--yes
```