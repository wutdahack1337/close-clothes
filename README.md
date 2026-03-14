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