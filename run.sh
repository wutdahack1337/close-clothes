#!/bin/bash
set -e

if [ ! -f .venv/bin/activate ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo

if [ ! -f color_histogram_and_hog/features.npz ]; then
    python3 color_histogram_and_hog/main.py --build
else
    echo "color_histogram_and_hog/features.npz already exists, skipping build."
fi

if [ ! -f vgg_net_19/features.npz ]; then
    python3 vgg_net_19/main.py --build
else
    echo "vgg_net_19/features.npz already exists, skipping build."
fi

echo

python3 app/app.py
