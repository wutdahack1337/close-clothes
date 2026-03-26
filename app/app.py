import importlib.util
import os
import threading
import time
import uuid

import numpy as np
import torch
from flask import Flask, abort, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CLOTHES_DATASET_ROOT = os.path.realpath(os.path.join(PROJECT_ROOT, "clothes_dataset"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_K = 3


def _load_module(name, rel_path):
    path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Loading modules...")
vgg_mod  = _load_module("vgg_main",  "vgg_net_19/main.py")
chog_mod = _load_module("chog_main", "color_histogram_and_hog/main.py")

# ----- VGG19 -----
print("Loading VGG19 model...")
vgg_model = vgg_mod.build_model()
vgg_model.eval()
_vgg_lock = threading.Lock()

print("Loading VGG19 feature index...")
_d = np.load(os.path.join(PROJECT_ROOT, "vgg_net_19", "features.npz"))
features_vgg = _d["features"].astype(np.float32)
paths_vgg    = _d["paths"]
labels_vgg   = _d["labels"]
nn_vgg = NearestNeighbors(metric="euclidean", algorithm="brute").fit(features_vgg)
print(f"  VGG19 index: {features_vgg.shape}")

# ----- Color Histogram + HOG -----
print("Loading Color Histogram + HOG feature index...")
_d = np.load(os.path.join(PROJECT_ROOT, "color_histogram_and_hog", "features.npz"))
features_chog = _d["features"].astype(np.float32)
paths_chog    = _d["paths"]
labels_chog   = _d["labels"]
nn_chog = NearestNeighbors(metric="euclidean", algorithm="brute").fit(features_chog)
print(f"  CHOG index: {features_chog.shape}")

print("\nAll models loaded. Starting Flask...")

app = Flask(__name__)


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _query_vgg(img: Image.Image, k: int):
    tensor = vgg_mod.transform(img).unsqueeze(0).to(vgg_mod.DEVICE)
    with _vgg_lock:
        with torch.no_grad():
            feat = vgg_model(tensor).cpu().numpy().squeeze()
    feat = feat / np.clip(np.linalg.norm(feat), 1e-8, None)
    distances, indices = nn_vgg.kneighbors(feat.reshape(1, -1), n_neighbors=k)
    return [
        {"rank": r + 1, "dist": float(distances[0][r]), "label": labels_vgg[i], "path": str(paths_vgg[i])}
        for r, i in enumerate(indices[0])
    ]


def _query_chog(img: Image.Image, k: int):
    feat = chog_mod.extract_features(img)
    distances, indices = nn_chog.kneighbors(feat.reshape(1, -1), n_neighbors=k)
    return [
        {"rank": r + 1, "dist": float(distances[0][r]), "label": labels_chog[i], "path": str(paths_chog[i])}
        for r, i in enumerate(indices[0])
    ]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/search")
def search():
    file = request.files.get("image")
    if not file or not file.filename:
        return render_template("index.html", error="Vui lòng chọn ảnh.")

    if not _allowed(file.filename):
        return render_template("index.html", error="Định dạng ảnh không hỗ trợ (only .jpg, .jpeg, .png).")

    try:
        k = int(request.form.get("k", DEFAULT_K))
        if k < 1:
            k = 1
        if k > len(features_vgg):
            k = len(features_vgg)
    except ValueError:
        k = DEFAULT_K

    # Save uploaded image
    ext = os.path.splitext(file.filename.lower())[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        img = Image.open(save_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        if os.path.exists(save_path):
            os.remove(save_path)
        return render_template("index.html", error="Tệp đã tải lên không phải là ảnh hợp lệ.")

    # VGG19 query
    t0 = time.perf_counter()
    vgg_results = _query_vgg(img, k)
    vgg_time_ms = round((time.perf_counter() - t0) * 1000)

    # Color Histogram + HOG query
    t0 = time.perf_counter()
    chog_results = _query_chog(img, k)
    chog_time_ms = round((time.perf_counter() - t0) * 1000)

    return render_template(
        "index.html",
        uploaded_filename=filename,
        k=k,
        vgg_results=vgg_results,
        vgg_time_ms=vgg_time_ms,
        chog_results=chog_results,
        chog_time_ms=chog_time_ms,
    )


@app.get("/dataset-image")
def dataset_image():
    rel_path = request.args.get("path", "")
    if not rel_path:
        abort(400)

    candidate = os.path.join(CLOTHES_DATASET_ROOT, rel_path)
    real = os.path.realpath(candidate)
    if not real.startswith(CLOTHES_DATASET_ROOT + os.sep):
        abort(403)
    if not real.lower().endswith(".jpg"):
        abort(403)
    if not os.path.isfile(real):
        abort(404)
    return send_file(real, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=1337)
