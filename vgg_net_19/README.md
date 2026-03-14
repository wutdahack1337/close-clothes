# VGG19 + KNN — Tìm kiếm ảnh quần áo

Tìm K ảnh giống nhất với một ảnh đầu vào, sử dụng VGG19 để trích xuất đặc trưng và KNN để tìm kiếm.

## Kiến trúc

```
Ảnh đầu vào
  → Resize(256) → CenterCrop(224) → Normalize(ImageNet)
  → VGG19 pretrained (bỏ FC 1000-class cuối)
  → Vector đặc trưng 4096-dim
  → L2-normalize
  → KNN (Euclidean distance)
  → Top-K ảnh giống nhất
```

**VGG19 classifier sau khi chỉnh sửa:**
```
Linear(25088, 4096) → ReLU → Dropout
Linear(4096,  4096) → ReLU → Dropout   ← output tại đây (4096-dim)
[Linear(4096, 1000)]                    ← bỏ lớp này
```

## Files

| File | Mô tả |
|------|-------|
| `main.py` | Toàn bộ logic: build index + query |
| `features.npz` | Feature index (sinh ra sau `--build`, ~120MB) |

## Sử dụng

### Build index (chạy 1 lần)

Trích xuất đặc trưng toàn bộ dataset và lưu vào `features.npz`:

```bash
python vgg_net_19/main.py --build
```

Output mẫu:
```
Using device: cuda
  Done: Blazer (500 images)
  Done: Celana_Panjang (500 images)
  ...
  Done: Sweter (500 images)

Index saved -> vgg_net_19/features.npz
Total: 7500 images, 4096 dims
```

### Query — tìm K ảnh giống nhất

```bash
python vgg_net_19/main.py --query <đường_dẫn_ảnh> [--k <số_lượng>]
```

Ví dụ:

```bash
python vgg_net_19/main.py --query clothes_dataset/Kaos/abc.jpg --k 5
```

Output mẫu:
```
Top-5 results for: clothes_dataset/Kaos/abc.jpg

  [1] dist=0.0000  label=Kaos  path=clothes_dataset/Kaos/abc.jpg
  [2] dist=0.1023  label=Kaos  path=clothes_dataset/Kaos/xyz.jpg
  [3] dist=0.1157  label=Polo  path=clothes_dataset/Polo/def.jpg
  [4] dist=0.1284  label=Kaos  path=clothes_dataset/Kaos/ghi.jpg
  [5] dist=0.1391  label=Kaos  path=clothes_dataset/Kaos/jkl.jpg
```

### Tham số CLI

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--build` | Xây dựng feature index | — |
| `--query IMAGE_PATH` | Đường dẫn ảnh truy vấn | — |
| `--k K` | Số lượng kết quả trả về | `5` |

## Ghi chú kỹ thuật

- **Distance metric:** Euclidean trên vector đã L2-normalize
- **Index size:** 7,500 hình × 4096 × 4 bytes ≈ 120MB