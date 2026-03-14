# Color Histogram + HOG + KNN — Tìm kiếm ảnh quần áo

Tìm K ảnh giống nhất với một ảnh đầu vào, sử dụng Color Histogram và HOG để trích xuất đặc trưng và KNN để tìm kiếm.

## Kiến trúc

```
Ảnh đầu vào
  → Resize(128×128)
  → Color Histogram (HSV, 32 bins/kênh) → L2-normalize → 96-dim
  → HOG (grayscale, 9 orient, 8×8 cells) → L2-normalize → 8100-dim
  → Concat → 8196-dim
  → KNN (Euclidean distance)
  → Top-K ảnh giống nhất
```

**Lý do normalize từng thành phần riêng:** đảm bảo cả hai đóng góp bằng nhau vào khoảng cách
- HOG tính từ gradient ảnh — có magnitude cao hơn nhiều so với Color Histogram
- Color Histogram là tần suất pixel — tự nhiên trong khoảng [0, 1]

## Files

| File | Mô tả |
|------|-------|
| `main.py` | Toàn bộ logic: build index + query |
| `features.npz` | Feature index (sinh ra sau `--build`, ~246MB) |

## Sử dụng

### Build index

Trích xuất đặc trưng toàn bộ dataset và lưu vào `features.npz`:

```bash
python color_histogram_and_hog/main.py --build
```

Output mẫu:
```
  Done: Blazer (500 images)
  Done: Celana_Panjang (500 images)
  ...
  Done: Sweter (500 images)

Index saved -> color_histogram_and_hog/features.npz
Total: 7500 images, 8196 dims
```

### Query — tìm K ảnh giống nhất

```bash
python color_histogram_and_hog/main.py --query <đường_dẫn_ảnh> [--k <số_lượng>]
```

Ví dụ:

```bash
python color_histogram_and_hog/main.py --query clothes_dataset/Kaos/abc.jpg --k 5
```

Output mẫu:
```
Top-5 results for: clothes_dataset/Kaos/abc.jpg

  [1] dist=0.0000  label=Kaos  path=clothes_dataset/Kaos/abc.jpg
  [2] dist=0.3241  label=Kaos  path=clothes_dataset/Kaos/xyz.jpg
  [3] dist=0.3587  label=Polo  path=clothes_dataset/Polo/def.jpg
  [4] dist=0.3712  label=Kaos  path=clothes_dataset/Kaos/ghi.jpg
  [5] dist=0.3894  label=Kaos  path=clothes_dataset/Kaos/jkl.jpg
```

### Tham số CLI

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--build` | Xây dựng feature index | — |
| `--query IMAGE_PATH` | Đường dẫn ảnh truy vấn | — |
| `--k K` | Số lượng kết quả trả về | `5` |

## Ghi chú kỹ thuật

- **Color Histogram:** HSV colorspace, 32 bins × 3 kênh = 96-dim
- **HOG:** grayscale, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2) = 8100-dim
- **Distance metric:** Euclidean trên vector đã normalize
- **Index size:** 7,500 hình × 8,196 × 4 bytes ≈ 246MB
