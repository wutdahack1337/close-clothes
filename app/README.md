# Web Server — Close Clothes

Giao diện web để tìm kiếm K hình ảnh tương tự với hình tải lên, hiển thị kết quả của tất cả mô hình cùng với thời gian phản hồi.

## Kiến trúc tổng quan

```
Ảnh đầu vào + k
  ├── VGG19
  │     → Resize(256) → CenterCrop(224) → Normalize(ImageNet)
  │     → VGG19 pretrained (4096-dim, bỏ FC 1000-class cuối)
  │     → L2-normalize → KNN (Euclidean)
  │     → Top-K kết quả + thời gian phản hồi
  │
  └── Color Histogram + HOG
        → Resize(128×128)
        → Color Histogram (HSV, 32 bins/kênh) → L2-normalize → 96-dim
        → HOG (grayscale, 9 orient, 8×8 cells) → L2-normalize → 8100-dim
        → Concat 8196-dim → KNN (Euclidean)
        → Top-K kết quả + thời gian phản hồi
```

Chi tiết từng mô hình xem tại:
- [`../vgg_net_19/README.md`](../vgg_net_19/README.md)
- [`../color_histogram_and_hog/README.md`](../color_histogram_and_hog/README.md)

## Files

| File | Mô tả |
|------|-------|
| `app.py` | Flask application — routes, startup loading, query logic |
| `templates/index.html` | Giao diện web (Tailwind CSS) |
| `static/uploads/` | Lưu ảnh người dùng tải lên (tạm thời) |

## Khởi động

```bash
python app/app.py
```

Mở trình duyệt tại **http://localhost:1337**

> Lần đầu khởi động sẽ tải VGG19 và hai bộ chỉ mục (~363 MB) vào bộ nhớ — chỉ xảy ra một lần.

## Cách dùng

1. Nhấn **chọn ảnh** (hỗ trợ JPG, JPEG, PNG) — xem preview ngay
2. Nhập **k** — số lượng kết quả muốn trả về (mặc định: 3)
3. Nhấn **Tìm kiếm**
4. Xem kết quả từ hai mô hình theo cột, sắp xếp từ giống nhất đến ít giống nhất

## Routes

| Route | Method | Mô tả |
|-------|--------|-------|
| `/` | GET | Trang chủ — form tải ảnh |
| `/search` | POST | Nhận ảnh + k, trả về kết quả |
| `/dataset-image?path=...` | GET | Load ảnh từ `clothes_dataset/` |

## Thiết kế hiệu năng

- Cả hai bộ chỉ mục `features.npz` và mô hình VGG19 được **load một lần khi khởi động**, không load lại theo từng request
- `NearestNeighbors` (brute-force, Euclidean) được fit sẵn — mỗi query chỉ cần gọi `kneighbors()`
- VGG19 forward pass được bảo vệ bằng `threading.Lock` để an toàn khi dùng CUDA

## Ghi chú kỹ thuật

- **Distance metric:** Euclidean trên vector đã L2-normalize (cả hai mô hình)
- **Index size:** VGG19 ~120 MB + Color Histogram + HOG ~246 MB ≈ **363 MB RAM**
