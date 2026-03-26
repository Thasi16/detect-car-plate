# detect-car-plate

Được, mình sẽ viết lại README với **nhấn mạnh phần fine-tune YOLOv8** để detect **biển số** và **ký tự trên biển**, vì đây là phần quan trọng để dự án hoạt động chính xác. Dưới đây là bản chi tiết:

---

# Vehicle & License Plate OCR with YOLOv8

## Mô tả dự án

Dự án này phát hiện **xe cộ** và **biển số** trong ảnh, sau đó đọc **ký tự trên biển số**. Quy trình gồm:

1. **Vehicle detection** – phát hiện các loại xe (Car, Motorbike, Bus, Truck).
2. **License plate detection** – phát hiện khung biển số. **Mô hình này cần fine-tune YOLOv8 trên dataset biển số riêng.**
3. **Character recognition** – đọc ký tự trên biển số. **Mô hình này cũng cần fine-tune YOLOv8 trên dataset ký tự biển số.**

Kết quả trả về là chuỗi biển số và hình ảnh đánh dấu khung với chữ đã đọc.

---

## Yêu cầu

* Python ≥ 3.8
* OpenCV
* NumPy
* ultralytics (YOLOv8)

Cài đặt:

```bash
pip install opencv-python numpy ultralytics
```

---

## Cấu trúc thư mục gợi ý

```text
project_plate_ocr/
├─ datasets/
│  ├─ plate/              # Dataset biển số (ảnh + labels YOLOv8)
│  └─ char/               # Dataset ký tự trên biển số (ảnh + labels YOLOv8)
├─ models/
│  ├─ yolov8n.pt          # Mô hình detect xe chuẩn sẵn
│  ├─ best_plate.pt       # Mô hình detect biển số (fine-tune)
│  └─ best_char.pt        # Mô hình đọc ký tự (fine-tune)
├─ images/
│  └─ test_image.png
├─ main.py
└─ README.md
```

---

## Fine-tune YOLOv8

### 1. Detect biển số

* Chuẩn bị dataset gồm ảnh xe và bounding box biển số theo **YOLO format**.
* Tạo file `data.yaml`:

```yaml
train: datasets/plate/train/images
val: datasets/plate/val/images
nc: 1
names: ['plate']
```

* Huấn luyện:

```bash
yolo detect train model=yolov8n.pt data=datasets/plate/data.yaml epochs=100 imgsz=640
```

* Kết quả: `best_plate.pt`

---

### 2. Detect ký tự trên biển số

* Dataset gồm **ảnh biển số nhỏ** và bounding box ký tự với nhãn `0-9` + `A-Z`.
* Tạo file `data.yaml`:

```yaml
train: datasets/char/train/images
val: datasets/char/val/images
nc: 36
names: ['0','1','2',...,'Z']
```

* Huấn luyện:

```bash
yolo detect train model=yolov8s.pt data=datasets/char/data.yaml epochs=100 imgsz=640
```

* Kết quả: `best_char.pt`

> Lưu ý: việc fine-tune là bắt buộc nếu muốn mô hình nhận diện chính xác biển số và ký tự, vì dataset chuẩn YOLOv8 không có dữ liệu biển số Việt Nam hoặc ký tự đặc thù.

---

## Cách sử dụng pipeline

1. Cập nhật đường dẫn mô hình và ảnh test trong `main.py`:

```python
vehicle_model = YOLO("models/yolov8n.pt")
plate_model = YOLO("models/best_plate.pt")
char_model = YOLO("models/best_char.pt")
image_path = "images/test_image.png"
```

2. Chạy script:

```bash
python main.py
```

* Pipeline sẽ: detect xe → detect biển số → phóng to → detect ký tự → sắp xếp chữ → hiển thị & in kết quả.

3. Output:

* **Terminal:** danh sách biển số đọc được.
* **Hình ảnh:** hiển thị khung biển số và ký tự đọc được.

---

