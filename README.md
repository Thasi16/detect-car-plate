# 🚗 ALPR: Nhận Diện Biển Số Xe Tự Động Với YOLOv8 (Fine-Tuned)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Multi--GPU-blue)

Dự án này là một hệ thống Nhận diện Biển số xe tự động (ALPR - Automatic License Plate Recognition) sử dụng kiến trúc **YOLOv8**. Điểm nhấn của dự án là việc **tự fine-tune (huấn luyện tinh chỉnh) mô hình YOLO** trên tập dữ liệu tùy chỉnh cho cả hai tác vụ: Phát hiện phương tiện/biển số và Nhận diện chính xác từng ký tự (OCR).

---

## 📑 Mục lục
1. [Giới thiệu Pipeline](#-giới-thiệu-pipeline)
2. [Chi Tiết Huấn Luyện (Training Details)](#-chi-tiết-huấn-luyện-training-details)
3. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
4. [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
5. [Cách sử dụng](#-cách-sử-dụng)

---

## ⚙️ Giới thiệu Pipeline

Hệ thống hoạt động dựa trên một chuỗi (pipeline) gồm 3 mô hình YOLO nối tiếp nhau để lọc nhiễu và tăng độ chính xác tối đa:

1. **Vehicle Detection (Phát hiện xe):** Tìm và cắt ra vùng ảnh chứa phương tiện (Car, Motorbike, Bus, Truck).
2. **Plate Detection (Phát hiện biển số):** Cắt ra chính xác khung chứa biển số xe từ vùng ảnh phương tiện.
3. **Character Recognition & OCR (Đọc ký tự):** Biển số được đưa qua mô hình YOLO Char (nhận diện 36 lớp ký tự). Thuật toán tùy chỉnh sẽ phân loại biển 1 dòng/2 dòng và sắp xếp chuỗi ký tự dựa trên tọa độ trung tâm (x-center, y-center).

## 🧠 Chi Tiết Huấn Luyện (Training Details)

Dự án không sử dụng pre-trained model có sẵn cho OCR mà tiến hành fine-tuning chuyên biệt qua 2 script độc lập để tối ưu hóa hiệu suất:

### 1. Mô hình phát hiện Xe và Biển số (`finetune_yolo_car.py`)
Mô hình được huấn luyện trên Local GPU để xử lý các bối cảnh đa dạng của ảnh gốc.
* **Base Model:** `yolov8n.pt` (Phiên bản Nano ưu tiên tốc độ)
* **Image Size:** 640
* **Epochs:** 40
* **Batch Size:** 16
* **Hardware:** Local GPU (`device=0`), 3 Workers.

### 2. Mô hình đọc ký tự OCR (`finetune_yolo_char.py`)
Do bài toán phân loại 36 lớp ký tự (0-9, A-Z) yêu cầu độ phức tạp cao hơn, mô hình được cấu hình nâng cao và huấn luyện trên nền tảng Kaggle với môi trường Multi-GPU.
* **Base Model:** `yolov8s.pt` (Phiên bản Small giúp trích xuất đặc trưng chữ cái tốt hơn)
* **Image Size:** 320 (Tối ưu cho ảnh biển số đã được crop)
* **Epochs:** 200
* **Batch Size:** 32
* **Hardware:** Kaggle Multi-GPU (`device=[0, 1]`), 8 Workers.
* **Project Name:** `Plate_OCR` / `yolov8s_char`

## 📁 Cấu trúc thư mục

```text
ALPR_Project/
│
├── training_scripts/
│   ├── finetune_yolo_car.py     # Script huấn luyện nhận diện xe/biển số
│   └── finetune_yolo_char.py    # Script huấn luyện OCR (Kaggle Multi-GPU)
│
├── weights/
│   ├── best_plate.pt            # Trọng số mô hình cắt biển số
│   └── best_char.pt             # Trọng số mô hình đọc chữ (yolov8s_char)
│
├── test_images/                 
│   └── sample.png               # Ảnh dùng để chạy thử nghiệm
│
├── config.py                    # Cấu hình đường dẫn, thông số và danh sách nhãn
├── utils.py                     # Thuật toán sắp xếp ký tự thành chuỗi
├── main.py                      # Pipeline chạy nhận diện chính
│
├── requirements.txt             # Danh sách thư viện cần thiết
└── README.md                    
```

## 🚀 Hướng dẫn cài đặt

**Bước 1: Clone kho lưu trữ này về máy**
```bash
git clone [https://github.com/your-username/ALPR-YOLOv8-Finetuned.git](https://github.com/your-username/ALPR-YOLOv8-Finetuned.git)
cd ALPR-YOLOv8-Finetuned
```

**Bước 2: Cài đặt thư viện**
```bash
pip install ultralytics opencv-python numpy
```

**Bước 3: Chuẩn bị Weights**
Đặt các file trọng số `.pt` đã được huấn luyện vào thư mục `weights/`.

## 🎮 Cách sử dụng

Chạy luồng nhận diện chính trên ảnh test bằng lệnh:

```bash
python main.py
```

Hệ thống sẽ hiển thị một cửa sổ OpenCV với khung viền xanh (bounding box) và kết quả biển số (ví dụ: `29A1-12345`). Quá trình này sẽ chạy mượt mà trên cả CPU do các mô hình đã được tối ưu hóa.

