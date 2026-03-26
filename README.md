Dưới đây là một mẫu file `README.md` được thiết kế chuẩn chỉnh cho GitHub. Tôi đã nhấn mạnh rõ việc bạn tự tay **fine-tune (huấn luyện tinh chỉnh)** mô hình YOLOv8 cho cả hai tác vụ: nhận diện xe và đọc ký tự (OCR), vì đây là một điểm cộng rất lớn thể hiện kỹ năng Deep Learning của bạn.

Bạn hãy copy đoạn mã dưới đây và dán vào file `README.md` nhé:

```markdown
# 🚗 ALPR: Nhận Diện Biển Số Xe Tự Động Với YOLOv8 (Fine-Tuned)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

Dự án này là một hệ thống Nhận diện Biển số xe tự động (ALPR - Automatic License Plate Recognition) sử dụng kiến trúc **YOLOv8**. Điểm nhấn của dự án là việc **tự fine-tune (huấn luyện tinh chỉnh) mô hình YOLO** trên tập dữ liệu tùy chỉnh để tối ưu hóa khả năng phát hiện phương tiện (xe máy, ô tô) và nhận diện chính xác từng ký tự trên biển số (OCR).

---

## 📑 Mục lục
1. [Giới thiệu Pipeline](#-giới-thiệu-pipeline)
2. [Quá trình Fine-tune Mô hình](#-quá-trình-fine-tune-mô-hình)
3. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
4. [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
5. [Cách sử dụng](#-cách-sử-dụng)
6. [Kết quả dự kiến](#-kết-quả-dự-kiến)

---

## ⚙️ Giới thiệu Pipeline

Hệ thống hoạt động dựa trên một chuỗi (pipeline) gồm 3 mô hình YOLO nối tiếp nhau để lọc nhiễu và tăng độ chính xác tối đa:

1. **Vehicle Detection (Phát hiện xe):** Tìm và cắt ra vùng ảnh chứa phương tiện (Car, Motorbike, Bus, Truck), loại bỏ các bối cảnh thừa xung quanh.
2. **Plate Detection (Phát hiện biển số):** Từ vùng ảnh chiếc xe, mô hình tiếp tục tìm và cắt ra chính xác khung chứa biển số xe.
3. **Character Recognition & OCR (Đọc ký tự):** Biển số được phóng to (Resize) và đưa qua mô hình YOLO Char. Kết quả sau đó được thuật toán tùy chỉnh phân loại thành biển 1 dòng/2 dòng và sắp xếp từ trái qua phải, trên xuống dưới.

## 🧠 Quá trình Fine-tune Mô hình

Thay vì sử dụng các pre-trained model có sẵn với độ chính xác không cao cho biển số Việt Nam, dự án này đã tiến hành fine-tuning sâu trên nền tảng Kaggle:

* **Fine-tune Vehicle/Plate Model:** Huấn luyện lại YOLOv8 để model nhạy bén hơn với các góc chụp nghiêng, lóa sáng hoặc biển số bị che khuất một phần.
* **Fine-tune Character Model (36 Classes):** Mô hình được train chuyên biệt để nhận diện 36 lớp ký tự (`0-9` và `A-Z`). Việc dùng YOLO cho tác vụ OCR giúp khắc phục các nhược điểm của Tesseract hay EasyOCR khi đối mặt với biển số mờ, xước hoặc font chữ đặc thù.

## 📁 Cấu trúc thư mục

```text
ALPR_Project/
│
├── weights/
│   ├── yolov8n.pt               # Mô hình tìm xe
│   ├── best_plate.pt            # Mô hình cắt biển số
│   └── best_char.pt             # Mô hình đọc chữ (Fine-tuned 36 classes)
│
├── test_images/                 # Thư mục chứa ảnh test
│   └── sample.png
│
├── config.py                    # Cấu hình đường dẫn và danh sách nhãn
├── utils.py                     # Thuật toán sắp xếp ký tự thành chuỗi
├── main.py                      # Pipeline chạy chính (Inference)
│
├── requirements.txt             # Danh sách thư viện cần thiết
└── README.md                    # File tài liệu hướng dẫn
```

## 🚀 Hướng dẫn cài đặt

**Bước 1: Clone kho lưu trữ này về máy**
```bash
git clone [https://github.com/your-username/ALPR-YOLOv8-Finetuned.git](https://github.com/your-username/ALPR-YOLOv8-Finetuned.git)
cd ALPR-YOLOv8-Finetuned
```

**Bước 2: Cài đặt thư viện**
Khuyến nghị sử dụng môi trường ảo (Virtual Environment).
```bash
pip install ultralytics opencv-python numpy
```

**Bước 3: Chuẩn bị Weights**
Đảm bảo bạn đã tải các file `.pt` (trọng số mô hình sau khi train) và đặt chúng vào thư mục `weights/` đúng như cấu trúc ở trên.

## 🎮 Cách sử dụng

Mở terminal và chạy lệnh sau để thực hiện nhận diện trên ảnh test:

```bash
python main.py
```

Hệ thống sẽ in kết quả biển số ra cửa sổ terminal và hiển thị một cửa sổ OpenCV với khung viền (bounding box) kèm kết quả nhận diện (ví dụ: `29A1-12345`). Bấm phím bất kỳ để đóng cửa sổ.

## 📊 Kết quả dự kiến

Hệ thống xử lý tốt các trường hợp:
* Cả biển số dài (1 dòng) và biển số vuông (2 dòng).
* Phân loại logic thứ tự chữ cái dựa trên tâm tọa độ (y-center và x-center).
* Loại bỏ nhiễu rác nhờ việc thiết lập ngưỡng tin cậy (`conf > 0.3`).

---
*Dự án thể hiện việc áp dụng toàn diện từ khâu thu thập dữ liệu, fine-tune mô hình Deep Learning đến xây dựng luồng xử lý Computer Vision thực tế.*
```

