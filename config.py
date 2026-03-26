from pathlib import Path

# Khởi tạo đường dẫn tương đối để dễ dàng đưa lên GitHub
BASE_DIR = Path(".")
WEIGHTS_DIR = BASE_DIR / "weights"

# 1. ĐƯỜNG DẪN MÔ HÌNH (Hãy copy các file .pt của bạn vào thư mục 'weights')
VEHICLE_MODEL_PATH = "yolov8n.pt"  # Model này tự động tải tải nếu chưa có
PLATE_MODEL_PATH = WEIGHTS_DIR / "best_plate.pt"
CHAR_MODEL_PATH = WEIGHTS_DIR / "best_char.pt"

# 2. ĐƯỜNG DẪN ẢNH TEST
TEST_IMAGE_PATH = BASE_DIR / "test_images" / "sample.png"

# 3. CÁC THÔNG SỐ KHÁC
# Lọc xe (2: Car, 3: Motorbike, 5: Bus, 7: Truck) theo chuẩn COCO của YOLOv8n
VEHICLE_CLASSES = [2, 3, 5, 7] 

# Danh sách 36 ký tự (Phải khớp chính xác với data.yaml)
CHAR_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
