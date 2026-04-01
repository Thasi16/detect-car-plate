import cv2
from ultralytics import YOLO
from config import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, CHAR_MODEL_PATH, TEST_IMAGE_PATH, VEHICLE_CLASSES
from utils import process_characters

def main():
    print("⏳ Đang tải các mô hình YOLO...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    char_model = YOLO(CHAR_MODEL_PATH)

    print(f"📸 Đang đọc ảnh từ: {TEST_IMAGE_PATH}")
    img = cv2.imread(str(TEST_IMAGE_PATH))
    
    if img is None:
        print("Lỗi: Không thể đọc được ảnh. Vui lòng kiểm tra lại đường dẫn!")
        return

    # 1. Phát hiện xe
    vehicle_results = vehicle_model(img, verbose=False)
    plates_found = []

    for r in vehicle_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            
            # Chỉ xử lý nếu đối tượng là xe
            if cls in VEHICLE_CLASSES:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle = img[y1:y2, x1:x2]

                if vehicle.size == 0:
                    continue

                # 2. Phát hiện biển số bên trong vùng chứa xe
                plate_results = plate_model(vehicle, verbose=False)

                for pr in plate_results:
                    for pbox in pr.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        plate = vehicle[py1:py2, px1:px2]

                        if plate.size == 0:
                            continue
                        
                        # Phóng to biển số 2 lần giúp YOLO đọc chữ rõ hơn
                        plate_resized = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        # 3. Đọc từng ký tự trên biển số
                        char_results = char_model(plate_resized, verbose=False)
                        
                        # Sắp xếp ký tự thành chuỗi hoàn chỉnh
                        text = process_characters(char_results)

                        if len(text) >= 4:  # Bỏ qua nếu đọc ra quá ít chữ (biển lỗi/nhiễu)
                            plates_found.append(text)

                        # Vẽ khung và text kết quả lên ảnh gốc
                        cv2.rectangle(img, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 255, 0), 2)
                        cv2.putText(img, text, (x1+px1, y1+py1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(" Các biển số nhận diện được:", plates_found)
    cv2.imshow("License Plate Recognition Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
