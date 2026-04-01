from config import CHAR_CLASSES

def process_characters(char_results):
    """
    Nhận kết quả từ YOLO char_model, phân loại biển 1 dòng/2 dòng,
    và sắp xếp ký tự từ trái qua phải, trên xuống dưới.
    """
    boxes_data = []
    
    for r in char_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.3:  # Bỏ qua các nhiễu rác (độ tin cậy thấp)
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            char = CHAR_CLASSES[cls_id]
            
            # Tính tâm tọa độ và chiều cao của chữ
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            h = y2 - y1 
            
            boxes_data.append({"char": char, "cx": cx, "cy": cy, "h": h})
            
    if not boxes_data:
        return ""

    # Sắp xếp tạm theo chiều dọc (từ trên xuống)
    boxes_data.sort(key=lambda b: b["cy"])
    
    avg_h = sum(b["h"] for b in boxes_data) / len(boxes_data)
    line_1, line_2 = [], []
    ref_cy = boxes_data[0]["cy"]  # Tâm Y của chữ cao nhất
    
    # Chia thành 2 dòng
    for b in boxes_data:
        # Nếu tâm Y tụt xuống hơn 60% chiều cao chữ -> thuộc dòng 2
        if (b["cy"] - ref_cy) > (avg_h * 0.6):
            line_2.append(b)
        else:
            line_1.append(b)
            
    # Sắp xếp từ trái qua phải cho từng dòng
    line_1.sort(key=lambda b: b["cx"])
    line_2.sort(key=lambda b: b["cx"])
    
    # Ghép chuỗi
    str_line_1 = "".join([b["char"] for b in line_1])
    str_line_2 = "".join([b["char"] for b in line_2])
    
    if len(line_2) > 0:
        return f"{str_line_1}-{str_line_2}"  # VD: 29A1-12345
    else:
        return str_line_1                    # VD: 30A12345
