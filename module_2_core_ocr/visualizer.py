import cv2
import numpy as np
from pathlib import Path
from .models import OcrResult

def draw_ocr_results(image: np.ndarray, ocr_result: OcrResult, output_path: str | Path) -> np.ndarray:
    """
    Vẽ Bounding Box kèm nhãn lên ảnh dựa trên block_type.
    - Bảng (table): Đỏ
    - Tiêu đề (title): Xanh dương
    - Văn bản thường (text): Xanh lá
    """
    annotated_img = image.copy()
    
    # Định nghĩa bảng màu (B, G, R) trong OpenCV
    COLOR_MAP = {
        "table": (0, 0, 255),    # Đỏ
        "title": (255, 0, 0),    # Xanh dương
        "text": (0, 255, 0),     # Xanh lá
        "figure": (0, 255, 255)  # Vàng
    }
    
    for word in ocr_result.words:
        block_type = word.block_type.lower()
        color = COLOR_MAP.get(block_type, (255, 0, 255)) # Mặc định màu tím nếu nhãn lạ
        
        # Lấy tọa độ
        points = np.array(word.bbox.points, dtype=np.int32)
        
        # Vẽ đa giác Bounding Box
        cv2.polylines(annotated_img, [points], isClosed=True, color=color, thickness=2)
        
        # Vẽ nhãn (Label) lên góc trên bên trái của Box
        x, y = points[0]
        label_text = f"[{block_type.upper()}] {word.confidence:.2f}"
        
        # Vẽ nền đen mờ cho chữ dễ đọc
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_img, (x, y - text_h - 5), (x + text_w, y + 2), (0, 0, 0), -1)
        
        # Ghi chữ
        cv2.putText(annotated_img, label_text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Lưu ảnh nếu có đường dẫn
    if output_path:
        cv2.imwrite(str(output_path), annotated_img)
        
    return annotated_img