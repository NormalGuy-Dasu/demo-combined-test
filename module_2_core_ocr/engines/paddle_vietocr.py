import numpy as np
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from .base_engine import BaseOcrEngine
from module_2_core_ocr.models import OcrResult, OcrWord, BoundingBox
from module_2_core_ocr.utils import get_rotated_crop
from shared_utils.logger import get_logger

logger = get_logger(__name__)

class PaddleVietOcrEngine(BaseOcrEngine):
    def __init__(self, config):
        self.config = config
        
        # 1. KHỞI TẠO PADDLE OCR (CHỈ DÙNG DETECTION ĐỂ TÌM KHUNG)
        logger.info("[ENGINE] Đang khởi tạo PaddleOCR (Detection)...")
        self.detector = PaddleOCR(
            use_angle_cls=True,  # Đã tắt vì Module 2 tự xoay toàn trang rồi
            lang=self.config.lang, # Lấy từ config gốc
            rec=False,
            show_log=False,
            # Lấy thông số từ class lồng nhau PaddleConfig
            det_limit_side_len=self.config.paddle.det_limit_side_len, 
            det_db_thresh=self.config.paddle.det_db_thresh,
            det_db_box_thresh=self.config.paddle.det_db_box_thresh
        )
        
        # Lấy model Classifier đã được bọc sẵn bên trong detector để dùng chung
        self.classifier = self.detector
        
        # 2. KHỞI TẠO VIETOCR (CHỈ DÙNG RECOGNITION ĐỂ ĐỌC CHỮ)
        logger.info("[ENGINE] Đang khởi tạo VietOCR (Recognition)...")
        vgg_config = Cfg.load_config_from_name(self.config.vietocr.config_name)
        vgg_config['cnn']['pretrained'] = False
        vgg_config['device'] = 'cuda:0' if self.config.use_gpu else 'cpu'
        
        # 🌟 VÁ LỖI ÉP XUNG: Ép cấu hình Dataset DataLoader phải nhận batch_size khổng lồ
        vgg_config['predictor']['batch_size'] = self.config.vietocr.batch_size
        
        self.recognizer = Predictor(vgg_config)

    def _sort_and_group_boxes(self, boxes: list) -> list:
        """
        Thuật toán gom dòng với Ngưỡng Động (Dynamic Ratio Threshold).
        Sắp xếp Từ Trên Xuống Dưới, Từ Trái Sang Phải.
        """
        if not boxes:
            return []

        # Tính toán các thông số hình học cho từng Box
        box_data = []
        for box in boxes:
            pts = np.array(box)
            y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
            x_min = np.min(pts[:, 0])
            center_y = (y_min + y_max) / 2
            height = y_max - y_min
            box_data.append({"box": box, "center_y": center_y, "x_min": x_min, "height": height})

        # 1. Tính Ngưỡng Động (Dựa trên trung vị chiều cao của các Box)
        median_height = np.median([b["height"] for b in box_data])
        
        # Lấy tỷ lệ dung sai từ class lồng nhau HeuristicSortingConfig
        y_tolerance = median_height * self.config.heuristic.y_tolerance_ratio 

        # 2. Sắp xếp sơ bộ theo trục Y
        box_data.sort(key=lambda b: b["center_y"])

        # 3. Gom Dòng (Line Grouping)
        lines = []
        current_line = [box_data[0]]

        for current_box in box_data[1:]:
            prev_box = current_line[-1]
            # Nếu chênh lệch Y nhỏ hơn dung sai -> Cùng 1 dòng
            if abs(current_box["center_y"] - prev_box["center_y"]) <= y_tolerance:
                current_line.append(current_box)
            else:
                lines.append(current_line)
                current_line = [current_box]
        lines.append(current_line)

        # 4. Sắp xếp các Box trong cùng 1 dòng theo trục X (Trái -> Phải)
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda b: b["x_min"])
            sorted_boxes.extend([b["box"] for b in line])

        return sorted_boxes

    def process_image(self, image: np.ndarray) -> OcrResult:
        """Thực thi luồng lai: Paddle Detect -> Sort -> VietOCR Recognize"""
        try:
            # BƯỚC 1: Tìm khung chữ
            detector_results = self.detector.ocr(image, cls=False, det=True, rec=False)
            if not detector_results or not detector_results[0]:
                return OcrResult(is_success=True, words=[], full_text="")

            raw_boxes = detector_results[0]
            
            # BƯỚC 2: Sắp xếp và gom dòng
            sorted_boxes = self._sort_and_group_boxes(raw_boxes)
            
            # BƯỚC 3: Cắt ảnh hàng loạt
            pil_images = []
            for box in sorted_boxes:
                cropped = get_rotated_crop(image, box)
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(cropped_rgb))
                
            # BƯỚC 4: Đọc chữ hàng loạt bằng VietOCR (Batch Processing)
            words = []
            full_text_parts = []
            if pil_images:
                texts, probs = self.recognizer.predict_batch(pil_images, return_prob=True)
                for i, box_coords in enumerate(sorted_boxes):
                    text, prob = texts[i], probs[i]

                    # Bỏ qua nếu text rỗng, toàn dấu cách, hoặc prob là NaN
                    if not text.strip() or prob is None or np.isnan(prob):
                        continue
                    # ---------------------------------

                    words.append(OcrWord(
                        text=text,
                        confidence=prob,
                        bbox=BoundingBox(points=[(int(p[0]), int(p[1])) for p in box_coords])
                    ))
                    full_text_parts.append(text)
                    
            return OcrResult(
                is_success=True,
                words=words,
                full_text="\n".join(full_text_parts)
            )
            
        except Exception as e:
            logger.error(f"Lỗi hệ thống Plugin OCR: {e}")
            return OcrResult(is_success=False, words=[], full_text="", error_message=str(e))