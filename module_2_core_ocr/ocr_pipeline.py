import os
import cv2
import json
from pathlib import Path
from typing import Dict

from shared_utils.logger import get_logger
from .config import OcrConfig
from .engines.factory import OcrEngineFactory
from .utils import auto_rotate_page
from shared_utils.models import OrientationStatus

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class OcrPipeline:
    def __init__(self, active_engine_name: str = "paddle_vietocr", config: OcrConfig | None = None):
        self._config = config or OcrConfig()
        # Gọi Factory để khởi tạo Động cơ đã chọn
        self.engine = OcrEngineFactory.get_engine(active_engine_name, self._config)
        
    def process_folder(self, input_dir: str | Path) -> Dict:
        """
        Quét thư mục trung gian, đọc Siêu dữ liệu từ Module 1, 
        xoay ảnh (nếu cần) và bàn giao cho Động cơ OCR đọc chữ.
        """
        input_path = Path(input_dir)
        metadata_file = input_path / "m1_summary.json"
        
        # 1. NẠP SIÊU DỮ LIỆU TỪ MODULE 1
        m1_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                m1_metadata = json.load(f)
            logger.info(f"Đã nạp siêu dữ liệu (Metadata) cho {len(m1_metadata)} file.")
        else:
            logger.warning("Không tìm thấy file m1_summary.json. Sẽ chạy OCR mù (không xoay).")

        results = {}
        
        # 2. VÒNG LẶP HÀNG LOẠT (BATCH PROCESSING)
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
                
            filename = file_path.name
            logger.info(f"Đang xử lý OCR: {filename}")
            
            image = cv2.imread(str(file_path))
            if image is None:
                continue

            # Lấy cờ chẩn đoán hướng từ file JSON (mặc định là LIKELY_CORRECT nếu không có)
            file_meta = m1_metadata.get(filename, {})
            orient_status_str = file_meta.get("orientation_status", "LIKELY_CORRECT")
            
            # Chuyển chuỗi về Enum để an toàn kiểu dữ liệu
            try:
                orient_status = OrientationStatus(orient_status_str)
            except ValueError:
                orient_status = OrientationStatus.LIKELY_CORRECT
                
            # 3. KÍCH HOẠT LỌC HÌNH HỌC (XOAY 3 MIỀN)
            # Hàm này sẽ mượn tạm Detector của Plugin để bỏ phiếu nếu plugin có hỗ trợ
            rotated_image, angle = auto_rotate_page(
                image, 
                orient_status, 
                getattr(self.engine, 'detector', None), 
                getattr(self.engine, 'classifier', None)
            )
            
            # 4. GIAO CHO ĐỘNG CƠ OCR ĐỌC CHỮ (1 TRÁCH NHIỆM DUY NHẤT)
            ocr_result = self.engine.process_image(rotated_image)
            
            # Lưu tạm vào dict bộ nhớ
            results[filename] = {
                "result": ocr_result,
                "rotated_image": rotated_image
            }
            
        return results