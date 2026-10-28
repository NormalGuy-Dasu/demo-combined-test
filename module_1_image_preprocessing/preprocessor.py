from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import os
import json

from .config import PreprocessConfig
from .models import PreprocessResult  # Đã đồng bộ Import Enum
from shared_utils.models import OrientationStatus
from ._crop_deskew import detect_and_crop
from ._detect import detect_barcodes, detect_blank_page, detect_wrong_orientation
from ._enhance import enhance_image
from shared_utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self._config = config or PreprocessConfig()

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ImagePreprocessor:
        return cls(config=PreprocessConfig.from_yaml(config_path))

    def process(self, image: str | np.ndarray, skip_crop: bool = False) -> PreprocessResult:
        source = image if not isinstance(image, np.ndarray) else "numpy array"
        logger.info(f"Processing: {source}")

        original, error_code, error_message = self._load_image(image)
        if original is None:
            logger.error(f"{error_code}: {error_message}")
            return PreprocessResult(
                processed_image=None,
                is_blank=False,
                orientation_status=OrientationStatus.LIKELY_CORRECT,
                skew_angle=0.0,
                barcodes=[],
                warnings=[],
                error_code=error_code,
                error_message=error_message,
            )

        warnings: list[str] = []

        barcodes = detect_barcodes(original, self._config.detect)

        # ==========================================
        # CƠ CHẾ OVERRIDE: YAML vs CLI
        # ==========================================
        crop_cfg = self._config.crop_deskew
        original_enabled = crop_cfg.enabled  
        
        if skip_crop:
            crop_cfg.enabled = False
            logger.info("🔘 MỆNH LỆNH CLI: Ép tắt chức năng cắt viền (Ghi đè YAML).")
            
        # ==========================================
        # BƯỚC 1: CẮT GÓC & XOAY PHẲNG
        # ==========================================
        cropped, skew_angle, debug_img = detect_and_crop(original, crop_cfg)
        crop_failed = False
        
        if crop_cfg.enabled:
            crop_area = cropped.shape[0] * cropped.shape[1]
            orig_area = original.shape[0] * original.shape[1]
            crop_ratio = crop_area / orig_area
            
            crop_failed = (crop_ratio >= 0.90) or (crop_ratio <= 0.30)
            
            if crop_failed:
                warnings.append("CROP_FAILED_PLEASE_RETAKE")
                logger.warning(f"⚠️ Lỗi Crop (Tỷ lệ: {crop_ratio:.2f}). Kích hoạt Bypass tự động.")
                cropped = original.copy() 
                skew_angle = 0.0

        crop_cfg.enabled = original_enabled

        # ==========================================
        # BƯỚC 2 & 3: TRANG TRẮNG VÀ CHẨN ĐOÁN HƯỚNG CHỮ (ĐÃ ĐỔI THÀNH ENUM)
        # ==========================================
        is_blank = detect_blank_page(cropped, self._config.detect)
        orientation_status = detect_wrong_orientation(cropped, self._config.detect)

        # LOGIC SMART ENHANCE (Áp dụng có điều kiện)
        is_bypassed = (not original_enabled) or skip_crop or crop_failed
        
        if is_bypassed:
            logger.info("🔘 Bỏ qua tẩy trắng (Enhance) để giữ nguyên độ sắc nét của bản Scan / Ảnh bypass.")
            processed = cropped 
        else:
            processed = enhance_image(cropped, self._config.enhance)
            
            if self._config.enhance.enabled:
                logger.info("🔘 Đã áp dụng thuật toán Enhance (Tẩy trắng/Tương phản).")
            else:
                logger.info("🔘 Công tắc Enhance đang TẮT. Truyền ảnh màu gốc sang Module 2.")

        logger.info(
            f"Done — blank={is_blank}, orientation={orientation_status.value}, angle={skew_angle:.1f}°, barcodes={len(barcodes)}"
        )

        # Khởi tạo gói hàng kết quả bàn giao sang Module 2
        result = PreprocessResult(
            processed_image=processed,
            is_blank=is_blank,
            orientation_status=orientation_status, # Đã đồng bộ tên trường dữ liệu mới
            skew_angle=skew_angle,
            barcodes=barcodes,
            warnings=warnings,
            error_code=None
        )
        
        result.debug_image = debug_img
        return result

    def _load_image(
        self, image: np.ndarray | str | Path
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        if isinstance(image, np.ndarray):
            if image.size == 0:
                return None, "ERR_EMPTY_ARRAY", "Input numpy array is empty"
            return image, None, None

        path = Path(image)
        if not path.exists():
            return None, "ERR_FILE_NOT_FOUND", f"File not found: {path}"

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return None, "ERR_UNSUPPORTED_FORMAT", f"Unsupported format: {path.suffix}"

        # --- FIX UNICODE: Dùng numpy đọc byte stream thay vì cv2.imread ---
        try:
            img_array = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {path}: {e}")
            img = None

        if img is None:
            return None, "ERR_CORRUPTED", f"Cannot decode image (corrupted or invalid): {path.name}"

        if img.shape[0] < 50 or img.shape[1] < 50:
            return None, "ERR_TOO_SMALL", f"Image too small: {img.shape[1]}x{img.shape[0]}px"

        return img, None, None
    
    def process_folder(self, input_dir: str | Path, output_dir: str | Path, skip_crop: bool = False) -> None:
        """
        Quét toàn bộ ảnh trong input_dir, xuất file m1_summary.json chứa thông số Enum mới.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_data = {}
        logger.info(f"Bắt đầu xử lý hàng loạt thư mục: {input_path}")

        if not input_path.exists():
            logger.error(f"Thư mục đầu vào không tồn tại: {input_path}")
            return

        for file_path in input_path.iterdir():
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            result = self.process(str(file_path), skip_crop=skip_crop)
            out_file = output_path / f"{file_path.name}"
            ext = out_file.suffix if out_file.suffix else '.png'
            
            # --- FIX UNICODE LƯU FILE: Dùng cv2.imencode và numpy tofile thay vì cv2.imwrite ---
            if result.processed_image is not None:
                success, encoded_image = cv2.imencode(ext, result.processed_image)
                if success:
                    encoded_image.tofile(str(out_file))
            elif result.is_blank:
                try:
                    # Nạp lại ảnh gốc bằng numpy và lưu lại nếu là trang trắng
                    img_array = np.fromfile(str(file_path), dtype=np.uint8)
                    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if original_img is not None:
                        success, encoded_image = cv2.imencode(ext, original_img)
                        if success:
                            encoded_image.tofile(str(out_file))
                except Exception as e:
                    logger.error(f"Lỗi khi copy trang trắng {file_path.name}: {e}")

            # Đã đồng bộ xuất nhãn dạng chuỗi chữ (value) vào file JSON bàn giao
            summary_data[file_path.name] = {
                "is_blank": result.is_blank,
                "orientation_status": result.orientation_status.value, 
                "skew_angle": result.skew_angle,
                "barcodes": [bc.data for bc in result.barcodes],
                "error": result.error_code
            }

        summary_file = output_path / "m1_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Hoàn tất! Đã lưu ảnh và file bàn giao tại: {summary_file}")