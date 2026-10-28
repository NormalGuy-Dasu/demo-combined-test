import importlib
from .base_engine import BaseOcrEngine
from shared_utils.logger import get_logger

logger = get_logger(__name__)

class OcrEngineFactory:
    # DANH BẠ TƯỜNG MINH (Explicit Registry)
    _REGISTRY = {
        "paddle_vietocr": "module_2_core_ocr.engines.paddle_vietocr.PaddleVietOcrEngine",
    }

    @classmethod
    def get_engine(cls, engine_name: str, config) -> BaseOcrEngine:
        if engine_name not in cls._REGISTRY:
            logger.error(f"Động cơ '{engine_name}' không tồn tại trong danh bạ.")
            raise ValueError(f"Động cơ '{engine_name}' không hợp lệ.")
            
        module_path, class_name = cls._REGISTRY[engine_name].rsplit(".", 1)
        
        try:
            # NẠP LƯỜI (LAZY LOAD): Chỉ import thư viện khi được réo tên
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)
            logger.info(f"Đã nạp thành công Plugin OCR: {engine_name}")
            return engine_class(config)
            
        except ImportError as e:
            # CHẾT NHANH (FAIL-FAST): Văng lỗi đỏ và dừng chương trình ngay lập tức
            logger.critical(f"❌ LỖI MÔI TRƯỜNG: Không thể nạp Plugin '{engine_name}'.")
            logger.critical(f"Chi tiết thiếu thư viện: {e}")
            raise SystemExit(1)