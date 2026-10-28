from pathlib import Path
import yaml
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PaddleConfig:
    det_limit_side_len: int = 1536
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6

@dataclass
class VietOcrConfig:
    config_name: str = "vgg_seq2seq"

    # 🌟 VÁ LỖI ÉP XUNG: Ép VietOCR phải xử lý một lô lớn ảnh cùng lúc
    batch_size: int = 64  # Tùy VRAM (nếu VRAM >= 8GB, có thể nâng lên 64 hoặc 128)

@dataclass
class HeuristicSortingConfig:
    y_tolerance_ratio: float = 0.5

@dataclass
class DeepDocConfig:
    yolo_layout_path: str = "../weights/layout.onnx"
    yolo_tsr_path: str = "../weights/tsr.onnx"
    vietocr_model_dir: str = "../weights"
    confidence_threshold: float = 0.5

    def validate_paths(self, base_dir: Path):
        layout_path = (base_dir / self.yolo_layout_path).resolve()
        tsr_path = (base_dir / self.yolo_tsr_path).resolve()
        vietocr_dir = (base_dir / self.vietocr_model_dir).resolve()
        
        if not layout_path.exists():
            logger.warning(f"⚠️ Không tìm thấy file Layout YOLO tại: {layout_path}")
        if not tsr_path.exists():
            logger.warning(f"⚠️ Không tìm thấy file TSR YOLO tại: {tsr_path}")
        if not vietocr_dir.exists() or not vietocr_dir.is_dir():
            logger.warning(f"⚠️ Không tìm thấy thư mục chứa VietOCR ONNX tại: {vietocr_dir}")

@dataclass
class OcrConfig:
    active_engine: str = "paddle_vietocr"
    use_gpu: bool = True
    lang: str = "vi"
    
    # Các class lồng nhau
    paddle: PaddleConfig = None
    vietocr: VietOcrConfig = None
    heuristic: HeuristicSortingConfig = None
    deepdoc: DeepDocConfig = None

    def __post_init__(self):
        # Đảm bảo luôn có object con dù không truyền vào
        if self.paddle is None: self.paddle = PaddleConfig()
        if self.vietocr is None: self.vietocr = VietOcrConfig()
        if self.heuristic is None: self.heuristic = HeuristicSortingConfig()
        if self.deepdoc is None: self.deepdoc = DeepDocConfig()

        # Kích hoạt tự động kiểm tra đường dẫn file ONNX ngay khi hệ thống khởi động
        current_dir = Path(__file__).parent
        self.deepdoc.validate_paths(base_dir=current_dir)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OcrConfig":
        """Ánh xạ thủ công (Manual Explicit Mapping) từ YAML sang Dataclass"""
        path_obj = Path(path)
        if not path_obj.exists():
            return cls() # Trả về cấu hình mặc định nếu không tìm thấy file

        with open(path_obj, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        engine_data = data.get("engine", {})
        paddle_data = data.get("paddle", {})
        vietocr_data = data.get("vietocr", {})
        heuristic_data = data.get("heuristic_sorting", {})
        deepdoc_data = data.get("deepdoc", {})

        return cls(
            active_engine=engine_data.get("active", "paddle_vietocr"),
            use_gpu=engine_data.get("use_gpu", True),
            lang=engine_data.get("lang", "vi"),
            paddle=PaddleConfig(**paddle_data),
            vietocr=VietOcrConfig(**vietocr_data),
            heuristic=HeuristicSortingConfig(**heuristic_data),
            deepdoc=DeepDocConfig(**deepdoc_data)
        )