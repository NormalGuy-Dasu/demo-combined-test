from dataclasses import dataclass, field
import numpy as np
from shared_utils.models import OrientationStatus


@dataclass
class BarcodeInfo:
    barcode_type: str
    data: str
    bbox: dict

@dataclass
class PreprocessResult:
    processed_image: np.ndarray | None
    is_blank: bool
    # 2. SỬA LẠI DÒNG NÀY (Từ bool chuyển thành Enum)
    orientation_status: OrientationStatus 
    skew_angle: float
    barcodes: list[BarcodeInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    debug_image: np.ndarray | None = None