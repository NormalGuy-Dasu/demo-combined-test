from __future__ import annotations

import cv2
import numpy as np

from .config import EnhanceConfig
from shared_utils.logger import get_logger

logger = get_logger(__name__)

def enhance_image(image: np.ndarray, cfg: EnhanceConfig) -> np.ndarray:
    """
    Pipeline Chuẩn:
    - Nếu Binarize (cho OCR): Grayscale → Pre-blur (San phẳng giấy) → Adaptive Threshold → Post-MedianBlur (Quét rác).
    - Nếu KHÔNG Binarize (Cho người đọc): Grayscale → CLAHE → Post-MedianBlur.
    """
    if not cfg.enabled:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # 1. PRE-BLUR: Dùng GaussianBlur để san phẳng các vân giấy và nhiễu camera
    if cfg.denoise.enabled:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    result = gray

    # 2. RẼ NHÁNH XỬ LÝ
    if cfg.binarize.enabled:
        # Nếu đã Binarize thì KHÔNG dùng CLAHE để tránh khuếch đại rác
        result = cv2.adaptiveThreshold(
            result,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            cfg.binarize.block_size,
            cfg.binarize.c_constant,
        )
    else:
        # Chỉ dùng CLAHE khi muốn giữ ảnh xám (không binarize) để mắt người dễ đọc
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe.clip_limit,
            tileGridSize=tuple(cfg.clahe.tile_grid_size),
        )
        result = clahe.apply(result)

    # 3. POST-DENOISE: Quét sạch các chấm đen li ti (muối tiêu) cuối cùng
    if cfg.denoise.enabled:
        result = cv2.medianBlur(result, cfg.denoise.kernel_size)

    return result