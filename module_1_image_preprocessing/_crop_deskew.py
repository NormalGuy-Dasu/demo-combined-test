from __future__ import annotations

import cv2
import numpy as np
from rembg import remove, new_session

from .config import CropDeskewConfig
from shared_utils.logger import get_logger

logger = get_logger(__name__)

# =========================================================================
# KHỞI TẠO AI MODEL (Tải 1 lần duy nhất vào RAM)
# Dùng u2netp (phiên bản siêu nhẹ, chỉ ~4MB) chuyên tách vật thể nổi bật
# =========================================================================
logger.info("Đang nạp AI Document Scanner (U2-Net)...")
_ai_session = new_session("u2netp")

def detect_and_crop(
    image: np.ndarray, cfg: CropDeskewConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Phiên bản nâng cấp bằng AI Deep Learning (Salient Object Detection).
    """
    debug_img = image.copy()

    if not cfg.enabled:
        return image, 0.0, debug_img

    # =====================================================================
    # 1. THU NHỎ ẢNH ĐỂ AI XỬ LÝ SIÊU TỐC
    # =====================================================================
    target_height = 500.0
    ratio = image.shape[0] / target_height
    new_width = int(image.shape[1] / ratio)
    resized_image = cv2.resize(image, (new_width, int(target_height)))

    # =====================================================================
    # 2. DÙNG AI TÁCH NỀN (Nhận diện tờ giấy)
    # =====================================================================
    rgb_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Lấy mặt nạ xác suất (Xám mờ đến Trắng)
    mask = remove(rgb_img, session=_ai_session, only_mask=True)

    # 🛑 LỚP PHÒNG THỦ 1: MÁY CHÉM (Confidence Thresholding)
    # Lọc bỏ các vùng xám (do bóng râm/phản chiếu), chỉ lấy điểm AI chắc chắn > 80% (225/255)
    _, mask = cv2.threshold(mask, 225, 255, cv2.THRESH_BINARY)

    # 🛑 LỚP PHÒNG THỦ 2: KÉO CẮT (Morphological Opening)
    # Xóa các hạt rác nhỏ và cắt đứt các "cây cầu" mỏng nối rác với tờ giấy
    kernel_open = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Làm mượt viền tờ giấy để điểm cực đại không bị gai góc
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # =====================================================================
    # 3. TÌM 4 GÓC TỪ MẶT NẠ CỦA AI (Xử lý giấy cong, nếp gấp)
    # =====================================================================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("AI không tìm thấy tờ giấy nào. Trả về ảnh gốc.")
        return image, 0.0, debug_img

    # Lấy mảng trắng to nhất (Chắc chắn là tờ giấy)
    c = max(contours, key=cv2.contourArea)

    # BỎ HÀM approxPolyDP CŨ. 
    # Sử dụng Toán học để trích xuất 4 điểm cực đại (Góc) của khối trắng.
    pts = c.reshape(-1, 2)
    document_corners = np.zeros((4, 2), dtype=np.float32)

    # Tính tổng (x + y). 
    # Điểm có tổng nhỏ nhất là góc Trên-Trái, lớn nhất là Dưới-Phải
    s = pts.sum(axis=1)
    document_corners[0] = pts[np.argmin(s)] # Top-Left
    document_corners[2] = pts[np.argmax(s)] # Bottom-Right

    # Tính hiệu (y - x). 
    # Điểm có hiệu nhỏ nhất là Trên-Phải, lớn nhất là Dưới-Trái
    diff = np.diff(pts, axis=1)
    document_corners[1] = pts[np.argmin(diff)] # Top-Right
    document_corners[3] = pts[np.argmax(diff)] # Bottom-Left

    # Vẽ khung xanh lá cây bọc 4 góc vừa tìm được lên ảnh debug
    corners_original = document_corners * ratio
    cv2.polylines(debug_img, [corners_original.astype(np.int32)], True, (0, 255, 0), 5)

    # =====================================================================
    # 4. NHÂN TỶ LỆ LÊN ẢNH GỐC, TÍNH GÓC & CẮT PHỐI CẢNH
    # =====================================================================
    
    # Tính góc nghiêng vật lý ban đầu dựa trên cạnh trên cùng (Top-Left -> Top-Right)
    tl, tr = corners_original[0], corners_original[1]
    delta_x = tr[0] - tl[0]
    delta_y = tr[1] - tl[1]
    skew_angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Cắt và nắn thẳng
    warped = _perspective_transform(image, corners_original, padding=10)

    # Trả về góc nghiêng vật lý thực sự
    return warped, float(skew_angle), debug_img


def _order_corner_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _perspective_transform(
    image: np.ndarray, corners: np.ndarray, padding: int
) -> np.ndarray:
    rect = _order_corner_points(corners)
    tl, tr, br, bl = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    p = padding
    dst = np.array(
        [[p, p], [width + p, p], [width + p, height + p], [p, height + p]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width + 2 * p, height + 2 * p))