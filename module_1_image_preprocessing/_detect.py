from __future__ import annotations

import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol

from .config import DetectConfig
from .models import BarcodeInfo
from shared_utils.logger import get_logger
from shared_utils.models import OrientationStatus

logger = get_logger(__name__)

def detect_blank_page(image: np.ndarray, cfg: DetectConfig) -> bool:
    if not cfg.blank_page.enabled:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # ==========================================
    # LỚP KHIÊN 1: CẠO LỀ (LIGHT MARGIN SHAVING)
    # ==========================================
    h_orig, w_orig = gray.shape
    margin_y = int(h_orig * 0.15)
    margin_x = int(w_orig * 0.15)
    core_roi = gray[margin_y:h_orig-margin_y, margin_x:w_orig-margin_x]
    blurred = cv2.GaussianBlur(core_roi, (5, 5), 0)

    # ==========================================
    # LỚP KHIÊN 2: CHIA LÔ ĐỘNG (DYNAMIC GRID)
    # ==========================================
    h, w = blurred.shape
    
    # 1. Tính toán Lưới tự động dựa trên Tỷ lệ ảnh (Aspect Ratio)
    # Cố định chia làm 3 cột (chia theo chiều ngang)
    cols = 3 
    aspect_ratio = h / float(w)
    # Tính số hàng tỷ lệ thuận với chiều cao (đảm bảo ít nhất 3 hàng)
    rows = max(3, int(round(cols * aspect_ratio))) 
    
    total_blocks = rows * cols
    grid_h = h // rows
    grid_w = w // cols

    active_blocks = 0
    threshold = 0.01  # Ngưỡng 1% nét viền

    # 2. Quét qua lưới tự động (rows x cols)
    for i in range(rows):
        for j in range(cols):
            y_start = i * grid_h
            y_end = (i + 1) * grid_h if i < rows - 1 else h
            x_start = j * grid_w
            x_end = (j + 1) * grid_w if j < cols - 1 else w

            block = blurred[y_start:y_end, x_start:x_end]
            edges = cv2.Canny(block, 30, 150)
            
            if edges.size == 0:
                continue

            edge_ratio = float(np.count_nonzero(edges) / edges.size)
            if edge_ratio > threshold:
                active_blocks += 1

    # ==========================================
    # LOGIC QUYẾT ĐỊNH VỚI DUNG SAI ĐỘNG (DYNAMIC TOLERANCE)
    # ==========================================
    # Cho phép rác chiếm tối đa 15% tổng số ô
    # Ví dụ: Giấy A4 (3x3=9 ô) -> tolerance = 1 ô
    # Giấy hóa đơn dài (5x3=15 ô) -> tolerance = 2 ô
    tolerance = max(1, int(total_blocks * 0.15))
    
    is_blank = (active_blocks <= tolerance)

    if is_blank:
        logger.warning(f"Blank page detected. (Active blocks: {active_blocks}/{total_blocks}, tolerance: {tolerance})")
    else:
        logger.debug(f"Blank page check passed. (Active blocks: {active_blocks}/{total_blocks}, tolerance: {tolerance})")

    return is_blank

def detect_wrong_orientation(image: np.ndarray, cfg: DetectConfig) -> bool:
    """
    Detect wrong orientation using an advanced hybrid approach:
    1. Aspect Ratio check to adjust thresholds.
    2. Morphological Line Removal to prevent 'Table Traps'.
    3. Sobel Gradient Energy for fast 90/270 degree detection.
    """
    if not cfg.orientation.enabled:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # ==========================================
    # CẢI TIẾN 1: KIỂM TRA KHỔ GIẤY (ASPECT RATIO)
    # ==========================================
    is_landscape = w > h
    current_threshold = cfg.orientation.gradient_xy_threshold
    
    # Nếu là khổ ngang (ví dụ Thẻ sinh viên, CCCD), nét ngang vốn dĩ sẽ nhiều hơn
    # Ta tự động hạ chuẩn (Threshold) xuống để tránh báo động giả
    if is_landscape:
        current_threshold -= 0.15 

    # ==========================================
    # CẢI TIẾN 2: TẨY BẢNG BIỂU (LINE REMOVAL)
    # ==========================================
    # Bảng biểu chứa các vạch ngang dài làm nhiễu SobelY. Ta sẽ "xóa mờ" chúng.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Tạo một "chổi quét" hình chữ nhật ngang cực dài (khoảng 1/10 chiều rộng ảnh)
    kernel_len = max(10, w // 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    
    # Tìm các đường ngang (Morphological Close kết hợp Threshold)
    # Bỏ qua bước này nếu ảnh quá nhỏ, nhưng trên A4 nó hoạt động rất tốt để làm mờ kẻ bảng
    temp_edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
    horizontal_lines = cv2.morphologyEx(temp_edges, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Cộng các đường ngang trắng này vào nền mờ (Làm cho vạch đen của bảng biến mất)
    cleaned_blurred = cv2.add(blurred, horizontal_lines)

    # ==========================================
    # TÍNH TOÁN NĂNG LƯỢNG SOBEL 
    # ==========================================
    energy_x = float(np.mean(np.abs(cv2.Sobel(cleaned_blurred, cv2.CV_64F, 1, 0, ksize=3))))
    energy_y = float(np.mean(np.abs(cv2.Sobel(cleaned_blurred, cv2.CV_64F, 0, 1, ksize=3))))
    ratio = energy_x / (energy_y + 1e-9)

    # BỘ LỌC NHANH (FAST HEURISTIC) 3 TRẠNG THÁI
    if ratio <= 1.0:
        status = OrientationStatus.LIKELY_CORRECT
    elif ratio <= 1.2:
        status = OrientationStatus.UNCERTAIN
    else:
        status = OrientationStatus.LIKELY_ROTATED

    logger.debug(f"Orientation check: X/Y={ratio:.3f}, status={status.value}")
    return status

def detect_barcodes(image: np.ndarray, cfg: DetectConfig) -> list[BarcodeInfo]:
    if not cfg.barcode.enabled:
        return []

    results: list[BarcodeInfo] = []
    seen_data = set() # Bộ lọc chống trùng lặp nếu tìm thấy 1 mã nhiều lần

    # 1. Chuyển ảnh gốc sang xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 2. Tạo phiên bản: Tăng cường độ tương phản (CLAHE) - Cứu tinh cho ảnh bóng râm, thiếu sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 3. Tạo phiên bản: Trắng Đen tuyệt đối (Otsu Threshold)
    _, binary = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Gói tất cả các phiên bản vào một mảng để đi "đánh bắt"
    versions_to_try = [
        ("Raw Gray", gray),
        ("CLAHE Enhanced", enhanced_gray),
        ("Binarized", binary)
    ]

    # 5. TẠO PHIÊN BẢN THU NHỎ NẾU ẢNH QUÁ TO (Chống nhiễu hạt)
    h, w = gray.shape
    scale = 1.0
    max_dim = 1500 # Đưa ảnh về kích thước lý tưởng cho pyzbar
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        resized_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        versions_to_try.append(("Resized Gray", resized_gray))

    # Danh sách các loại mã thông dụng trong hồ sơ, hóa đơn (Bỏ qua Databar lỗi thời)
    allowed_symbols = [
        ZBarSymbol.QRCODE, 
        ZBarSymbol.CODE128, 
        ZBarSymbol.EAN13, 
        ZBarSymbol.EAN8,
        ZBarSymbol.CODE39
    ]

    # BẮT ĐẦU QUÉT: Quét lần lượt từng phiên bản ảnh
    for name, img_version in versions_to_try:
        try:
            decoded_objects = decode(img_version, symbols=allowed_symbols)
            
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                
                # Nếu mã này đã đọc được ở phiên bản trước rồi thì bỏ qua
                if data in seen_data:
                    continue
                
                seen_data.add(data)
                btype = obj.type
                rect = obj.rect
                
                # Xử lý tọa độ Bounding Box nếu đang dùng ảnh thu nhỏ
                if name == "Resized Gray":
                    x, y = int(rect.left / scale), int(rect.top / scale)
                    bw, bh = int(rect.width / scale), int(rect.height / scale)
                else:
                    x, y, bw, bh = rect.left, rect.top, rect.width, rect.height

                results.append(BarcodeInfo(
                    barcode_type=btype,
                    data=data,
                    bbox={"x": x, "y": y, "width": bw, "height": bh},
                ))
                logger.info(f"Barcode detected ({btype}) via [{name}]: {data[:60]!r}")
            
            # EARLY EXIT: Nếu phiên bản hiện tại đã vét sạch được mã, DỪNG LẠI NGAY để tiết kiệm CPU!
            if len(results) > 0:
                logger.debug(f"Đã tìm thấy mã vạch thành công ở bộ lọc [{name}]. Dừng quét.")
                break

        except Exception as e:
            logger.warning(f"Lỗi giải mã ở bộ lọc [{name}]: {e}")
    

    # ==========================================
    # CẢI TIẾN: ĐỘNG CƠ KÉP (OPENCV FALLBACK)
    # ==========================================
    if len(results) == 0:
        logger.debug("PyZbar bất lực (Có thể do QR chứa Logo). Kích hoạt OpenCV Fallback...")
        try:
            qr_cv = cv2.QRCodeDetector()
            # Quét trên ảnh gốc xám
            ok, decoded_info, points, _ = qr_cv.detectAndDecodeMulti(gray)
            
            if ok and decoded_info is not None:
                for info, pts in zip(decoded_info, points):
                    if not info: # Bỏ qua nếu tìm thấy hình vuông nhưng không đọc được chữ
                        continue
                        
                    # Lọc trùng lặp
                    if info in seen_data:
                        continue
                    seen_data.add(info)
                    
                    pts = pts.astype(int)
                    x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
                    w, h = int(pts[:, 0].max()) - x, int(pts[:, 1].max()) - y
                    
                    results.append(BarcodeInfo(
                        barcode_type="QR_CODE",
                        data=info,
                        bbox={"x": x, "y": y, "width": w, "height": h},
                    ))
                    logger.info(f"Barcode detected (QR_CODE) via [OpenCV Fallback]: {info[:60]!r}")
        except Exception as e:
            logger.warning(f"Lỗi OpenCV Fallback: {e}")

    logger.debug(f"Barcode scan complete: {len(results)} found")
    return results