import cv2
import numpy as np
import math
from shared_utils.models import OrientationStatus

def get_rotated_crop(image, box, padding=3):
    """
    Cắt ảnh dựa trên 4 điểm tọa độ, nắn thẳng và thêm padding để không rớt dấu.
    box: Danh sách 4 tọa độ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    pts = np.array(box, dtype="float32")
    
    # Tính toán chiều rộng và chiều cao của khung chữ
    width_1 = np.linalg.norm(pts[0] - pts[1])
    width_2 = np.linalg.norm(pts[2] - pts[3])
    height_1 = np.linalg.norm(pts[0] - pts[3])
    height_2 = np.linalg.norm(pts[1] - pts[2])
    
    max_width = int(max(width_1, width_2)) + padding * 2
    max_height = int(max(height_1, height_2)) + padding * 2
    
    # Định nghĩa 4 góc của bức ảnh nhỏ sau khi cắt (đã cộng padding)
    dst_pts = np.array([
        [padding, padding],
        [max_width - padding - 1, padding],
        [max_width - padding - 1, max_height - padding - 1],
        [padding, max_height - padding - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def auto_rotate_page(image: np.ndarray, orientation_status: OrientationStatus, paddle_detector, paddle_classifier) -> tuple[np.ndarray, str]:
    """
    Hàm tiền xử lý xoay ảnh của Module 2 sử dụng chiến lược Spatial Stratified Sampling (3 miền).
    Trả về: (Ảnh đã nắn thẳng đứng, Góc đã xoay ["0", "90", "180", "270"])
    """
    # Nếu Module 1 báo an toàn -> Bỏ qua AI, không làm gì cả để tối ưu tốc độ
    if orientation_status == OrientationStatus.LIKELY_CORRECT:
        return image, "0"
        
    print(f"\n[AI-ORIENTATION] Trạng thái {orientation_status.value}: Kích hoạt AI Voting 3 miền...")
    
    # 1. Gọi Detector quét toàn trang để lấy danh sách các Bounding Box
    detector_results = paddle_detector.ocr(image, cls=False, det=True, rec=False)
    if not detector_results or not detector_results[0]:
        print("[AI-ORIENTATION] Không tìm thấy khung chữ nào để lấy mẫu. Giữ nguyên hướng.")
        return image, "0"
        
    boxes = detector_results[0]
        
    # 2. Phân bổ không gian 3 miền (Spatial Stratified Sampling)
    h, w = image.shape[:2]
    top_boxes, mid_boxes, bot_boxes = [], [], []
    
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cy = np.mean(pts[:, 1])  # Lấy tọa độ Y của tâm Box để phân miền
        
        # Tính diện tích bằng công thức hình học OpenCV
        area = cv2.contourArea(pts)
        
        box_data = {"points": box, "area": area}
        if cy < h / 3:
            top_boxes.append(box_data)
        elif cy < 2 * h / 3:
            mid_boxes.append(box_data)
        else:
            bot_boxes.append(box_data)
            
    # Lấy ra Box có diện tích LỚN NHẤT ở mỗi miền để đảm bảo chất lượng ảnh cắt
    samples = []
    if top_boxes: samples.append(max(top_boxes, key=lambda x: x["area"])["points"])
    # Ưu tiên lấy 2 mẫu ở miền giữa nếu có nhiều chữ
    if mid_boxes:
        sorted_mid = sorted(mid_boxes, key=lambda x: x["area"], reverse=True)
        samples.append(sorted_mid[0]["points"])
        if len(sorted_mid) > 1:
            samples.append(sorted_mid[1]["points"])
    if bot_boxes: samples.append(max(bot_boxes, key=lambda x: x["area"])["points"])
    
    if not samples:
        return image, "0"
    
    # 3. Cắt các vùng chữ này đưa cho AI Classifier chấm điểm và bỏ phiếu (Voting)
    votes = []
    for pts in samples:
        try:
            crop_img = get_rotated_crop(image, pts)
            # Gọi khối Classifier của Paddle để dự đoán hướng của dòng chữ
            cls_res = paddle_classifier.ocr(crop_img, cls=True, det=False, rec=False)
            if cls_res and cls_res[0]:
                angle, score = cls_res[0][0]
                votes.append(angle)
        except Exception as e:
            continue
        
    if not votes:
        return image, "0"
        
    # Lấy ra kết quả góc xoay xuất hiện nhiều nhất (Bỏ phiếu số đông)
    final_angle = max(set(votes), key=votes.count)
    print(f"[AI-ORIENTATION] Kết quả bỏ phiếu từ các miền chữ: {votes} -> Quyết định xoay: {final_angle}°")
    
    # 4. Thực thi xoay toàn bộ bức ảnh gốc bằng OpenCV hình học
    if final_angle == "90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "90"
    elif final_angle == "180":
        return cv2.rotate(image, cv2.ROTATE_180), "180"
    elif final_angle == "270":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "270"
        
    return image, "0"