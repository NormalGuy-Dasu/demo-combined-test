# File: orchestrator/engine/ocr_utils.py
from .logger import log_xray

class LineBlock:
    """Class đại diện cho một dòng văn bản đã được gom tọa độ"""
    def __init__(self, text: str, y_min: float, y_max: float, page_num: int):
        self.text = text
        self.y_min = y_min
        self.y_max = y_max
        self.page_num = page_num

def cluster_y_overlap(ocr_words: list, page_num: int) -> list:
    """Hàm tiện ích gom dòng dùng chung cho cả Tầng 1 (Router) và Tầng 2 (Session)"""
    if not ocr_words: return []
    boxes = []
    for w in ocr_words:
        pts = w['bbox']['points']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        boxes.append({'text': w['text'], 'x_min': min(xs), 'x_max': max(xs), 'y_min': min(ys), 'y_max': max(ys)})
        
    boxes.sort(key=lambda b: b['y_min'])
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        line_y_min = min(b['y_min'] for b in current_line)
        line_y_max = max(b['y_max'] for b in current_line)
        
        overlap = max(0, min(box['y_max'], line_y_max) - max(box['y_min'], line_y_min))
        min_height = min(box['y_max'] - box['y_min'], line_y_max - line_y_min)
        
        floor_threshold = 12
        safe_box_height = max(box['y_max'] - box['y_min'], floor_threshold)
        safe_line_height = max(line_y_max - line_y_min, floor_threshold)
        
        max_h = max(safe_box_height, safe_line_height)
        min_h = min(safe_box_height, safe_line_height)
        
        height_ratio = (max_h / min_h) if min_h > 0 else 999
        is_height_valid = height_ratio <= 2.5
        
        if min_height > 0 and (overlap / min_height) > 0.3 and is_height_valid:
            current_line.append(box)
        else:
            if min_height > 0 and (overlap / min_height) > 0.3 and not is_height_valid:
                current_text = " ".join([b['text'] for b in current_line])
                log_xray(f"[SHIELD_ACTIVATED] Từ chối gộp '{box['text']}' vào '{current_text}'. Tỷ lệ lệch: {height_ratio:.2f} > 2.5")
            
            current_line.sort(key=lambda b: b['x_min'])
            lines.append(LineBlock(" ".join([b['text'] for b in current_line]), line_y_min, line_y_max, page_num))
            current_line = [box]
    
    if current_line:
        current_line.sort(key=lambda b: b['x_min'])
        lines.append(LineBlock(" ".join([b['text'] for b in current_line]), min(b['y_min'] for b in current_line), max(b['y_max'] for b in current_line), page_num))
    return lines