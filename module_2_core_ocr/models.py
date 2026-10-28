from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class BoundingBox:
    # Lưu 4 góc: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    points: List[Tuple[int, int]] 

@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox: BoundingBox
    # Hướng B: Nền tảng Siêu dữ liệu (Metadata Foundation)
    block_type: str = "text"  # Các nhãn từ YOLO: 'text', 'title', 'table', 'figure'...
    metadata: Dict = field(default_factory=dict)  # Chứa tọa độ hàng/cột hoặc mã HTML nếu là bảng

@dataclass
class OcrResult:
    is_success: bool
    words: List[OcrWord]
    full_text: str
    page_number: int = 1  
    error_message: Optional[str] = None
    markdown_text: str = ""