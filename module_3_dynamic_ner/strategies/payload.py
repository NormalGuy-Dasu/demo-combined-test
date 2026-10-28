from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExtractionPayload:
    all_lines: List[Any]          # Chứa danh sách các đối tượng LineBlock
    rule_config: Dict[str, Any]   # Cấu hình JSON của trường đang bóc tách
    dossier_context: Dict[str, Any] # Vùng nhớ dùng chung (Context)
    global_text: str              # Toàn bộ text dạng chuỗi (nếu cần)