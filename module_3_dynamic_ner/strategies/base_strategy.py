from abc import ABC, abstractmethod
from typing import Any
from rapidfuzz import fuzz
from .payload import ExtractionPayload

class BaseStrategy(ABC):
    def execute(self, payload: ExtractionPayload, logger=None) -> Any:
        rule = payload.rule_config
        scan_direction = rule.get('scan_direction', 'top_down')
        scan_limit = rule.get('scan_limit', None)
        skip_lines = rule.get('skip_lines', 0)
        
        # [NEW CONFIGS]
        skip_to_anchor = rule.get('skip_to_anchor')
        # Cầu chì an toàn: Nếu JSON quên cấu hình, mặc định chỉ quét 15 dòng đầu
        anchor_scan_limit = rule.get('anchor_scan_limit', 15) 
        
        # 1. Bắt đầu Tiền xử lý dữ liệu không gian
        working_lines = list(payload.all_lines)
        
        if scan_direction == 'bottom_up':
            working_lines.reverse()
            if logger: logger(f"   -> [Strategy] Đã lật mảng OCR: Quét từ Dưới lên Trên.")
            
        if skip_lines > 0:
            working_lines = working_lines[skip_lines:]
            
        # [UPGRADE 2.0] Logic Skip bằng Mỏ neo (Hỗ trợ Mảng + Ưu tiên Line-First + Giới hạn dòng)
        if skip_to_anchor:
            # Đồng nhất dữ liệu: Biến string đơn thành List để xử lý chung 1 luồng
            anchors = [skip_to_anchor] if isinstance(skip_to_anchor, str) else skip_to_anchor
            anchor_found_idx = -1
            
            # Vùng an toàn: Chỉ quét trong giới hạn anchor_scan_limit
            scan_pool = working_lines[:anchor_scan_limit]
            
            # LINE-FIRST: Quét từng dòng trước
            for idx, line in enumerate(scan_pool):
                line_text_lower = line.text.lower()
                
                # Tại mỗi dòng, đọ với toàn bộ mỏ neo trong danh sách
                for anchor in anchors:
                    if fuzz.partial_ratio(anchor.lower(), line_text_lower) > 85:
                        anchor_found_idx = idx
                        if logger: logger(f"   -> [Strategy] BẮT TRÚNG MỎ NEO '{anchor}' tại dòng số {idx}.")
                        break # Phá vòng lặp mỏ neo
                
                if anchor_found_idx != -1:
                    break # Phá luôn vòng lặp dòng
            
            # Thực thi cắt mảng nếu tìm thấy
            if anchor_found_idx != -1:
                working_lines = working_lines[anchor_found_idx:]
            else:
                if logger: logger(f"   -> [Strategy] CẢNH BÁO: Không tìm thấy bất kỳ mỏ neo nào trong {anchor_scan_limit} dòng quét. Giữ nguyên mảng.")

        # Thực thi Scan Limit SAU KHI đã cắt dòng (bằng skip_lines hoặc skip_to_anchor)
        if scan_limit and isinstance(scan_limit, int) and scan_limit > 0:
            working_lines = working_lines[:scan_limit]
            
        # 2. Bàn giao mảng đã xử lý cho Class Con (Hàm nội bộ)
        return self._do_extract(working_lines, payload, logger)
        
    @abstractmethod
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None) -> Any:
        pass