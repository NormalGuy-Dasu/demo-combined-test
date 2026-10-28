import re
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class GlobalRegexStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        field = rule.get('field_name')
        pattern = rule.get('pattern', '')
        # Đọc tham số nối chuỗi từ JSON (Mặc định là 1 khoảng trắng nếu không cấu hình)
        group_separator = rule.get('group_separator', ' ') 
        
        # Gộp dòng thành Scoped Text từ mảng đã qua tiền xử lý
        scoped_text = "\n".join([line.text for line in working_lines])
        
        match = re.search(pattern, scoped_text)
        if match:
            # KIẾN TRÚC ĐỘNG: Xử lý linh hoạt theo số lượng nhóm bắt (groups)
            if len(match.groups()) > 1:
                # Nếu bắt được nhiều nhóm: Làm sạch từng nhóm và nối lại bằng ký tự phân cách
                extracted_groups = [g.strip() for g in match.groups() if g and g.strip()]
                result = group_separator.join(extracted_groups)
            else:
                # CƠ CHẾ CŨ: Lấy group 1 nếu có, ngược lại lấy toàn bộ match (Đảm bảo tương thích 100%)
                result = match.group(1).strip() if match.groups() else match.group(0).strip()
                
            if logger: logger(f"EVENT=EXTRACTION_SUCCESS | FIELD={field} | METHOD=global_regex | VALUE='{result}'")
            return result
        else:
            if logger: logger(f"EVENT=EXTRACTION_FAILED | FIELD={field} | METHOD=global_regex | REASON=no_match")
            return None