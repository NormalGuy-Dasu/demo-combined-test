import re
import unicodedata
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class HybridDictionaryStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        dict_list = rule.get('master_data') or rule.get('dictionary_list')
        
        if not isinstance(dict_list, list) or not dict_list:
            if logger: logger("   -> [Lỗi Hybrid] Cấu hình sai, danh sách từ điển rỗng hoặc chưa nạp.")
            return None
            
        best_match_value = None
        
        if logger: logger(f"   -> [Debug Hybrid] Bắt đầu rà soát Regex trên {len(working_lines)} dòng (đã ép chuẩn Unicode NFC)...")
        
        for idx, line in enumerate(working_lines):
            if len(line.text.strip()) < 5:
                continue
                
            normalized_line = unicodedata.normalize('NFC', line.text)
            
            for item in dict_list:
                normalized_item = unicodedata.normalize('NFC', item)
                
                # Tạo siêu Regex
                parts = normalized_item.split()
                pattern_core = r'[\s\.]+'.join([re.escape(p) for p in parts])
                full_pattern = f"(?i)({pattern_core})"
                
                # So khớp
                if re.search(full_pattern, normalized_line):
                    best_match_value = item
                    if logger: logger(f"   -> [Debug Hybrid] BẮT TRÚNG ĐÍCH tại dòng {idx + 1}/{len(working_lines)} vùng quét!")
                    break 
                    
            if best_match_value:
                break 

        if best_match_value:
            if logger: logger(f"   -> [Debug Hybrid] Kết quả chuẩn hóa cuối cùng: '{best_match_value}'")
            return best_match_value
        else:
            if logger: logger("   -> [Debug Hybrid] THẤT BẠI. Không tìm thấy từ khóa nào khớp trong vùng quét.")
            return None