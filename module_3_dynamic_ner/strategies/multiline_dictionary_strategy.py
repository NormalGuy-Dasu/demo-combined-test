import re
import json
from pathlib import Path
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class MultilineDictionaryStrategy(BaseStrategy):
    def __init__(self):
        # Đóng gói bộ nhớ đệm (Cache) ngay tại đây
        self._typo_caches = {}

    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        raw_keywords = rule.get("master_data", [])
        typo_mapping_path = rule.get("typo_mapping", "") 
        
        if not isinstance(raw_keywords, list) or not raw_keywords:
            if logger: logger("   -> [WARNING] LỖI: Danh sách từ điển rỗng hoặc chưa nạp.")
            return None
            
        # Sắp xếp từ dài đến ngắn để ưu tiên match từ khóa dài nhất
        raw_keywords.sort(key=len, reverse=True)
        aggregated_text = " ".join([line.text for line in working_lines])
        
        # --- LỚP LÀM SẠCH DỮ LIỆU (PRE-PROCESSING) BẰNG CACHE ---
        if typo_mapping_path:
            # Đường dẫn từ module_3_dynamic_ner/strategies lên thư mục gốc
            typo_path = Path(__file__).resolve().parents[1] / "configs" / typo_mapping_path
            
            if typo_path.exists():
                if typo_path not in self._typo_caches:
                    try:
                        with open(typo_path, 'r', encoding='utf-8') as f:
                            self._typo_caches[typo_path] = json.load(f)
                        if logger: logger(f"   -> [Cache] Đã nạp rules sửa lỗi OCR từ '{typo_path.name}' vào RAM.")
                    except Exception as e:
                        if logger: logger(f"   -> [Lỗi] Không đọc được file sửa lỗi: {e}")
                
                if typo_path in self._typo_caches:
                    typo_dict = self._typo_caches[typo_path]
                    clean_text = aggregated_text.upper() 
                    is_cleaned = False
                    
                    for err_word, correct_word in typo_dict.items():
                        if err_word in clean_text:
                            clean_text = clean_text.replace(err_word, correct_word)
                            if logger: logger(f"   -> [Pre-process] Đã tự động sửa lỗi OCR: '{err_word}' -> '{correct_word}'")
                            is_cleaned = True
                            
                    if is_cleaned:
                        aggregated_text = clean_text
            else:
                if logger: logger(f"   -> [WARNING] Không tìm thấy file Typo tại: {typo_path}")
        
        if logger: logger(f"   -> [Deep Trace] Chuỗi mồi nhử: '{aggregated_text}'")
        
        # --- RÀ SOÁT REGEX VỚI NEO NHẢY CÓC '|' ---
        for raw_kw in raw_keywords:
            parts = [p.strip() for p in raw_kw.split('|')]
            normalized_value = " ".join(parts)
            escaped_parts = [re.escape(p) for p in parts]
            regex_pattern = r".*?".join(escaped_parts)
            
            if re.search(regex_pattern, aggregated_text, re.IGNORECASE):
                if logger: logger(f"   -> [Deep Trace] TRÚNG ĐÍCH! Từ khóa gốc: '{raw_kw}'")
                return normalized_value
                
        if logger: logger("   -> [Deep Trace] THẤT BẠI. Không tìm thấy từ khóa nào khớp trong chuỗi gộp.")
        return None