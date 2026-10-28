import re
from rapidfuzz import fuzz
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class FuzzyKeyValueStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        anchor = rule.get('anchor_keyword', '')
        line_offset = rule.get('line_offset', 0)
        val_regex = rule.get('value_regex', '')
        threshold = rule.get('match_threshold', 80)
        
        best_score = 0
        matched_line_idx = -1
        anchor_len = len(anchor.strip())
        
        # Quét tìm dòng chứa mỏ neo tốt nhất
        for i, line in enumerate(working_lines):
            clean_line = line.text.strip()
            
            # Dynamic Shield: Bỏ qua dòng rác quá ngắn
            if len(clean_line) < (anchor_len * 0.3):
                continue
                
            score = fuzz.WRatio(anchor.lower(), clean_line.lower())
            if score > best_score:
                best_score = score
                matched_line_idx = i
                
        if best_score > threshold:
            target_idx = matched_line_idx + line_offset
            
            # Kiểm tra an toàn: target_idx có vượt quá độ dài mảng không
            if 0 <= target_idx < len(working_lines):
                raw_text = working_lines[target_idx].text
                if logger: logger(f"EVENT=ANCHOR_MATCH | METHOD=fuzzy_key_value | ANCHOR='{anchor}' | OFFSET={line_offset} | SCORE={best_score} | LINE='{raw_text}'")
                
                if val_regex:
                    match = re.search(val_regex, raw_text)
                    if match:
                        return match.group(1).strip() if match.groups() else match.group(0).strip()
                    else:
                        if logger: logger("EVENT=EXTRACTION_FAILED | METHOD=fuzzy_key_value | REASON=regex_shield_blocked")
                        return None
                else:
                    return raw_text.strip()
            else:
                if logger: logger("EVENT=EXTRACTION_FAILED | METHOD=fuzzy_key_value | REASON=offset_out_of_bounds")
                return None
        else:
            if logger: logger(f"EVENT=EXTRACTION_FAILED | METHOD=fuzzy_key_value | REASON=anchor_not_found | SCORE={best_score}")
            return None