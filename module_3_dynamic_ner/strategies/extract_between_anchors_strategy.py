from rapidfuzz import fuzz
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class ExtractBetweenAnchorsStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        start_anchor = rule['start_anchor']
        end_anchor = rule['end_anchor']
        strict_start = rule.get('strict_start', False)
        strict_end = rule.get('strict_end', False)
        threshold = rule.get('match_threshold', 85)
        
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(working_lines):
            # Tìm mỏ neo bắt đầu
            if start_idx == -1:
                start_scorer = fuzz.ratio if strict_start else fuzz.partial_ratio
                start_score = start_scorer(start_anchor.lower(), line.text.lower())
                
                if start_score > threshold:
                    start_idx = i
                    if logger: logger(f"EVENT=ANCHOR_MATCH | METHOD=extract_between | TYPE=start | ANCHOR='{start_anchor}' | SCORE={start_score}")
                    continue
                    
            # Tìm mỏ neo kết thúc
            if start_idx != -1 and end_idx == -1:
                end_scorer = fuzz.ratio if strict_end else fuzz.partial_ratio
                end_score = end_scorer(end_anchor.lower(), line.text.lower())
                
                if end_score > threshold:
                    end_idx = i
                    if logger: logger(f"EVENT=ANCHOR_MATCH | METHOD=extract_between | TYPE=end | ANCHOR='{end_anchor}' | SCORE={end_score}")
                    break
                    
        # Trích xuất dữ liệu kẹp giữa
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted_lines = working_lines[start_idx + 1 : end_idx]
            if extracted_lines:
                return " ".join([l.text for l in extracted_lines]).strip()
            else:
                if logger: logger("EVENT=EXTRACTION_FAILED | METHOD=extract_between | REASON=empty_between_anchors")
                return None
        else:
            if logger: logger(f"EVENT=EXTRACTION_FAILED | METHOD=extract_between | START_IDX={start_idx} | END_IDX={end_idx}")
            return None