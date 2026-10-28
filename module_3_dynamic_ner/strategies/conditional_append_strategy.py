from rapidfuzz import fuzz
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class ConditionalAppendStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        val = rule.get('base_string', '')
        
        if 'conditions' in rule:
            for cond in rule['conditions']:
                # So khớp trên toàn bộ text của văn bản (global_text)
                score = fuzz.partial_ratio(cond['keyword'].upper(), payload.global_text.upper())
                threshold = cond.get('threshold', 75)
                
                if score > threshold:
                    if logger: logger(f"EVENT=CONDITION_MET | METHOD=conditional | KEYWORD='{cond['keyword']}' | SCORE={score} | THRESHOLD={threshold} | APPENDED='{cond['append_string']}'")
                    val += cond['append_string']
                else:
                    if logger: logger(f"EVENT=CONDITION_FAILED | METHOD=conditional | KEYWORD='{cond['keyword']}' | SCORE={score} | THRESHOLD={threshold}")
                    
        return val