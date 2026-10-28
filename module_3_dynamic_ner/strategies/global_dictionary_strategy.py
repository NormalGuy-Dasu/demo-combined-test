from rapidfuzz import process, fuzz
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class GlobalDictionaryStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        dict_list = rule.get('master_data') or rule.get('dictionary_list')
        
        # Kiểm tra an toàn: Đảm bảo dữ liệu đã được Cache nạp thành List
        if not isinstance(dict_list, list) or not dict_list:
            if logger: logger("[EXTRACTION_ABORTED] REASON=empty_or_invalid_dictionary_list")
            return None

        window_size = 4 
        if logger: logger(f"[SLIDING_WINDOW_START] TOTAL_LINES={len(working_lines)} | WINDOW_SIZE={window_size}")
        
        dict_choices = {i: item.lower() for i, item in enumerate(dict_list)}
        top_candidates = {}
        best_score = 0
        best_value = None

        # VÒNG 1: Chấm điểm Tập hợp từ (Token Set Ratio)
        for i in range(len(working_lines)):
            chunk_lines = working_lines[i : i + window_size]
            chunk_text = " ".join([l.text.strip() for l in chunk_lines])
            
            if len(chunk_text) > 5:
                chunk_text_lower = chunk_text.lower()
                match_result = process.extractOne(chunk_text_lower, dict_choices, scorer=fuzz.token_set_ratio)
                
                if match_result:
                    matched_key = match_result[2]
                    original_match = dict_list[matched_key] 
                    score = match_result[1]
                    
                    if score > best_score:
                        best_score = score
                        top_candidates = {original_match: chunk_text_lower}
                        if logger: logger(f"[SCORE_UPDATE] NEW_HIGH={score} | CANDIDATE='{original_match}'")
                    elif score == best_score and score > 0:
                        if original_match not in top_candidates:
                            top_candidates[original_match] = chunk_text_lower
                            if logger: logger(f"[SCORE_TIE] TIE_AT={score} | ADDED_CANDIDATE='{original_match}'")
        
        # VÒNG 2: Trọng tài Tie-break (Partial Ratio)
        if top_candidates:
            if len(top_candidates) == 1:
                best_value = list(top_candidates.keys())[0]
            else:
                if logger: logger(f"[TIE_BREAK_START] Có {len(top_candidates)} ứng viên cùng đạt {best_score} điểm. Đấu Vòng 2...")
                best_partial_score = -1
                
                for cand_match, cand_chunk in top_candidates.items():
                    p_score = fuzz.partial_ratio(cand_match.lower(), cand_chunk)
                    if logger: logger(f"   -> [VÒNG 2] Ứng viên: '{cand_match}' | Điểm: {p_score}")
                    if p_score > best_partial_score:
                        best_partial_score = p_score
                        best_value = cand_match
                        
                if logger: logger(f"[TIE_BREAK_RESULT] Trọng tài chốt: '{best_value}' (Điểm: {best_partial_score})")
        
        # THẨM ĐỊNH NGƯỠNG CUỐI
        threshold = rule.get('match_threshold', 85)
        if isinstance(threshold, float) and threshold < 1:
            threshold = int(threshold * 100) 
            
        if best_score >= threshold and best_value:
            if logger: logger(f"[EXTRACTION_RESULT] STATUS=success | FINAL_SCORE={best_score} | THRESHOLD={threshold} | VALUE='{best_value}'")
            return best_value
        else:
            if logger: logger(f"[EXTRACTION_RESULT] STATUS=failed | FINAL_SCORE={best_score} | THRESHOLD={threshold}")
            return None