import re
from rapidfuzz import fuzz
from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class ProximityNumberStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        anchors = rule.get('anchor_keywords', [])
        look_up_lines = rule.get('look_up_lines', 2)
        field = rule.get('field_name')
        
        raw_scan_direction = rule.get('scan_direction', 'top_down')
        raw_context_direction = rule.get('context_direction') 
        date_format_mode = rule.get('date_format_mode', 'any') 
        scan_limit = rule.get('scan_limit', None) 
        
        # [CỜ ĐIỀU HƯỚNG MỚI] - Mặc định ưu tiên quét từ dòng sát mỏ neo nhất
        line_scan_priority = rule.get('line_scan_priority', 'anchor_first')

        original_lines = payload.all_lines 
        
        if raw_context_direction is None:
            if raw_scan_direction == 'bottom_up':
                actual_scan = 'top_down' 
                actual_context = 'below' 
            else:
                actual_scan = 'top_down'
                actual_context = 'above'
        else:
            actual_scan = raw_scan_direction
            actual_context = raw_context_direction

        # Loại bỏ [a-zA-ZÀ-ỹ...] để cấm tuyệt đối chữ cái rác, chỉ cho phép số, các ký tự giống số và dấu chấm
        val_group = r'([0-9SsOoQqLlIi|BbZzđvV.]{1,4})?'
        year_group = r'([0-9SsOoQqLlIi|BbZzđvV.]{2,5})'
        
        ngay_dict = r'(?:ngày|ngay|ngäy|ngqy|ng)'
        thang_dict = r'(?:tháng|thang|thảng|thãng|thg|th|nhàng|nhang)'
        nam_dict = r'(?:năm|nam|näm|nãm|n)'
        bridge_space = r'[\s.,:\-_]*'
        
        if date_format_mode == 'text_only':
            date_pattern = r'(?:' + ngay_dict + r')?' + bridge_space + val_group + bridge_space + thang_dict + bridge_space + val_group + bridge_space + nam_dict + bridge_space + year_group
        elif date_format_mode == 'symbol_only':
            date_pattern = val_group + r'[\s]*[/.\-_\\]+[\s]*' + val_group + r'[\s]*[/.\-_\\]+[\s]*' + year_group
        else:
            bridge_1 = r'(?:' + bridge_space + thang_dict + bridge_space + r'|[\s]*[/.\-_\\]+[\s]*)'
            bridge_2 = r'(?:' + bridge_space + nam_dict + bridge_space + r'|[\s]*[/.\-_\\]+[\s]*)'
            date_pattern = r'(?:' + ngay_dict + r')?' + bridge_space + val_group + bridge_1 + val_group + bridge_2 + year_group

        lines_to_search = list(enumerate(original_lines))
        
        if actual_scan == 'bottom_up':
            lines_to_search.reverse()

        if scan_limit and isinstance(scan_limit, int) and scan_limit > 0:
            lines_to_search = lines_to_search[:scan_limit]

        for anchor in anchors:
            for i, line in lines_to_search:
                if len(line.text.strip()) >= 6:
                    score = fuzz.partial_ratio(anchor.lower(), line.text.lower())
                    
                    if score > 85: 
                        if logger: logger(f"EVENT=ANCHOR_MATCH | FIELD={field} | METHOD=proximity | ANCHOR='{anchor}' | SCORE={score}")
                        
                        if actual_context == 'below':
                            end_idx = min(len(original_lines), i + look_up_lines + 1)
                            context_slice = original_lines[i:end_idx]
                        else: 
                            start_idx = max(0, i - look_up_lines)
                            context_slice = original_lines[start_idx:i+1]
                            
                        # KIẾN TRÚC HYBRID: CHUẨN BỊ MẢNG THEO THỨ TỰ ƯU TIÊN
                        ordered_slice = list(context_slice)
                        if line_scan_priority == 'anchor_first':
                            if actual_context == 'above':
                                ordered_slice.reverse() # [anchor, ..., outer]
                        elif line_scan_priority == 'outer_first':
                            if actual_context == 'below':
                                ordered_slice.reverse() # [outer, ..., anchor]

                        # PHASE 1: QUÉT TỪNG DÒNG (LINE-BY-LINE)
                        if logger: logger(f"   -> [Phase 1] Quét từng dòng với ưu tiên '{line_scan_priority}'")
                        for ctx_line in ordered_slice:
                            date_matches = re.findall(date_pattern, ctx_line.text, re.IGNORECASE)
                            if date_matches:
                                last_match = date_matches[-1] 
                                result = self._process_date_match(last_match, logger, field, date_format_mode)
                                if result: return result
                        
                        # PHASE 2: FALLBACK (QUÉT GỘP DÒNG BẢO VỆ ĐỨT GÃY OCR)
                        if logger: logger(f"   -> [Phase 2] Quét gộp dòng (Fallback) do đứt gãy OCR")
                        context_text = " ".join([l.text for l in context_slice]) 
                        date_matches = re.findall(date_pattern, context_text, re.IGNORECASE)
                        
                        if date_matches:
                            # Phân luồng lấy kết quả dựa trên cờ điều hướng
                            if actual_context == 'above':
                                target_match = date_matches[-1] if line_scan_priority == 'anchor_first' else date_matches[0]
                            else: 
                                target_match = date_matches[0] if line_scan_priority == 'anchor_first' else date_matches[-1]
                                
                            result = self._process_date_match(target_match, logger, field, date_format_mode)
                            if result: return result

                        continue 
                            
        if logger: logger(f"EVENT=EXTRACTION_FAILED | FIELD={field} | REASON=no_valid_anchor_or_date")
        return None

    def _process_date_match(self, match_tuple, logger, field, date_format_mode):
        raw_d = match_tuple[0].strip() if match_tuple[0] else ""
        raw_m = match_tuple[1].strip() if match_tuple[1] else ""
        raw_y = match_tuple[2].strip() if match_tuple[2] else ""
        
        if not raw_y:
            return None
            
        final_d = raw_d if raw_d else "00"
        final_m = raw_m if raw_m else "00"
        final_y = raw_y
        
        result = f"{final_d}/{final_m}/{final_y}"
        if logger: logger(f"EVENT=EXTRACTION_SUCCESS | FIELD={field} | METHOD=proximity | MODE={date_format_mode} | VALUE='{result}'")
        return result