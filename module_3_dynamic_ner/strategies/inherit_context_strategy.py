from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class InheritContextStrategy(BaseStrategy):
    def execute(self, payload: ExtractionPayload, logger=None):
        """
        GHI ĐÈ TẦNG KHIÊN: Thuật toán này không cần đọc text OCR.
        Nó thọc tay thẳng vào RAM (dossier_context) lấy data rồi về luôn.
        """
        rule = payload.rule_config
        context_key = rule.get('context_key')
        
        if context_key and context_key in payload.dossier_context:
            value = payload.dossier_context[context_key]
            if logger: logger(f"EVENT=EXTRACTION_SUCCESS | METHOD=inherit_from_context | INHERITED_FROM='{context_key}' | VALUE='{value}'")
            return value
        else:
            if logger: logger(f"EVENT=EXTRACTION_FAILED | METHOD=inherit_from_context | REASON=context_key_not_found_or_empty")
            return None
            
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        # Hàm này khai báo để tuân thủ luật của BaseStrategy nhưng sẽ không bao giờ được gọi
        pass