from .base_strategy import BaseStrategy
from .payload import ExtractionPayload

class FixedValueStrategy(BaseStrategy):
    def _do_extract(self, working_lines: list, payload: ExtractionPayload, logger=None):
        rule = payload.rule_config
        field = rule.get('field_name')
        value = rule.get('value', '')
        
        if logger: logger(f"EVENT=EXTRACTION_SUCCESS | FIELD={field} | METHOD=fixed_value | VALUE='{value}'")
        return value