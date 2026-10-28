from .proximity_strategy import ProximityNumberStrategy
from .fixed_value_strategy import FixedValueStrategy
from .global_regex_strategy import GlobalRegexStrategy
from .global_dictionary_strategy import GlobalDictionaryStrategy
from .hybrid_dictionary_strategy import HybridDictionaryStrategy
from .multiline_dictionary_strategy import MultilineDictionaryStrategy
from .conditional_append_strategy import ConditionalAppendStrategy
from .fuzzy_key_value_strategy import FuzzyKeyValueStrategy
from .extract_between_anchors_strategy import ExtractBetweenAnchorsStrategy
from .inherit_context_strategy import InheritContextStrategy



class ExtractorFactory:
    _dispatch_table = {
        "proximity_number_search": ProximityNumberStrategy(),
        "fixed_value": FixedValueStrategy(),
        "global_regex": GlobalRegexStrategy(),

        "global_dictionary_match": GlobalDictionaryStrategy(),
        "hybrid_dictionary_regex": HybridDictionaryStrategy(),
        "multiline_dictionary_match": MultilineDictionaryStrategy(),
        
        "fixed_string_with_conditional_append": ConditionalAppendStrategy(),
        "fuzzy_key_value": FuzzyKeyValueStrategy(),
        "extract_between_anchors": ExtractBetweenAnchorsStrategy(),
        "inherit_from_context": InheritContextStrategy()
    }
    
    @classmethod
    def get_strategy(cls, method_name: str):
        # Trả về Strategy nếu có, nếu không trả về None (rất quan trọng cho bước sau)
        return cls._dispatch_table.get(method_name)