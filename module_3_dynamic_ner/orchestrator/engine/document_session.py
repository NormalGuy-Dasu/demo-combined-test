import re
import json
from pathlib import Path
from rapidfuzz import process, fuzz
import unicodedata
from ..post_processors import PostProcessor
from strategies.extractor_factory import ExtractorFactory
from strategies.payload import ExtractionPayload
from .logger import log_xray
from .ocr_utils import cluster_y_overlap, LineBlock

class DocumentSession:
    def __init__(self, config: dict):
        self.config = config
        self.document_type = config.get('document_type', 'UNKNOWN_DOCUMENT')
        self.pages_data = [] 
        self.start_page = None
        self.end_page = None
        self._typo_caches = {}

    def add_page_data(self, page_num: int, ocr_words: list):
        if self.start_page is None:
            self.start_page = page_num
        self.end_page = page_num
        self.pages_data.append((page_num, ocr_words))

    # ĐÃ XÓA HÀM GOM DÒNG Ở ĐÂY VÌ ĐÃ CÓ OCR_UTILS LO

    def flush_and_extract(self, dossier_context: dict = None) -> dict:
        if dossier_context is None: dossier_context = {}
            
        log_xray(f"EVENT=SESSION_FLUSH | DOC_TYPE={self.document_type} | START_PAGE={self.start_page} | END_PAGE={self.end_page}")
        all_lines = []
        for page_num, words in self.pages_data:
            all_lines.extend(cluster_y_overlap(words, page_num))

        global_text = "\n".join([line.text for line in all_lines])
        result = {"to_so_trang_so": f"{self.start_page:02d}-{self.end_page:02d}" if self.start_page != self.end_page else f"{self.start_page:02d}"}

        for rule in self.config.get('extraction_rules', []):
            field = rule['field_name']
            method = rule['extraction_method']
            log_xray(f"EVENT=EXTRACTION_START | FIELD={field} | METHOD={method}")

            strategy = ExtractorFactory.get_strategy(method)
            if strategy:
                payload = ExtractionPayload(
                    all_lines=all_lines, 
                    rule_config=rule,
                    dossier_context=dossier_context, 
                    global_text=global_text
                )
                result[field] = strategy.execute(payload, logger=log_xray)
            else:
                log_xray(f"[CẢNH BÁO] Không tìm thấy thuật toán '{method}' trong ExtractorFactory!")
                result[field] = None

            if 'export_to_context' in rule and result.get(field):
                dossier_context[rule['export_to_context']] = result[field]

        post_processor = PostProcessor(logger=log_xray) 
        for rule in self.config.get('extraction_rules', []):
            field = rule['field_name']
            if result.get(field) and 'post_processing' in rule:
                result[field] = post_processor.run_pipeline(
                    result[field], 
                    rule['post_processing']
                )
        return result