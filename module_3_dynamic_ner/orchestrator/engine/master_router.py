# File: orchestrator/engine/master_router.py
import json
import re
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz
from .logger import log_xray
from .ocr_utils import cluster_y_overlap  # Hoặc utils.ocr_utils tùy cấu trúc dự án của bạn

class MasterRouter:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.master_config = {}
        self.routes = {}
        self.threshold = 90
        self._load_config()

    def _load_config(self):
        master_path = self.config_dir / "master_router.json"
        if master_path.exists():
            try:
                with open(master_path, 'r', encoding='utf-8') as f:
                    self.master_config = json.load(f)
                    self.routes = self.master_config.get("routes", {})
                    self.threshold = self.master_config.get("master_threshold", 90)
                log_xray(f"[MASTER_ROUTER] Đã nạp tuyến đường cho {len(self.routes)} thư mục.")
            except Exception as e:
                log_xray(f"[MASTER_ROUTER_ERROR] Lỗi đọc master_router.json: {e}")
        else:
            log_xray("[MASTER_ROUTER] CẢNH BÁO: Không tìm thấy file master_router.json!")

    def _normalize_for_router(self, text: str) -> str:
        if not text: return ""
        text = text.replace('Đ', 'D').replace('đ', 'd')
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^a-zA-Z0-9]', '', text).upper()
        return text

    def identify_folder(self, page_words: list, page_num: int) -> str:
        if not page_words: 
            log_xray(f"ROUTER_EVAL | PAGE={page_num} | STATUS=EMPTY_PAGE | LABEL=UNKNOWN")
            return "UNKNOWN"
            
        # 1. Tách không gian vật lý
        top_20_words = page_words[:20]
        clustered_lines = cluster_y_overlap(top_20_words, page_num)
        
        log_xray(f"ROUTER_DUMP | PAGE={page_num} | WORDS_SCANNED={len(top_20_words)} | LINES_FORMED={len(clustered_lines)} | THRESHOLD={self.threshold}")

        # 2. Rà soát từng dòng (Line-by-line evaluation)
        for line_block in clustered_lines:
            normalized_line = self._normalize_for_router(line_block.text)
            
            # Tấm khiên gạt rác tối thiểu (Minimal Noise Floor)
            if len(normalized_line) < 5:
                continue
                
            best_line_score = 0
            best_folder = "UNKNOWN"
            best_match_info = ""

            # 3. Quét chéo thư mục (Thứ tự ưu tiên được đảm bảo bằng trình tự JSON)
            for folder_name, keywords in self.routes.items():
                for keyword in keywords:
                    normalized_keyword = self._normalize_for_router(keyword)
                    score = fuzz.partial_ratio(normalized_keyword, normalized_line)
                    
                    # Ngắt sớm tuyến 1: Trúng đích tuyệt đối
                    if score == 100:
                        log_xray(f"[MASTER_ROUTER] EXACT MATCH 100%! PAGE={page_num} | FOLDER=/{folder_name} | KEYWORD='{keyword}' | LINE='{line_block.text}'")
                        return folder_name
                        
                    # Cập nhật điểm cao nhất của dòng. Dấu > giải quyết đụng độ (người đến trước thắng).
                    if score > best_line_score:
                        best_line_score = score
                        best_folder = folder_name
                        best_match_info = keyword

            # Ngắt sớm tuyến 2: Đạt chuẩn 90% trở lên
            if best_line_score >= self.threshold:
                log_xray(f"[MASTER_ROUTER] MATCHED! PAGE={page_num} | FOLDER=/{best_folder} | KEYWORD='{best_match_info}' | SCORE={best_line_score:.2f} >= {self.threshold} | LINE='{line_block.text}'")
                return best_folder

        # Trả về UNKNOWN nếu toàn bộ các dòng vật lý đều trượt
        log_xray(f"ROUTER_FAIL | PAGE={page_num} | STATUS=NO_LINE_MET_THRESHOLD")
        return "UNKNOWN"