# File: orchestrator/session_manager.py
import json
from pathlib import Path
from rapidfuzz import fuzz
from .engine.logger import log_xray
from .engine.master_router import MasterRouter
from .engine.config_cache import ConfigCache
from .engine.document_session import DocumentSession

class SessionManager:
    def __init__(self, catalog_config: dict = None):
        self.extracted_records = []
        self.dossier_context = {}
        
        self.config_dir = Path(__file__).resolve().parents[1] / "configs"
        
        self.master_router = MasterRouter(self.config_dir)
        self.config_cache = ConfigCache(self.config_dir)
        
        self.active_session = None
        self.active_session_folder = None

    def _check_triggers(self, text: str, triggers: list, threshold: float, exclude_triggers: list = None) -> bool:
        if exclude_triggers:
            for ex_trigger in exclude_triggers:
                if fuzz.partial_ratio(ex_trigger.upper(), text) > threshold:
                    log_xray(f"EVENT=TRIGGER_BLOCKED | TYPE=exclude | KEYWORD='{ex_trigger}'")
                    return False

        for trigger in triggers:
            if fuzz.partial_ratio(trigger.upper(), text) > threshold: 
                return True
        return False
    
    def _close_active_session(self, trigger_page):
        if self.active_session:
            log_xray(f"[X-RAY] EVENT=SESSION_END | TRIGGER_AT_PAGE={trigger_page} | DOC_TYPE={self.active_session.document_type}")
            record = self.active_session.flush_and_extract(self.dossier_context)
            self.extracted_records.append(record)
            self.active_session = None
            self.active_session_folder = None
    
    def process_document_stream(self, stream_of_pages):
        for page in stream_of_pages:
            page_text = page['full_text']
            clean_text = page_text.replace('\n', ' ').upper()
            page_num = page['page_num']
            
            log_xray(f"\n\n[DEBUG_ROUTER] === BẮT ĐẦU XỬ LÝ TRANG {page_num} ===")
            
            # --- [BƯỚC 1] TẦNG 1: ĐỊNH TUYẾN MASTER ---
            # Gọi MasterRouter truyền vào mảng words để lấy 20 ô đầu tiên
            folder_name = self.master_router.identify_folder(page.get('words', []), page_num)
            
            matched_plugin = None
            
            # --- [BƯỚC 2] TẦNG 2: RÀ SOÁT SUB-CONFIGS ---
            if folder_name != "UNKNOWN":
                candidate_configs = self.config_cache.sub_configs.get(folder_name, [])
                for config in candidate_configs:
                    radar_limit = config.get('radar_lines', 20)
                    dynamic_header_text = " ".join(page_text.split('\n')[:radar_limit]).upper()
                    
                    if self._check_triggers(dynamic_header_text, 
                                            config.get('start_triggers', []), 
                                            config.get('start_threshold', 85), 
                                            config.get('exclude_triggers', [])):
                        matched_plugin = config
                        log_xray(f"[DEBUG_ROUTER] MATCHED SUB-PLUGIN: {config.get('document_type')}")
                        break

            # --- [BƯỚC 3] KIỂM TRA ĐỤNG ĐỘ / CHUYỂN PHIÊN ---
            if self.active_session:
                if matched_plugin:
                    is_single = matched_plugin.get('is_single_page', False)
                    # [UPGRADE]: Bổ sung cờ ép cắt phiên cho tài liệu cùng loại
                    force_restart = matched_plugin.get('force_restart_on_same_trigger', False)
                    
                    # Nếu Tầng 2 xác nhận có tài liệu mới -> Auto-Flush phiên cũ ngay lập tức khi:
                    # 1. Khác loại tài liệu (!=)
                    # 2. Hoặc là tài liệu 1 trang (is_single)
                    # 3. Hoặc cùng loại nhưng có cờ ép buộc (force_restart)
                    if (self.active_session.document_type != matched_plugin.get('document_type')) or is_single or force_restart:
                        log_xray(f"[DEBUG_ROUTER] PHÁT HIỆN TÀI LIỆU MỚI -> AUTO-FLUSH PHIÊN CŨ.")
                        self._close_active_session(page_num - 1)
                elif folder_name != "UNKNOWN" and self.active_session_folder != folder_name:
                    # Tầng 1 báo đổi folder nhưng Tầng 2 không thấy trigger -> Báo động giả
                    log_xray(f"[DEBUG_ROUTER] BÁO ĐỘNG GIẢ TỪ TẦNG 1 (Không có start_trigger) -> Ép về UNKNOWN kế thừa.")
                    folder_name = "UNKNOWN"

            # --- [BƯỚC 4] MỞ PHIÊN MỚI ---
            if matched_plugin and self.active_session is None:
                doc_type = matched_plugin.get('document_type')
                log_xray(f"[X-RAY] EVENT=SESSION_START | PAGE={page_num} | DOC_TYPE={doc_type}")
                
                # [REFACTOR]: Đọc cờ giới hạn bộ hồ sơ từ JSON thay vì hardcode
                if matched_plugin.get('is_dossier_boundary', False):
                    log_xray(f"[X-RAY] EVENT=DOSSIER_BOUNDARY_DETECTED | ACTION=clear_context | DOC_TYPE={doc_type}")
                    self.dossier_context.clear()
                    
                self.active_session = DocumentSession(matched_plugin)
                self.active_session_folder = folder_name

            # --- [BƯỚC 5] BƠM DỮ LIỆU & KIỂM TRA ĐIỂM DỪNG (END TRIGGER) ---
            if self.active_session:
                # 5.1 Bơm dữ liệu của trang hiện tại vào phiên
                self.active_session.add_page_data(page_num, page['words'])
                
                # 5.2 Kiểm tra End Trigger nằm ở ĐÁY vòng lặp (Bảo vệ 1-page documents)
                current_config = self.active_session.config
                if self._check_triggers(clean_text, current_config.get('end_triggers', []), current_config.get('end_threshold', 85)):
                    log_xray(f"[DEBUG_ROUTER] MATCHED END_TRIGGER TẠI TRANG {page_num} -> ĐÓNG PHIÊN NGAY.")
                    self._close_active_session(page_num)
            else:
                log_xray(f"[DEBUG_ROUTER] TRANG RÁC (Không thuộc phiên nào) -> BỎ QUA.")

        if self.active_session:
            log_xray("[X-RAY] EVENT=STREAM_END | ACTION=force_close_session")
            self._close_active_session("STREAM_END")
            
        return self.extracted_records