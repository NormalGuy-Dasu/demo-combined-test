import json
from pathlib import Path
from .logger import log_xray

class ConfigCache:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.sub_configs = {} 
        self.pre_load_all()

    def pre_load_all(self):
        for folder_path in sorted(self.config_dir.iterdir()):
            if folder_path.is_dir() and folder_path.name != "dictionaries":
                folder_name = folder_path.name
                self.sub_configs[folder_name] = []
                for json_file in sorted(folder_path.glob("*.json")):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            self._resolve_dictionary_paths(config)
                            self.sub_configs[folder_name].append(config)
                    except Exception as e:
                        log_xray(f"[CACHE_ERROR] Lỗi nạp file {json_file.name}: {e}")
        log_xray(f"[CACHE] Hoàn tất nạp cấu hình lên RAM cho {len(self.sub_configs)} thư mục.")

    def _resolve_dictionary_paths(self, config):
        if 'extraction_rules' not in config: return
        for rule in config['extraction_rules']:
            method = rule.get('extraction_method')
            if method in ['global_dictionary_match', 'hybrid_dictionary_regex', 'multiline_dictionary_match']:
                master_data = rule.get('master_data')
                if isinstance(master_data, str) and master_data.endswith('.txt'):
                    txt_path = self.config_dir / master_data
                    if txt_path.exists():
                        with open(txt_path, 'r', encoding='utf-8') as txt_f:
                            cached_list = [line.strip() for line in txt_f if line.strip()]
                        rule['master_data'] = cached_list
                    else:
                        rule['master_data'] = []