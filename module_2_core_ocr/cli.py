import argparse
import os
import json
import dataclasses
from pathlib import Path
import numpy as np 
import logging

# Initialize the logger instance
logger = logging.getLogger(__name__)

from module_2_core_ocr.ocr_pipeline import OcrPipeline
from module_2_core_ocr.config import OcrConfig

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # 1. Xử lý Numpy Array (Ma trận ảnh, tọa độ)
        if isinstance(o, np.ndarray):
            return o.tolist()
            
        # 2. Xử lý Numpy Scalars (np.float32, np.int64...)
        if isinstance(o, np.generic):
            return o.item()
            
        # 3. 🌟 XỬ LÝ OBJECT TỰ ĐỊNH NGHĨA (OcrResult, OcrWord, BoundingBox...)
        # Nếu model.py của bạn dùng @dataclass
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
            
        # Nếu model.py của bạn dùng Pydantic (rất phổ biến trong dự án AI)
        if hasattr(o, 'model_dump') and callable(o.model_dump):
            return o.model_dump()
        if hasattr(o, 'dict') and callable(o.dict):
            return o.dict()
            
        # Nếu model.py của bạn dùng Class Python thuần túy
        if hasattr(o, '__dict__'):
            return o.__dict__
            
        # 4. Trả về mặc định cho các kiểu dữ liệu khác
        return super().default(o)

def main():
    parser = argparse.ArgumentParser(description="CLI Khởi chạy Module 2 (Core OCR)")
    parser.add_argument("--input_dir", required=True, help="Thư mục chứa ảnh và JSON từ Module 1")
    parser.add_argument("--output_dir", required=True, help="Thư mục lưu file JSON kết quả OCR")
    
    # Hướng B: Đặt default=None để ưu tiên nhận cấu hình từ file YAML trung tâm
    parser.add_argument("--engine", default=None, help="Tên Động cơ OCR muốn khởi chạy (Mặc định đọc từ YAML)")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. ĐỌC CẤU HÌNH ĐỘNG TỪ FILE YAML TRUNG TÂM
    # Khi chạy lệnh từ thư mục gốc, đường dẫn tương đối sẽ là configs/module2_defaults.yaml
    yaml_path = Path("configs/module2_defaults.yaml")
    if yaml_path.exists():
        config = OcrConfig.from_yaml(yaml_path)
    else:
        print(f"⚠️ [M2-CLI] Không tìm thấy file cấu hình tại {yaml_path}. Khởi tạo cấu hình mặc định.")
        config = OcrConfig()
    
    # 2. ĐỒNG BỘ ĐỘNG CƠ: Nếu người dùng không truyền cờ --engine, lôi cấu hình active từ YAML ra
    active_engine = args.engine if args.engine is not None else config.active_engine
    
    print(f"[M2-CLI] Khởi tạo Nhạc trưởng Pipeline với động cơ: {active_engine}")
    pipeline = OcrPipeline(active_engine_name=active_engine, config=config)
    
    print(f"[M2-CLI] Bắt đầu quét và xử lý thư mục: {input_path}")
    batch_results = pipeline.process_folder(input_path)
    
    # LƯU KẾT QUẢ RA FILE JSON
    for filename, result_obj in batch_results.items():
        base_name = os.path.splitext(filename)[0]
        json_file_path = output_path / f"{base_name}_ocr.json"

        # Thêm dòng này để vứt bỏ ma trận ảnh (tránh file nặng 1.8GB)
        if isinstance(result_obj, dict) and "rotated_image" in result_obj:
            del result_obj["rotated_image"]
        
        # ... (Code lưu JSON cũ giữ nguyên)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_obj, f, cls=EnhancedJSONEncoder, indent=4, ensure_ascii=False)

        # 🌟 VÁ LỖI ĐỌC BIẾN: Ép kiểu linh hoạt và cảnh báo nếu rỗng
        md_output_path = str(json_file_path).replace('_ocr.json', '_layout.md')
        md_content = ""
        
        # 1. Bóc lớp vỏ bọc của ocr_pipeline.py (nếu result_obj đang là dict chứa key "result")
        actual_result = result_obj
        if isinstance(result_obj, dict) and "result" in result_obj:
            actual_result = result_obj["result"]
            
        # 2. Rút ruột Markdown từ lõi
        if isinstance(actual_result, dict):
            md_content = actual_result.get("markdown_text", "")
        else:
            md_content = getattr(actual_result, "markdown_text", "")
            
        # 3. Ghi file
        if md_content and md_content.strip() != "":
            with open(md_output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"✅ Đã lưu cấu trúc Markdown tại: {md_output_path}")
        else:
            logger.warning(f"⚠️ CẢNH BÁO: Không có text Markdown nào được sinh ra cho {md_output_path}")
            
        print(f"✅ Đã lưu kết quả JSON tại: {json_file_path}")

if __name__ == "__main__":
    main()

# python -m module_2_core_ocr.cli --input_dir test_input --output_dir test_output