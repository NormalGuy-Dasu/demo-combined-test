import argparse
import json
import os
from pipeline import DocumentPipeline

def main():
    parser = argparse.ArgumentParser(description="CLI Wrapper cho Module 3 (Dynamic NER)")
    parser.add_argument("--input_ocr_json", required=True, help="Đường dẫn tới file kết quả OCR từ Module 2")
    parser.add_argument("--registry_file", required=True, help="Đường dẫn tới file Kho lược đồ (Schema Registry)")
    parser.add_argument("--template_name", required=True, help="Từ khóa của loại hồ sơ (vd: toa_an, bao_hiem)")
    parser.add_argument("--output_file", required=True, help="Đường dẫn lưu file JSON bóc tách cuối cùng")
    args = parser.parse_args()

    print(f"[M3-CLI] Đang khởi chạy bóc tách thực thể động cho biểu mẫu: '{args.template_name}'")

    # 1. Tải Lược đồ (Schema) từ Registry
    if not os.path.exists(args.registry_file):
        print(f"[LỖI] Không tìm thấy file Registry tại: {args.registry_file}")
        return
        
    with open(args.registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    if args.template_name not in registry:
        print(f"[LỖI] Không tìm thấy template '{args.template_name}' trong Registry. Vui lòng kiểm tra lại cấu hình!")
        return
        
    target_schema = registry[args.template_name]

    # 2. Tải dữ liệu OCR
    if not os.path.exists(args.input_ocr_json):
        print(f"[LỖI] Không tìm thấy file OCR đầu vào tại: {args.input_ocr_json}")
        return
        
    with open(args.input_ocr_json, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    # 3. Kích hoạt Lõi AI (Thay bằng logic thật của bạn)
    pipeline = DocumentPipeline()
    final_result = pipeline.process(ocr_data, target_schema)
    
    # Dữ liệu Mock để test luồng
    final_result = {
        "status": "success",
        "template_applied": args.template_name,
        "extracted_data": "Dữ liệu đã được bóc tách thành công dựa trên Lược đồ"
    }

    # 4. Ghi file kết quả
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

    print(f"[M3-CLI] Hoàn tất! Kết quả bóc tách đã được lưu tại: {args.output_file}")

if __name__ == "__main__":
    main()