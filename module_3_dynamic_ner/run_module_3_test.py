import os
import sys
import json
import re
from pathlib import Path

# ==========================================
# ⚙️ CÔNG TẮC TÙY BIẾN ĐẦU RA (OUTPUT TOGGLES)
# ==========================================
EXPORT_EXCEL = True   
EXPORT_PDF = True     
EVALUATE_RESULTS = True  # CÔNG TẮC MỚI: Bật tính năng Chấm điểm Tự động

# Thêm đường dẫn gốc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module_3_dynamic_ner.orchestrator.session_manager import SessionManager
from module_3_dynamic_ner.orchestrator.export_excel import export_to_excel
from module_3_dynamic_ner.orchestrator.export_pdf import export_to_pdf  

def extract_page_number(filename: str) -> int:
    match = re.search(r'page_(\d+)', filename.lower())
    if match: return int(match.group(1))
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def print_rich_console_table(records: list):
    print("\n" + "═" * 85)
    print(f" 🔎 BẢNG ĐỐI CHIẾU TRỰC QUAN - KẾT QUẢ BÓC TÁCH ({len(records)} HỒ SƠ)")
    print("═" * 85)
    for idx, record in enumerate(records, 1):
        print(f"\n 📁 HỒ SƠ SỐ {idx}:")
        print("┌──────────────────────┬──────────────────────────────────────────────────────────────┐")
        print("│ TRƯỜNG THÔNG TIN     │ GIÁ TRỊ BÓC TÁCH ĐƯỢC                                        │")
        print("├──────────────────────┼──────────────────────────────────────────────────────────────┤")
        for key, value in record.items():
            display_val = str(value) if value is not None else "[KHÔNG TÌM THẤY - LỖI/RỖNG]"
            if len(display_val) > 58: display_val = display_val[:55] + "..."
            print(f"│ {key:<20} │ {display_val:<60} │")
        print("└──────────────────────┴──────────────────────────────────────────────────────────────┘")

def run_auto_evaluator(extracted_records: list, gt_path: Path):
    """Động cơ Chấm điểm Tự động thông minh dựa trên Khóa chính (Trang số)"""
    if not gt_path.exists():
        print(f"\n⚠️ KHÔNG THỂ CHẤM ĐIỂM: Không tìm thấy file đáp án tại {gt_path}")
        return

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_records = json.load(f)

    # Đưa đáp án vào Dictionary với Key là to_so_trang_so
    gt_dict = {str(r.get('to_so_trang_so', '')).strip(): r for r in gt_records if r.get('to_so_trang_so')}

    total_fields = 0
    passed_fields = 0
    missing_pages = set(gt_dict.keys())
    
    fields_to_check = ['so_ky_hieu_tai_lieu', 'ngay_thang_van_ban', 'ten_loai_tai_lieu', 'tac_gia']

    print("\n" + "█" * 85)
    print(" 📊 BÁO CÁO ĐÁNH GIÁ TỰ ĐỘNG (AUTO-EVALUATION BENCHMARK)")
    print("█" * 85)

    for record in extracted_records:
        page_key = str(record.get('to_so_trang_so', '')).strip()
        print(f"\n📄 Đang chấm điểm Hồ sơ trang: [{page_key}]")

        if page_key not in gt_dict:
            print(f"   ⚠️ [BÁO ĐỘNG] Máy bóc ra trang {page_key} nhưng KHÔNG CÓ trong Đáp án chuẩn (Bóc vống hoặc lệch trang).")
            continue

        missing_pages.discard(page_key)
        gt_record = gt_dict[page_key]

        for field in fields_to_check:
            val_extracted = str(record.get(field, '') or '').strip()
            val_gt = str(gt_record.get(field, '') or '').strip()

            total_fields += 1
            
            # Smart Match: Chuyển chữ thường, gộp khoảng trắng dư thừa ở giữa các chữ
            val_ex_norm = " ".join(val_extracted.lower().split())
            val_gt_norm = " ".join(val_gt.lower().split())

            if val_ex_norm == val_gt_norm:
                passed_fields += 1
                print(f"   ✅ {field:<20}: ĐÚNG")
            else:
                print(f"   ❌ {field:<20}: SAI")
                print(f"      - Bóc được  : '{val_extracted}'")
                print(f"      - Đáp án    : '{val_gt}'")

    if missing_pages:
        print(f"\n🚨 [LỖI NGHIÊM TRỌNG] HỆ THỐNG ĐÃ NUỐT MẤT CÁC TRANG SAU: {', '.join(missing_pages)}")

    print("\n" + "═" * 85)
    if total_fields > 0:
        acc = (passed_fields / total_fields) * 100
        print(f" 📈 ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {acc:.2f}% ({passed_fields}/{total_fields} trường dữ liệu đúng)")
    print("═" * 85)


def main():
    INPUT_JSON_DIR = PROJECT_ROOT / "tests/data/unit_tests/module_3/test_sodo_only"
    OUTPUT_DIR = PROJECT_ROOT / "tests/data/outputs/unit_tests/module_3"
    GROUND_TRUTH_PATH = PROJECT_ROOT / "tests/data/outputs/unit_tests/module_3/ground_truth/ground_truth.json"
    
    print("==================================================")
    print("🚀 KHỞI ĐỘNG LUỒNG KIỂM THỬ MODULE 3 (VERTICAL SLICE)")
    print("==================================================")

    catalog_config_path = PROJECT_ROOT / "module_3_dynamic_ner" / "configs" / "document_catalog.json"
    if not catalog_config_path.exists():
        return

    with open(catalog_config_path, 'r', encoding='utf-8') as f:
        catalog_config = json.load(f)

    json_files = [f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')]
    json_files.sort(key=extract_page_number)
    print(f"📂 Thứ tự nạp file: {json_files}")

    stream_of_pages = []
    for file_name in json_files:
        with open(INPUT_JSON_DIR / file_name, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        stream_of_pages.append({
            "page_num": extract_page_number(file_name),
            "full_text": ocr_data.get("full_text", ""),
            "words": ocr_data.get("words", [])
        })

    print("\n🧠 Đang khởi tạo SessionManager và thực thi bóc tách...")
    manager = SessionManager(catalog_config)
    extracted_records = manager.process_document_stream(stream_of_pages)

    if extracted_records:
        print_rich_console_table(extracted_records)
        
        # --- KÍCH HOẠT HỆ THỐNG CHẤM ĐIỂM ---
        if EVALUATE_RESULTS:
            run_auto_evaluator(extracted_records, GROUND_TRUTH_PATH)
            
        # --- KÍCH HOẠT ĐẦU RA BÁO CÁO ---
        if EXPORT_EXCEL:
            excel_path = OUTPUT_DIR / "muc_luc_quan_ly.xlsx"
            export_to_excel(extracted_records, str(excel_path))
            
        if EXPORT_PDF:
            pdf_path = OUTPUT_DIR / "muc_luc_quan_ly.pdf"
            export_to_pdf(extracted_records, str(pdf_path))
            
    else:
        print("\n⚠️ Cảnh báo: Không bóc được hồ sơ nào!")

if __name__ == "__main__":
    main()