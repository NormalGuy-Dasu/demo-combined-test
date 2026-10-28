import argparse
from module_1_image_preprocessing.preprocessor import ImagePreprocessor

def main():
    parser = argparse.ArgumentParser(description="Màng bọc CLI cho Module 1")
    parser.add_argument("--input_dir", required=True, help="Thư mục chứa ảnh gốc")
    parser.add_argument("--output_dir", required=True, help="Thư mục lưu ảnh sạch")
    
    # 1. THÊM CÔNG TẮC NÀY VÀO LỚP VỎ
    parser.add_argument("--skip_crop", action="store_true", help="Bỏ qua bước cắt góc AI (Dành cho ảnh phẳng/PDF)")
    
    args = parser.parse_args()
    
    print(f"[M1-CLI] Đang khởi chạy tiền xử lý cho thư mục: {args.input_dir}")
    
    processor = ImagePreprocessor()
    # 2. TRUYỀN CÔNG TẮC XUỐNG LÕI
    processor.process_folder(args.input_dir, args.output_dir, skip_crop=args.skip_crop)
    
    print("[M1-CLI] Hoàn thành tiền xử lý 100%.")

if __name__ == "__main__":
    main()