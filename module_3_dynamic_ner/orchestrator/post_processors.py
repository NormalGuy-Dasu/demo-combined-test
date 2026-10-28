import re

class PostProcessor:
    def __init__(self, logger=None):
        # Thiết kế Tiêm phụ thuộc (DI) với Fallback linh hoạt
        self.log = logger or print
        
        # Cuốn "Danh bạ" (Registry) ánh xạ tên hàm trong JSON với code Python
        self.registry = {
            "clean_targeted_punctuation": self._clean_targeted_punctuation,
            "format_date_vi": self._format_date_vi,
            "format_with_suffix": self._format_with_suffix,
            "extract_date_from_mixed_text": self._extract_date_from_mixed_text,
            "fix_ocr_date_typos": self._fix_ocr_date_typos,
            "clean_so_ky_hieu": self._clean_so_ky_hieu,
            "fix_ocr_number_typos": self._fix_ocr_number_typos,
            "clean_so_cong_chung": self._clean_so_cong_chung
        }

    def run_pipeline(self, raw_value: str, processors: list) -> str:
        """Bộ định tuyến chạy qua các lớp lọc (Pipeline) tuần tự"""
        if not raw_value or not isinstance(raw_value, str) or not processors:
            return raw_value

        current_value = raw_value
        for method_str in processors:
            # 1. Cắt chuỗi để lấy tên hàm và tham số (nếu có)
            parts = method_str.split('|')
            proc_name = parts[0]
            
            if proc_name in self.registry:
                try:
                    func = self.registry[proc_name]
                    
                    # 2. Phân luồng: Có tham số vs Không có tham số
                    if len(parts) > 1:
                        suffix = parts[1]
                        current_value = func(current_value, suffix)
                    else:
                        current_value = func(current_value)
                        
                except Exception as e:
                    # Safe Fallback: Báo lỗi nhưng vẫn bảo toàn giá trị thô trước khi sập
                    self.log(f"    -> [PostProcessor] Cảnh báo: Lỗi khi chạy '{proc_name}': {e}. Hoàn tác về bước trước đó.")
            else:
                self.log(f"    -> [PostProcessor] Bỏ qua: Không tìm thấy hàm '{proc_name}' trong Registry.")
        
        return current_value

    # --- CÁC HÀM HELPER ĐƠN NHIỆM TRỰC THUỘC CLASS ---

    def _clean_targeted_punctuation(self, text: str) -> str:
        """Dọn rác có mục tiêu: Chuẩn hóa chuỗi số và bảo vệ cấu trúc pháp lý"""
        text = text.strip()
        
        # 1. Xóa dấu chấm/phẩy ở vị trí CUỐI CÙNG của chuỗi (Vd: "14.12." -> "14.12")
        text = re.sub(r'[.,]+$', '', text)
        
        # 2. Xóa dấu kẹt giữa Chữ và Số (Vd: "tháng.2" -> "tháng 2")
        text = re.sub(r'([a-zA-ZÀ-ỹ])[.,]+(\d)', r'\1 \2', text)
        
        # 3. Xóa dấu kẹt giữa Số và Chữ (Vd: "2.năm" -> "2 năm")
        text = re.sub(r'(\d)[.,]+([a-zA-ZÀ-ỹ])', r'\1 \2', text)
        
        # 4. [NEW] TẨY RỬA DẤU PHÂN CÁCH NGÀY THÁNG: Xóa chấm/phẩy bám quanh dấu gạch chéo, gạch ngang
        # (Vd: "20./5./2014" -> "20/5/2014", "20./5../2014" -> "20/5/2014")
        text = re.sub(r'[.,]*([/-])[.,]*', r'\1', text)
        
        # 5. CHÉM THẲNG TAY: Nối liền các con số bị ngăn cách bởi dấu chấm/phẩy (Vd: "201.3" -> "2013")
        text = re.sub(r'(\d)[.,]+(\d)', r'\1\2', text)
        
        # 6. Gom các khoảng trắng thừa thành 1
        return re.sub(r'\s+', ' ', text).strip()

    def _format_date_vi(self, text: str) -> str:
        """Chuẩn hóa ngày tháng sang định dạng dd/mm/yyyy, hỗ trợ nội suy năm 2 chữ số"""
        # Cấu trúc Regex đã được cập nhật để bắt cả dấu gạch ngang (-) của giấy tay
        match = re.search(r'(?:ngày\s*)?(\d{1,2})\s*(?:tháng|/|-)?\s*(\d{1,2})\s*(?:năm|/|-)?\s*(\d{2,4})', text, re.IGNORECASE)
        
        if match:
            d, m, y = match.groups()
            y = int(y)
            
            # Logic thông minh xử lý năm 2 chữ số (ví dụ: 96)
            if y < 100:
                if y > 50:
                    y += 1900  # Nếu năm > 50 (ví dụ 96), nội suy thành thế kỷ 20 (1996)
                else:
                    y += 2000  # Nếu năm <= 50 (ví dụ 14), nội suy thành thế kỷ 21 (2014)
                    
            return f"{int(d):02d}/{int(m):02d}/{y}"
            
        return text

    def _format_with_suffix(self, text: str, suffix: str):
        """
        Lọc sạch rác OCR (giữ lại số, chữ cái và dấu gạch ngang) và nối hậu tố.
        Trả về None nếu chuỗi không chứa dữ liệu hợp lệ.
        """
        if not text:
            return None
            
        # [UPGRADE]: Vét các ký tự số (0-9), chữ cái (a-zA-Z) và dấu gạch ngang (-)
        clean_text = re.sub(r'[^0-9a-zA-Z\-]', '', text)
        
        # Xử lý ngoại lệ: Trả về None nếu sau khi lọc không còn gì
        if not clean_text:
            return None
            
        # Nối hậu tố chuẩn và trả về
        return f"{clean_text}{suffix}"

    def _extract_date_from_mixed_text(self, value: str, *args) -> str:
        """
        Trích xuất ngày/tháng/năm từ một chuỗi lộn xộn.
        - Chấp nhận 00 cho trường hợp khuyết dữ liệu.
        - Tích hợp logic nội suy năm 2 chữ số thành 4 chữ số.
        """
        if not value or not isinstance(value, str):
            return value
            
        numbers = re.findall(r'\d+', value)
        
        for i in range(len(numbers) - 1, 1, -1):
            y_str, m_str, d_str = numbers[i], numbers[i-1], numbers[i-2]
            
            try:
                y, m, d = int(y_str), int(m_str), int(d_str)
                
                # --- LOGIC CẤY GHÉP TỪ FORMAT_DATE_VI ---
                if y < 100:
                    if y > 50:
                        y += 1900  # Nội suy thế kỷ 20
                    else:
                        y += 2000  # Nội suy thế kỷ 21
                # -----------------------------------------
                
                valid_year = (y >= 1900)
                
                # Hàng rào kiểm định (Đã nới lỏng cho phép số 0)
                if valid_year and (0 <= m <= 12) and (0 <= d <= 31):
                    # Trả về y (đã được nội suy) thay vì y_str gốc
                    return f"{str(d).zfill(2)}/{str(m).zfill(2)}/{y}"
            except ValueError:
                continue
                
        return value

    def _fix_ocr_date_typos(self, text: str) -> str:
        """
        Dọn dẹp và nội suy các lỗi OCR khi quét khối ngày tháng.
        Sử dụng cơ chế Token-based Callback kết hợp Từ điển Typo toàn diện.
        """
        if not text:
            return text

        # 1. TẤM KHIÊN NGOẶC ĐƠN: Quét sạch mọi cụm ký tự nằm trong ngoặc tròn.
        text = re.sub(r'\(.*?\)', '', text)
            
        # 2. TẤM KHIÊN TỪ KHÓA MỞ RỘNG (Fuzzy Regex):
        text = re.sub(r'(?i)\b(d[a-z]*te|m[a-z]*nth|y[a-z]*r|d[a-z]*y)\b', '', text)
            
        # 3. TỪ ĐIỂN ÁNH XẠ chuẩn xác của bạn
        typo_map = {
            'S': '5', 's': '5',
            'O': '0', 'o': '0',
            'Q': '0', 'q': '0',
            'l': '1', 'I': '1', 'i': '1', '|': '1',
            'B': '8', 'b': '8',
            'z': '2', 'Z': '2',
            'đ': '2',
            '[': '', ']': '', 
            '{': '', '}': '',
            'v': '0', 'V': '0',
            '(': '', ')': ''
        }

        # 4. Hàm Callback xử lý độc lập
        def process_garbage_token(match):
            token = match.group(0)
            fixed_chars = []

            for char in token:
                if char in typo_map:
                    fixed_chars.append(typo_map[char])

            result = "".join(fixed_chars)
            
            # TRẢ VỀ CHUỖI ĐÃ CỨU. NẾU RỖNG -> NEO LẠI BẰNG '0'
            return result if result else "0"

        # 5. LƯỚI QUÉT MỞ RỘNG
        return re.sub(r'[a-zA-ZÀ-ỹ\(\)\[\]\{\}\|]+', process_garbage_token, text)
            

    def _fix_ocr_number_typos(self, text: str) -> str:
        """
        [NEW] Ép các ký tự chữ cái bị OCR nhận diện nhầm về dạng số nguyên thủy.
        Chỉ áp dụng cho các trường dữ liệu được đảm bảo 100% là số.
        """
        if not text:
            return text

        # Bảng ánh xạ tương đồng thị giác (Visual Similarity Typo Map)
        typo_map = {
            'S': '5', 's': '5',
            'O': '0', 'o': '0', 'Q': '0', 'q': '0', 'D': '0',
            'B': '8', 'b': '8',
            'Z': '2', 'z': '2',
            'l': '1', 'I': '1', 'i': '1', '|': '1',
            'G': '6', 'g': '6',
            'A': '4'
        }

        # Kỹ thuật Pythonic: Duyệt qua từng ký tự, nếu có trong từ điển thì đổi thành số, không thì giữ nguyên
        result = "".join(typo_map.get(char, char) for char in text)
        
        return result
    
    def _clean_so_ky_hieu(self, text: str) -> str:
        """
        Dọn dẹp rác OCR cho trường Số ký hiệu tài liệu.
        Đã nâng cấp: Ủi phẳng mọi dấu (.,;:) kẹt giữa các con số 
        và san phẳng các cụm rác từ 2 dấu chấm trở lên.
        """
        if not text:
            return text
            
        # 1. LƯỚI QUÉT RỘNG (CHÉM THẲNG TAY): Xóa dấu chấm, phẩy, chấm phẩy, hai chấm kẹt giữa 2 con số
        # (Vd: "77.2" -> "772", "12;3" -> "123", "20,14" -> "2014")
        text = re.sub(r'(\d)[.;,:]+(\d)', r'\1\2', text)
        
        # 2. XỬ LÝ RÁC CỤM: Thay thế 2 dấu chấm trở lên (\.{2,}) HOẶC khoảng trắng (\s+) thành 1 khoảng trắng duy nhất
        # (Vd: "772....TB-CCT" -> "772 TB-CCT")
        text = re.sub(r'\.{2,}|\s+', ' ', text)
        
        # 3. DỌN DẸP BIÊN: Xóa các dấu chấm, gạch ngang, gạch chéo thừa ở 2 đầu (nếu có)
        # (Vd: "-772/TB-" -> "772/TB")
        return text.strip(' .-/')
    
    def _clean_so_cong_chung(self, text: str) -> str:
        """
        [UPGRADE] Hút chân không tuyệt đối: Chỉ giữ lại duy nhất chữ số (0-9).
        Do hàm fix_ocr_number_typos đã cứu các chữ cái hợp lệ thành số từ trước,
        mọi ký tự còn sót lại (như ?, !, phẩy, khoảng trắng) đều là rác và bị tiêu diệt.
        """
        if not text:
            return text
            
        # Dùng \D (Non-digit) để tìm và xóa SẠCH mọi thứ không phải là số
        return re.sub(r'\D', '', text)