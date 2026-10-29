"""VN-Digitize AI Engine - public demo (one app, two clusters).

Cluster 1: Module 1 on camera photos (crop and straighten, QR codes).
Cluster 2: Modules 2 + 3 on scanned land-record PDFs (OCR, then record fields).

Thin wrapper only: it calls the project code unchanged. Both engines are loaded once and shared, one visitor
runs at a time (one lock for both tabs), sizes and pages are limited, and nothing is kept (D-019): photos stay
in memory, PDF pages go to a temporary folder that is deleted afterwards.
"""
import dataclasses
import gc
import io
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

# The project does not use alpha matting; this avoids 30-60 s of numba compile at every start.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_3_dynamic_ner"))  # same as PYTHONPATH=module_3_dynamic_ner

# Limits (shown to the visitor in the Limits section)
MAX_PHOTO_MB = 15
MAX_MEGAPIXELS = 30
MIN_SIDE = 50
SHOW_MAX_SIDE = 1400
MAX_PAGES = 5
MAX_PDF_MB = 30
ZOOM = 2.0

SAMPLES = ROOT / "samples"
PHOTO_SAMPLES = [SAMPLES / "photos" / f"photo_{i}.jpg" for i in (1, 2, 3, 4)]
QR_SAMPLES = [SAMPLES / "qr" / f"qr_{i}.jpg" for i in (1, 2)]
MANIFEST = Path(os.environ.get("DEMO_SAMPLE_MANIFEST", SAMPLES / "pdf" / "manifest.json"))

# ---------- texts (every visible message exists in both languages) ----------

TEXT = {
    "English": {
        "tagline": "Turns camera photos and scanned land records into clean images and structured data.",
        "intro": "Pick a tab, then upload a file or try a sample. Everything runs on a free server without a graphics card, so a scanned page takes a few seconds.",
        "tab_photo": "Camera photo",
        "tab_pdf": "Scanned PDF",
        "photo_intro": "Finds the page in a photo, then crops and straightens it. Blank pages are flagged.",
        "pdf_intro": "Reads a scanned land-record PDF page by page, splits it into documents and pulls out the fields of each one.",
        "sub_crop": "Crop and straighten",
        "sub_qr": "QR code",
        "upload_photo": "Photo of a document (JPG or PNG, up to {mb} MB)",
        "upload_qr": "Image with a QR code or barcode (JPG or PNG, up to {mb} MB)",
        "upload_pdf": "Scanned PDF (up to {pages} pages, {mb} MB)",
        "samples": "Or try a sample",
        "sample_n": "Sample {n}",
        "category": "Category",
        "doc_type": "Document type",
        "pages_count": "{n} page(s)",
        "use_sample": "Use this sample",
        "run_crop": "Crop and straighten",
        "run_qr": "Read the codes",
        "run_pdf": "Extract the fields",
        "before": "Before",
        "after": "After",
        "download_image": "Download the straightened image",
        "done_in": "Done in {s} s.",
        "tech": "Technical details",
        "t_size": "Image size",
        "t_mp": "Megapixels",
        "t_skew": "Rotation corrected (degrees)",
        "t_time": "Processing time (s)",
        "t_mem": "Memory in use (MB)",
        "t_peak": "Peak memory (MB)",
        "t_pages": "Pages",
        "t_boxes": "Text boxes found",
        "t_render": "Page rendering (s)",
        "t_ocr": "Reading text (s)",
        "t_extract": "Field extraction (s)",
        "t_per_page": "Per page (s)",
        "t_deleted": "Temporary files deleted",
        "yes": "Yes",
        "no": "No",
        "col_item": "Item",
        "col_value": "Value",
        "col_type": "Type",
        "col_content": "Content",
        "qr_found": "Found {n} code(s).",
        "qr_none": "No QR code or barcode was found. Try a sharper image where the code fills more of the picture.",
        "qr_note": "Reads the codes in the whole picture, with no cropping.",
        "err_bad_image": "This file could not be read. Choose a valid JPG or PNG image.",
        "err_too_big": "The image has {mp} megapixels, above the limit of {max} megapixels. Use a lower resolution.",
        "err_too_small": "The image is too small (shorter side under {px} px).",
        "err_file_big": "The file is larger than {mb} MB.",
        "err_m1": "The photo step reported an error: {code}",
        "err_bad_pdf": "This file could not be read. Choose a valid PDF.",
        "err_pages": "The PDF has {n} pages, above the limit of {max} pages. Cut it down and try again.",
        "warn_crop": "The page could not be found clearly, so the original photo is kept. Retake it with the whole page in view against a contrasting background.",
        "warn_blank": "This page looks blank.",
        "info_orient": "The text may be upside down or sideways.",
        "busy": "Another visitor is being served. Please try again in a few minutes.",
        "working_photo": "Working on the photo...",
        "working_qr": "Looking for codes...",
        "progress_page": "Page {i} of {n}: reading the text",
        "progress_extract": "Extracting the fields...",
        "found_records": "Found {n} record(s).",
        "no_records": "This PDF was not recognised as one of the supported land-record documents, so there is nothing to show. Try one of the samples.",
        "sub_fields": "Extracted fields",
        "sub_text": "Text read from the pages",
        "sub_boxes": "Text boxes on the pages",
        "boxes_note": "Each red frame is a piece of text the reader found on the page.",
        "accuracy": "This project is still being improved, so accuracy is not yet 100%. Some results may be incomplete or wrong, especially on blurry photos, unusual layouts or documents outside the supported types. Please check the output before relying on it.",
        "page_n": "Page {n}",
        "no_text": "No text was found on this page.",
        "record_n": "Record {n}",
        "dl_excel": "Download Excel",
        "dl_json": "Download JSON",
        "privacy_h": "Privacy",
        "privacy": "Files are processed in memory or in a temporary folder that is deleted right after the run. Nothing is stored. Please do not upload photos or documents with personal data; use the samples. The sample land records were edited so their personal data is not real.",
        "limits_h": "Limits",
        "limits": "Photos: JPG or PNG, up to {photo_mb} MB and {mp} megapixels.\n\nPDFs: up to {pages} pages and {pdf_mb} MB, about 12 seconds per page.\n\nOne visitor is served at a time. After a long idle time the app needs about a minute to wake up.",
        "credits_h": "Credits",
        "credits": "Page finding and clean-up: OpenCV, U²-Net-P through rembg. QR codes: ZBar through pyzbar. Text detection: PaddleOCR. Vietnamese text recognition: VietOCR. PDF pages: PyMuPDF. Interface: Streamlit.\n\nSample photos come from the WarpDoc dataset (CVPR 2022). The QR images come from web receipts. A file is removed on request.",
        "col_field": "Field",
    },
    "Tiếng Việt": {
        "tagline": "Biến ảnh chụp và bản scan hồ sơ đất đai thành ảnh sạch và dữ liệu có cấu trúc.",
        "intro": "Chọn một thẻ, rồi tải file lên hoặc thử một mẫu có sẵn. Mọi thứ chạy trên máy chủ miễn phí không có card đồ họa, nên một trang scan mất vài giây.",
        "tab_photo": "Ảnh chụp",
        "tab_pdf": "PDF scan",
        "photo_intro": "Tìm tờ giấy trong ảnh, rồi cắt và làm thẳng. Trang trống sẽ được gắn cờ.",
        "pdf_intro": "Đọc từng trang của PDF hồ sơ đất đai đã scan, tách thành từng tài liệu và lấy ra các trường thông tin của mỗi tài liệu.",
        "sub_crop": "Cắt và làm thẳng",
        "sub_qr": "Mã QR",
        "upload_photo": "Ảnh chụp tài liệu (JPG hoặc PNG, tối đa {mb} MB)",
        "upload_qr": "Ảnh có mã QR hoặc mã vạch (JPG hoặc PNG, tối đa {mb} MB)",
        "upload_pdf": "PDF đã scan (tối đa {pages} trang, {mb} MB)",
        "samples": "Hoặc thử một mẫu",
        "sample_n": "Mẫu {n}",
        "category": "Nhóm",
        "doc_type": "Loại tài liệu",
        "pages_count": "{n} trang",
        "use_sample": "Dùng mẫu này",
        "run_crop": "Cắt và làm thẳng",
        "run_qr": "Đọc mã",
        "run_pdf": "Trích xuất thông tin",
        "before": "Trước",
        "after": "Sau",
        "download_image": "Tải ảnh đã làm thẳng",
        "done_in": "Xong trong {s} giây.",
        "tech": "Chi tiết kỹ thuật",
        "t_size": "Kích thước ảnh",
        "t_mp": "Số megapixel",
        "t_skew": "Góc xoay đã chỉnh (độ)",
        "t_time": "Thời gian xử lý (giây)",
        "t_mem": "Bộ nhớ đang dùng (MB)",
        "t_peak": "Bộ nhớ cao nhất (MB)",
        "t_pages": "Số trang",
        "t_boxes": "Số khung chữ tìm thấy",
        "t_render": "Dựng trang (giây)",
        "t_ocr": "Đọc chữ (giây)",
        "t_extract": "Trích xuất thông tin (giây)",
        "t_per_page": "Mỗi trang (giây)",
        "t_deleted": "File tạm đã xóa",
        "yes": "Có",
        "no": "Không",
        "col_item": "Mục",
        "col_value": "Giá trị",
        "col_type": "Loại",
        "col_content": "Nội dung",
        "qr_found": "Đọc được {n} mã.",
        "qr_none": "Không tìm thấy mã QR hay mã vạch. Hãy thử ảnh nét hơn, trong đó mã chiếm nhiều diện tích hơn.",
        "qr_note": "Đọc mã trên toàn bộ ảnh, không cắt ảnh.",
        "err_bad_image": "Không đọc được file này. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.",
        "err_too_big": "Ảnh có {mp} megapixel, vượt giới hạn {max} megapixel. Hãy dùng ảnh độ phân giải thấp hơn.",
        "err_too_small": "Ảnh quá nhỏ (cạnh ngắn dưới {px} px).",
        "err_file_big": "File lớn hơn {mb} MB.",
        "err_m1": "Bước xử lý ảnh báo lỗi: {code}",
        "err_bad_pdf": "Không đọc được file này. Hãy chọn một file PDF hợp lệ.",
        "err_pages": "PDF có {n} trang, vượt giới hạn {max} trang. Hãy cắt bớt rồi thử lại.",
        "warn_crop": "Không tìm thấy rõ tờ giấy nên giữ nguyên ảnh gốc. Hãy chụp lại: tờ giấy nằm trọn trong khung, nền tương phản.",
        "warn_blank": "Trang này có vẻ trống.",
        "info_orient": "Chữ có thể đang bị ngược hoặc nằm ngang.",
        "busy": "Đang phục vụ một người dùng khác. Vui lòng thử lại sau ít phút.",
        "working_photo": "Đang xử lý ảnh...",
        "working_qr": "Đang tìm mã...",
        "progress_page": "Trang {i}/{n}: đang đọc chữ",
        "progress_extract": "Đang trích xuất thông tin...",
        "found_records": "Tìm thấy {n} bản ghi.",
        "no_records": "PDF này không được nhận ra là một trong các loại hồ sơ đất đai được hỗ trợ, nên không có gì để hiển thị. Hãy thử một trong các mẫu.",
        "sub_fields": "Thông tin trích xuất",
        "sub_text": "Chữ đọc được từ các trang",
        "sub_boxes": "Khung chữ trên trang",
        "boxes_note": "Mỗi khung đỏ là một đoạn chữ mà hệ thống tìm thấy trên trang.",
        "accuracy": "Dự án vẫn đang được hoàn thiện nên độ chính xác chưa đạt 100%. Một số kết quả có thể thiếu hoặc sai, nhất là với ảnh mờ, bố cục lạ hoặc tài liệu ngoài các loại được hỗ trợ. Vui lòng kiểm tra lại kết quả trước khi sử dụng.",
        "page_n": "Trang {n}",
        "no_text": "Không tìm thấy chữ nào trên trang này.",
        "record_n": "Bản ghi {n}",
        "dl_excel": "Tải Excel",
        "dl_json": "Tải JSON",
        "privacy_h": "Quyền riêng tư",
        "privacy": "File được xử lý trong bộ nhớ hoặc trong thư mục tạm, xóa ngay sau khi chạy xong. Không có gì được lưu lại. Vui lòng không tải lên ảnh hay tài liệu có thông tin cá nhân; hãy dùng các mẫu. Các hồ sơ đất mẫu đã được chỉnh sửa nên thông tin cá nhân trong đó không có thật.",
        "limits_h": "Giới hạn",
        "limits": "Ảnh: JPG hoặc PNG, tối đa {photo_mb} MB và {mp} megapixel.\n\nPDF: tối đa {pages} trang và {pdf_mb} MB, khoảng 12 giây mỗi trang.\n\nMỗi lần chỉ phục vụ một người. Sau thời gian dài không ai dùng, ứng dụng cần khoảng một phút để thức dậy.",
        "credits_h": "Ghi công",
        "credits": "Tìm tờ giấy và làm sạch ảnh: OpenCV, U²-Net-P qua rembg. Mã QR: ZBar qua pyzbar. Phát hiện chữ: PaddleOCR. Nhận dạng chữ tiếng Việt: VietOCR. Trang PDF: PyMuPDF. Giao diện: Streamlit.\n\nẢnh chụp mẫu lấy từ bộ dữ liệu WarpDoc (CVPR 2022). Ảnh QR lấy từ biên lai trên mạng. File sẽ được gỡ khi có yêu cầu.",
        "col_field": "Trường",
    },
}

# Field names of a Module 3 record -> label in each language
FIELD_LABELS = {
    "ten_loai_tai_lieu": ("Document type", "Loại tài liệu"),
    "so_ky_hieu_tai_lieu": ("Document number", "Số, ký hiệu"),
    "ngay_thang_van_ban": ("Date", "Ngày tháng văn bản"),
    "tac_gia": ("Issuing agency", "Cơ quan ban hành"),
    "to_so_trang_so": ("Pages", "Tờ số, trang số"),
    "thong_tin_chu_so_huu_an": ("Owner details", "Thông tin chủ sở hữu"),
}


def lang():
    return st.session_state.get("lang_choice") or "English"


def tr(key, **kw):
    s = TEXT[lang()][key]
    return s.format(**kw) if kw else s


def field_label(key):
    pair = FIELD_LABELS.get(key)
    if pair:
        return pair[0 if lang() == "English" else 1]
    return key.replace("_", " ").capitalize()


# ---------- engines ----------

def rss_mb():
    import psutil

    return round(psutil.Process().memory_info().rss / 1024 / 1024)


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def seed_models():
    """Put the bundled models where the libraries look for them, so nothing is downloaded at start."""
    src = ROOT / "models" / "paddleocr" / "whl"
    if src.exists():
        shutil.copytree(src, Path.home() / ".paddleocr" / "whl", dirs_exist_ok=True)
    pth = ROOT / "models" / "vgg_seq2seq.pth"
    target = Path(tempfile.gettempdir()) / "vgg_seq2seq.pth"
    if pth.exists() and not target.exists():
        shutil.copy2(pth, target)
    u2 = ROOT / "models" / "u2netp.onnx"
    u2_target = Path.home() / ".u2net" / "u2netp.onnx"
    if u2.exists() and not u2_target.exists():
        u2_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(u2, u2_target)


@st.cache_resource(show_spinner="Loading the models (first start only)...")
def load_engines():
    seed_models()
    from module_2_core_ocr.config import OcrConfig
    from module_2_core_ocr.ocr_pipeline import OcrPipeline

    cfg = OcrConfig.from_yaml(ROOT / "configs" / "module2_defaults.yaml")
    pipe = OcrPipeline(cfg.active_engine, cfg)
    catalog = json.loads((ROOT / "module_3_dynamic_ner" / "configs" / "document_catalog.json").read_text(encoding="utf-8"))
    from module_1_image_preprocessing.preprocessor import ImagePreprocessor

    pre = ImagePreprocessor.from_yaml(ROOT / "configs" / "module1_defaults.yaml")
    return {"pipe": pipe, "catalog": catalog, "pre": pre, "lock": threading.Lock()}


@st.cache_data(show_spinner=False)
def thumbnail(path):
    from PIL import Image, ImageOps

    im = ImageOps.fit(Image.open(path).convert("RGB"), (300, 300), centering=(0.5, 0.35))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def pdf_pages(path):
    """Every page of a sample PDF as a small JPEG (whole page, not cropped)."""
    import fitz
    from PIL import Image

    pages = []
    with fitz.open(path) as doc:
        for page in doc:
            zoom = 900 / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            buf = io.BytesIO()
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(buf, "JPEG", quality=80)
            pages.append(buf.getvalue())
    return pages


def load_manifest():
    """Categories of PDF samples. A type without a file yet is left out, and so is an empty category."""
    base = MANIFEST.parent
    cats = []
    for cat in json.loads(MANIFEST.read_text(encoding="utf-8"))["categories"]:
        types = [t for t in cat["types"] if (base / t["file"]).exists()]
        if types:
            cats.append({**cat, "types": [{**t, "path": base / t["file"]} for t in types]})
    return cats


def shown(name):
    """A plain string is a Vietnamese document name, shown the same in both languages; a dict is translated."""
    return name if isinstance(name, str) else name[lang()]


def pdf_gallery():
    """Samples by category: each type shows all its pages and a button. Returns the bytes of the chosen sample."""
    cats = load_manifest()
    if not cats:
        return None
    names = {c["id"]: c["name"][lang()] for c in cats}
    # the language is part of the widget key so the names are redrawn when the language changes; the choice is kept in session_state
    ids = list(names)
    chosen = st.session_state.get("pdf_cat_id", ids[0])
    cat_id = st.selectbox(tr("category"), ids, index=ids.index(chosen) if chosen in ids else 0,
                          format_func=names.get, key=f"pdf_category_{lang()}")
    st.session_state["pdf_cat_id"] = cat_id
    cat = next(c for c in cats if c["id"] == cat_id)
    types = cat["types"]
    labels = {i: shown(t["name"]) for i, t in enumerate(types)}  # one button per document type, even when there is just one
    k = st.pills(tr("doc_type"), list(labels), format_func=labels.get, default=0, key=f"pdf_type_{cat_id}_{lang()}")
    k = 0 if k is None else k
    typ = types[k]
    pages = pdf_pages(str(typ["path"]))
    with st.container(border=True):
        st.markdown(f"**{shown(typ['name'])}**")
        st.caption(tr("pages_count", n=len(pages)))
        for start in range(0, len(pages), 3):
            cols = st.columns(3)
            for col, img in zip(cols, pages[start:start + 3]):
                col.image(img, width="stretch")
        if st.button(tr("use_sample"), key=f"pdf_sample_{cat_id}_{k}"):
            return typ["path"].read_bytes()
    return None


# ---------- Cluster 1: photos ----------

def to_jpeg(bgr, max_side=SHOW_MAX_SIDE, quality=85):
    import cv2

    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def check_image(data):
    """Returns (error dict or None, (w, h))."""
    from PIL import Image

    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:  # noqa: BLE001
        return {"error": "bad_image"}, None
    if w * h > MAX_MEGAPIXELS * 1_000_000:
        return {"error": "too_big", "mp": round(w * h / 1e6, 1)}, None
    if min(w, h) < MIN_SIDE:
        return {"error": "too_small"}, None
    return None, (w, h)


def decode_image(data):
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def run_crop(data, eng):
    t_all = time.perf_counter()
    err, size = check_image(data)
    if err:
        return err
    img = decode_image(data)
    if img is None:
        return {"error": "bad_image"}
    t = time.perf_counter()
    res = eng["pre"].process(img)
    t_m1 = time.perf_counter() - t
    w, h = size
    out = {
        "before": to_jpeg(img),
        "after": to_jpeg(res.processed_image) if res.processed_image is not None else None,
        "after_full": to_jpeg(res.processed_image, 100000, 92) if res.processed_image is not None else None,
        "is_blank": res.is_blank,
        "orientation": res.orientation_status.value,
        "warnings": list(res.warnings),
        "error_code": res.error_code,
        "total_s": round(time.perf_counter() - t_all, 1),
        "stats": [
            ("t_size", f"{w} x {h}"), ("t_mp", round(w * h / 1e6, 1)), ("t_skew", round(float(res.skew_angle), 1)),
            ("t_time", round(t_m1, 2)), ("t_mem", rss_mb()), ("t_peak", peak_mb()),
        ],
    }
    del img, res
    gc.collect()
    return out


def run_qr(data, eng):
    t_all = time.perf_counter()
    err, size = check_image(data)
    if err:
        return err
    img = decode_image(data)
    if img is None:
        return {"error": "bad_image"}
    from module_1_image_preprocessing._detect import detect_barcodes

    codes = detect_barcodes(img, eng["pre"]._config.detect)
    out = {
        "image": to_jpeg(img),
        "codes": [{"type": b.barcode_type, "data": b.data} for b in codes],
        "total_s": round(time.perf_counter() - t_all, 1),
        "stats": [("t_size", f"{size[0]} x {size[1]}"), ("t_time", round(time.perf_counter() - t_all, 2)), ("t_mem", rss_mb()), ("t_peak", peak_mb())],
    }
    del img
    gc.collect()
    return out


# ---------- Cluster 2: PDFs ----------

def boxed_page(png_path, words, max_side=1000):
    """The page image with a red frame around every text box the OCR found (JPEG bytes)."""
    from PIL import Image, ImageDraw

    im = Image.open(png_path).convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    for w in words:
        draw.polygon([(x * s, y * s) for x, y in w["bbox"]["points"]], outline=(230, 50, 50), width=2)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return buf.getvalue()


def run_pipeline(data, eng, progress):
    """PDF bytes -> page images -> Module 2 -> Module 3. Never keeps files."""
    import fitz
    from module_3_dynamic_ner.orchestrator.session_manager import SessionManager

    t_all = time.perf_counter()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:  # noqa: BLE001
        return {"error": "bad_pdf"}
    n = doc.page_count
    if n > MAX_PAGES:
        doc.close()
        return {"error": "too_many_pages", "pages": n}

    stream, boxed, boxes, t_render, t_ocr = [], [], 0, 0.0, 0.0
    with tempfile.TemporaryDirectory(prefix="demo_") as work:
        work = Path(work)
        for i in range(n):
            progress.progress(i / n, text=tr("progress_page", i=i + 1, n=n))
            pdir = work / f"p{i + 1}"
            pdir.mkdir()
            t = time.perf_counter()
            doc.load_page(i).get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM)).save(str(pdir / "page.png"))
            t_render += time.perf_counter() - t
            t = time.perf_counter()
            res = eng["pipe"].process_folder(pdir)
            t_ocr += time.perf_counter() - t
            r = res["page.png"]["result"]
            r.page_number = i + 1
            d = json.loads(json.dumps(dataclasses.asdict(r), ensure_ascii=False))  # same shape as the page JSON files
            stream.append({"page_num": i + 1, "full_text": d.get("full_text", ""), "words": d.get("words", [])})
            boxes += len(d.get("words", []))
            boxed.append(boxed_page(pdir / "page.png", d.get("words", [])))
            del res, r
            shutil.rmtree(pdir)
        doc.close()
        progress.progress(1.0, text=tr("progress_extract"))
        t = time.perf_counter()
        records = SessionManager(eng["catalog"]).process_document_stream(stream)
        t_m3 = time.perf_counter() - t
        left = len(list(work.iterdir()))
    total = time.perf_counter() - t_all
    gc.collect()
    return {
        "records": records,
        "page_text": [(p["page_num"], p["full_text"]) for p in stream],
        "page_boxes": boxed,
        "total_s": round(total, 1),
        "deleted": left == 0 and not work.exists(),
        "stats": [
            ("t_pages", n), ("t_boxes", boxes), ("t_render", round(t_render, 1)), ("t_ocr", round(t_ocr, 1)),
            ("t_extract", round(t_m3, 2)), ("t_per_page", round(total / n, 1)), ("t_mem", rss_mb()), ("t_peak", peak_mb()),
        ],
    }


def cell(v):
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, ensure_ascii=False)


def records_to_rows(records):
    return [{field_label(k): cell(v) for k, v in rec.items()} for rec in records]


def excel_bytes(records):
    import pandas as pd

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(records_to_rows(records)).to_excel(xw, index=False, sheet_name="records")
    return buf.getvalue()


# ---------- screens ----------

def tech_details(result):
    import pandas as pd

    rows = []
    for key, val in result["stats"]:
        rows.append({tr("col_item"): tr(key), tr("col_value"): str(val)})
    if "deleted" in result:
        rows.append({tr("col_item"): tr("t_deleted"), tr("col_value"): tr("yes" if result["deleted"] else "no")})
    with st.expander(tr("tech")):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def show_error(r):
    e = r.get("error")
    msgs = {
        "bad_image": lambda: tr("err_bad_image"),
        "too_big": lambda: tr("err_too_big", mp=r.get("mp"), max=MAX_MEGAPIXELS),
        "too_small": lambda: tr("err_too_small", px=MIN_SIDE),
        "bad_pdf": lambda: tr("err_bad_pdf"),
        "too_many_pages": lambda: tr("err_pages", n=r.get("pages"), max=MAX_PAGES),
    }
    if e in msgs:
        st.error(msgs[e]())
        return True
    return False


def show_crop(r):
    if show_error(r):
        return
    if r.get("error_code"):
        st.error(tr("err_m1", code=r["error_code"]))
    if "CROP_FAILED_PLEASE_RETAKE" in r["warnings"]:
        st.warning(tr("warn_crop"))
    if r["is_blank"]:
        st.warning(tr("warn_blank"))
    if r["orientation"] != "LIKELY_CORRECT":
        st.info(tr("info_orient"))
    from PIL import Image

    def aspect(jpeg):
        w, h = Image.open(io.BytesIO(jpeg)).size
        return min(max(w / h, 0.4), 2.5)

    # column widths follow the pictures' proportions, so both come out the same height and nothing is left empty
    c1, c2 = st.columns([aspect(r["before"]), aspect(r["after"]) if r["after"] else 1])
    c1.caption(tr("before"))
    c1.image(r["before"], width="stretch")
    c2.caption(tr("after"))
    if r["after"]:
        c2.image(r["after"], width="stretch")
        c2.download_button(tr("download_image"), r["after_full"], file_name="straightened.jpg", mime="image/jpeg", key="dl_crop")
    st.caption(tr("done_in", s=r["total_s"]))
    tech_details(r)


def show_qr(r):
    if show_error(r):
        return
    c1, c2 = st.columns([1, 1])
    c1.image(r["image"], width="stretch")
    if r["codes"]:
        c2.success(tr("qr_found", n=len(r["codes"])))
        import pandas as pd

        c2.dataframe(pd.DataFrame([{tr("col_type"): c["type"], tr("col_content"): c["data"]} for c in r["codes"]]), hide_index=True, width="stretch")
    else:
        c2.warning(tr("qr_none"))
    st.caption(tr("done_in", s=r["total_s"]))
    tech_details(r)


def kv_markdown(rec):
    lines = [f"| {tr('col_field')} | {tr('col_value')} |", "|---|---|"]
    order = list(FIELD_LABELS)
    for k, v in sorted(rec.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)):
        text = str(cell(v)).replace("|", "\\|").replace("\n", " ") or "-"
        lines.append(f"| {field_label(k)} | {text} |")
    return "\n".join(lines)


def show_pdf(r):
    if show_error(r):
        return
    records = r["records"]
    sub_fields, sub_text, sub_boxes = st.tabs([tr("sub_fields"), tr("sub_text"), tr("sub_boxes")])
    with sub_fields:
        if not records:
            st.warning(tr("no_records"))
        else:
            st.success(tr("found_records", n=len(records)))
            for i, rec in enumerate(records, 1):
                with st.container(border=True):
                    st.markdown(f"**{tr('record_n', n=i)}**")
                    st.markdown(kv_markdown(rec))
            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button(tr("dl_excel"), excel_bytes(records), file_name="records.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
            d2.download_button(tr("dl_json"), json.dumps(records, ensure_ascii=False, indent=2), file_name="records.json",
                               mime="application/json", key="dl_json")
    with sub_text:
        for num, text in r["page_text"]:
            with st.expander(tr("page_n", n=num)):
                st.text(text.strip() or tr("no_text"))
    with sub_boxes:
        st.caption(tr("boxes_note"))
        for num, img in enumerate(r["page_boxes"], 1):
            with st.expander(tr("page_n", n=num)):
                st.image(img, width="stretch")
    st.caption(tr("done_in", s=r["total_s"]))
    tech_details(r)


# ---------- shared: lock and runners ----------

def guarded(fn, label_key, *args):
    """One visitor at a time (one lock for both tabs); the lock is always released."""
    eng = args[-1]
    if not eng["lock"].acquire(blocking=False):
        st.warning(tr("busy"))
        return None
    try:
        with st.spinner(tr(label_key)):
            return fn(*args)
    finally:
        eng["lock"].release()


def guarded_pdf(data, eng):
    if not eng["lock"].acquire(blocking=False):
        st.warning(tr("busy"))
        return None
    try:
        bar = st.progress(0.0, text=tr("progress_page", i=1, n="?"))
        result = run_pipeline(data, eng, bar)
        bar.empty()
        return result
    finally:
        eng["lock"].release()


def input_block(prefix, upload_label, samples, run_label, max_mb, sample_kind, runner):
    """Uploader + run button + sample buttons. Stores the result in session_state[prefix]."""
    up = st.file_uploader(upload_label, type=["pdf"] if prefix == "pdf" else ["jpg", "jpeg", "png"], key=f"{prefix}_up")
    data = None
    if up is not None and st.button(run_label, type="primary", key=f"{prefix}_run"):
        if up.size > max_mb * 1024 * 1024:
            st.session_state[prefix] = None
            st.error(tr("err_file_big", mb=max_mb))
        else:
            data = up.getvalue()
    st.markdown(f"<div class='samples-title'>{tr('samples')}</div>", unsafe_allow_html=True)
    if prefix == "pdf":
        data = pdf_gallery() or data
    else:
        cols = st.columns(4, vertical_alignment="top")
        for i, path in enumerate(samples):
            with cols[i]:
                st.image(thumbnail(str(path)), width="stretch")
                if st.button(tr("sample_n", n=i + 1), key=f"{prefix}_sample_{i}", width="stretch"):
                    data = path.read_bytes()
    if data is not None:
        res = runner(data)
        if res is not None:
            st.session_state[prefix] = res


# ---------- page ----------

CSS = """
<style>
:root { --accent: #6c78e6; --accent-soft: rgba(108,120,230,.13); --line: rgba(128,128,128,.28); }
.block-container { max-width: 1080px; padding-top: 4.6rem; padding-bottom: 3rem; }
div[data-testid="stExpander"] p { text-align: justify; }
.brand { font-size: 1.55rem; font-weight: 650; letter-spacing: -.01em; line-height: 1.2; }
.brand span { color: var(--accent); }
.tagline { opacity: .75; margin: .15rem 0 .9rem 0; }
.intro { max-width: 46rem; opacity: .9; margin-bottom: 1.2rem; }
.samples-title { margin: 1.1rem 0 .4rem 0; font-weight: 600; font-size: .95rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-size: 1.02rem; font-weight: 600; padding-left: 0; padding-right: 0; }
.stTabs [aria-selected="true"] { color: var(--accent); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }
div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button { border-radius: 8px; }
[data-testid="stFileUploaderDropzoneInstructions"] span { display: none; }
footer { visibility: hidden; }
</style>
"""

st.set_page_config(page_title="VN-Digitize AI Engine", page_icon="📄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

head_l, head_r = st.columns([4, 1.4], vertical_alignment="center")
with head_r:
    st.segmented_control("Language", ["English", "Tiếng Việt"], default="English", key="lang_choice", label_visibility="collapsed")
with head_l:
    st.markdown("<div class='brand'>VN-Digitize <span>AI Engine</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tagline'>{tr('tagline')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='intro'>{tr('intro')}<br><br>{tr('accuracy')}</div>", unsafe_allow_html=True)

engine = load_engines()
for key in ("crop", "qr", "pdf"):
    st.session_state.setdefault(key, None)

tab_photo, tab_pdf = st.tabs([tr("tab_photo"), tr("tab_pdf")])

with tab_photo:
    st.write(tr("photo_intro"))
    sub_crop, sub_qr = st.tabs([tr("sub_crop"), tr("sub_qr")])
    with sub_crop:
        input_block("crop", tr("upload_photo", mb=MAX_PHOTO_MB), PHOTO_SAMPLES, tr("run_crop"), MAX_PHOTO_MB, "photo",
                    lambda d: guarded(run_crop, "working_photo", d, engine))
        if st.session_state["crop"] is not None:
            show_crop(st.session_state["crop"])
    with sub_qr:
        st.caption(tr("qr_note"))
        input_block("qr", tr("upload_qr", mb=MAX_PHOTO_MB), QR_SAMPLES, tr("run_qr"), MAX_PHOTO_MB, "qr",
                    lambda d: guarded(run_qr, "working_qr", d, engine))
        if st.session_state["qr"] is not None:
            show_qr(st.session_state["qr"])

with tab_pdf:
    st.write(tr("pdf_intro"))
    input_block("pdf", tr("upload_pdf", pages=MAX_PAGES, mb=MAX_PDF_MB), None, tr("run_pdf"), MAX_PDF_MB, "pdf",
                lambda d: guarded_pdf(d, engine))
    if st.session_state["pdf"] is not None:
        show_pdf(st.session_state["pdf"])

st.divider()
f1, f2, f3 = st.columns(3)
with f1:
    with st.expander(tr("privacy_h")):
        st.write(tr("privacy"))
with f2:
    with st.expander(tr("limits_h")):
        st.write(tr("limits", photo_mb=MAX_PHOTO_MB, mp=MAX_MEGAPIXELS, pages=MAX_PAGES, pdf_mb=MAX_PDF_MB))
with f3:
    with st.expander(tr("credits_h")):
        st.write(tr("credits"))
