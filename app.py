"""Combined test build of the demo: Module 1 (camera photos) and Modules 2+3 (scanned land PDFs) in one app.

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

# Module 1 (photos)
MAX_PHOTO_MB = 15
MAX_MEGAPIXELS = 30
MIN_SIDE = 50
SHOW_MAX_SIDE = 1400
# Modules 2+3 (PDF)
HARD_MAX_PAGES = 8
MAX_PDF_MB = 30
ZOOM = 2.0


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
    t = time.perf_counter()
    from module_2_core_ocr.config import OcrConfig
    from module_2_core_ocr.ocr_pipeline import OcrPipeline

    cfg = OcrConfig.from_yaml(ROOT / "configs" / "module2_defaults.yaml")
    pipe = OcrPipeline(cfg.active_engine, cfg)
    catalog = json.loads((ROOT / "module_3_dynamic_ner" / "configs" / "document_catalog.json").read_text(encoding="utf-8"))
    t_ocr = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    from module_1_image_preprocessing.preprocessor import ImagePreprocessor

    pre = ImagePreprocessor.from_yaml(ROOT / "configs" / "module1_defaults.yaml")
    t_m1 = round(time.perf_counter() - t, 1)
    return {"pipe": pipe, "catalog": catalog, "pre": pre, "lock": threading.Lock(), "load_ocr_s": t_ocr, "load_m1_s": t_m1}


# ---------- synthetic examples (no personal data) ----------

def make_synthetic_photo(kind, tilt_deg, width=3000, height=4000):
    """A drawn photo of a page on a dark table."""
    import cv2
    import numpy as np

    img = np.full((height, width, 3), (60, 70, 85), np.uint8)
    if kind == "No paper (plain table)":
        return img
    page = np.full((int(height * 0.8), int(width * 0.8), 3), 245, np.uint8)
    if kind != "Blank page":
        for i in range(30):
            cv2.putText(page, f"Line {i + 1}: Giay chung nhan quyen su dung dat so {1000 + i * 7}", (60, 110 + i * 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    ph, pw = page.shape[:2]
    y0, x0 = (height - ph) // 2, (width - pw) // 2
    img[y0:y0 + ph, x0:x0 + pw] = page
    m = cv2.getRotationMatrix2D((width / 2, height / 2), tilt_deg, 1.0)
    return cv2.warpAffine(img, m, (width, height), borderValue=(60, 70, 85))


def make_synthetic_pdf(n_pages, land_like=True):
    """A drawn landscape A4 PDF with 28 text lines per page."""
    import fitz

    doc = fitz.open()
    for p in range(n_pages):
        page = doc.new_page(width=842, height=595)
        for i in range(28):
            if land_like:
                text = f"Page {p + 1} line {i + 1}: Giay chung nhan quyen su dung dat so {1000 + i * 7} ngay 12/05/2015"
            else:
                text = f"Page {p + 1} line {i + 1}: the weather report for the coast is calm with light winds and clear skies"
            page.insert_text((30, 30 + i * 19), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


# ---------- Module 1: photo ----------

def to_jpeg(bgr):
    """Reduced JPEG for the screen, so the page stays light."""
    import cv2

    h, w = bgr.shape[:2]
    s = SHOW_MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def run_module1(data, eng):
    """Image bytes -> Module 1, all in memory. Returns a dict."""
    import cv2
    import numpy as np
    from PIL import Image

    t_all = time.perf_counter()
    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:  # noqa: BLE001
        return {"error": "bad_image"}
    if w * h > MAX_MEGAPIXELS * 1_000_000:
        return {"error": "too_big", "mp": round(w * h / 1e6, 1)}
    if min(w, h) < MIN_SIDE:
        return {"error": "too_small"}
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "bad_image"}
    t = time.perf_counter()
    res = eng["pre"].process(img)
    t_m1 = time.perf_counter() - t
    out = {
        "before": to_jpeg(img),
        "after": to_jpeg(res.processed_image) if res.processed_image is not None else None,
        "is_blank": res.is_blank,
        "orientation": res.orientation_status.value,
        "warnings": list(res.warnings),
        "barcodes": [{"type": b.barcode_type, "data": b.data} for b in res.barcodes],
        "error_code": res.error_code,
        "stats": {
            "size_px": f"{w}x{h}", "megapixels": round(w * h / 1e6, 1), "skew_angle": round(float(res.skew_angle), 2),
            "module1_s": round(t_m1, 2), "total_s": round(time.perf_counter() - t_all, 2),
            "rss_now_mb": rss_mb(), "peak_rss_mb": peak_mb(), "files_written": 0,
        },
    }
    del img, res
    gc.collect()
    return out


def show_module1(r):
    err = r.get("error")
    if err == "bad_image":
        st.error("Không đọc được file này. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.")
        return
    if err == "too_big":
        st.error(f"Ảnh {r['mp']} MP, vượt giới hạn {MAX_MEGAPIXELS} MP. Hãy chụp ở độ phân giải thấp hơn.")
        return
    if err == "too_small":
        st.error(f"Ảnh quá nhỏ (cạnh ngắn dưới {MIN_SIDE} px).")
        return
    if r.get("error_code"):
        st.error(f"Module 1 báo lỗi: {r['error_code']}")
    if "CROP_FAILED_PLEASE_RETAKE" in r["warnings"]:
        st.warning("Không tìm thấy rõ tờ giấy trong ảnh nên giữ nguyên ảnh gốc. Hãy chụp lại: tờ giấy nằm trọn trong khung, nền tương phản.")
    if r["is_blank"]:
        st.warning("Trang này có vẻ trống.")
    if r["orientation"] != "LIKELY_CORRECT":
        st.info(f"Hướng chữ: {r['orientation']} (chưa chắc là đúng chiều).")
    if r["barcodes"]:
        st.success(f"Đọc được {len(r['barcodes'])} mã vạch/QR.")
        st.table(r["barcodes"])
    c1, c2 = st.columns(2)
    c1.caption("Before")
    c1.image(r["before"])
    c2.caption("After (cropped and straightened)")
    if r["after"]:
        c2.image(r["after"])
    with st.expander("Diagnostics"):
        st.json(r["stats"])


# ---------- Modules 2+3: PDF ----------

def run_pipeline(data, max_pages, eng, progress):
    """PDF bytes -> page images -> Module 2 -> Module 3. Returns a dict; never keeps files."""
    import fitz
    from module_3_dynamic_ner.orchestrator.session_manager import SessionManager

    t_all = time.perf_counter()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:  # noqa: BLE001
        return {"error": "bad_pdf"}
    n = doc.page_count
    if n > max_pages:
        doc.close()
        return {"error": "too_many_pages", "pages": n}

    stream, boxes, t_render, t_ocr = [], 0, 0.0, 0.0
    with tempfile.TemporaryDirectory(prefix="demo_") as work:
        work = Path(work)
        for i in range(n):
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
            del res, r
            shutil.rmtree(pdir)
            progress.progress((i + 1) / n, text=f"Page {i + 1} of {n} done")
        doc.close()
        t = time.perf_counter()
        records = SessionManager(eng["catalog"]).process_document_stream(stream)
        t_m3 = time.perf_counter() - t
        work_left_before_exit = len(list(work.iterdir()))
    total = time.perf_counter() - t_all
    gc.collect()
    return {
        "records": records,
        "stats": {
            "pages": n, "ocr_boxes": boxes, "render_s": round(t_render, 1), "ocr_s": round(t_ocr, 1),
            "module3_s": round(t_m3, 2), "total_s": round(total, 1), "per_page_s": round(total / n, 1),
            "rss_now_mb": rss_mb(), "peak_rss_mb": peak_mb(), "page_files_left": work_left_before_exit,
            "work_dir_deleted": not work.exists(),
        },
    }


def show_pdf(result, max_pages):
    if result.get("error") == "bad_pdf":
        st.error("Không đọc được file này. Hãy chọn một file PDF hợp lệ.")
        return
    if result.get("error") == "too_many_pages":
        st.error(f"PDF có {result['pages']} trang, vượt giới hạn {max_pages} trang. Hãy cắt bớt rồi thử lại.")
        return
    records = result["records"]
    if not records:
        st.warning("Không nhận ra loại tài liệu nào trong PDF này. Hệ thống chỉ nhận dạng hồ sơ đất đai thuộc các loại đã hỗ trợ, nên không có bản ghi nào để hiển thị.")
    else:
        import pandas as pd

        rows = [{k: (v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=False)) for k, v in rec.items()} for rec in records]
        st.success(f"Tìm thấy {len(records)} bản ghi.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("Diagnostics"):
        st.json(result["stats"])


# ---------- shared ----------

def busy_message():
    st.warning("Đang xử lý cho một người dùng khác. Vui lòng thử lại sau ít phút.")


def guarded_photo(data, eng):
    """One visitor at a time (one lock for both tabs); the lock is always released."""
    if not eng["lock"].acquire(blocking=False):
        busy_message()
        return None
    try:
        with st.spinner("Processing the photo..."):
            return run_module1(data, eng)
    finally:
        eng["lock"].release()


def guarded_pdf(data, max_pages, eng):
    if not eng["lock"].acquire(blocking=False):
        busy_message()
        return None
    try:
        bar = st.progress(0.0, text="Starting...")
        result = run_pipeline(data, max_pages, eng, bar)
        bar.empty()
        return result
    finally:
        eng["lock"].release()


CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #0f2b57 0%, #1f6feb 100%); color: #fff; border-radius: 14px;
       padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2rem; line-height: 1.2;}
.hero p {color: #dbe7ff; margin: 0; font-size: 1.02rem;}
.chips {margin-top: .9rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px; padding: .15rem .8rem;
       margin: 0 .4rem .3rem 0; font-size: .85rem; color: #fff;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600;}
</style>
"""

st.set_page_config(page_title="VN-Digitize AI Engine - live demo", page_icon="📄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
  <h1>VN-Digitize AI Engine</h1>
  <p>From a camera photo or a scanned land-record PDF to clean images and structured records.</p>
  <div class="chips"><span class="chip">Module 1 · photo clean-up</span><span class="chip">Module 2 · OCR</span>
  <span class="chip">Module 3 · record extraction</span><span class="chip">CPU only</span></div>
</div>
""",
    unsafe_allow_html=True,
)

engine = load_engines()

with st.sidebar:
    st.header("Settings")
    max_pages = st.slider("PDF page limit", 1, HARD_MAX_PAGES, 5, help="A page takes about 12 s on the free host.")
    st.header("Privacy")
    st.caption("Test build: please do not upload photos or documents with personal data. Photos are processed in memory; PDF pages go to a temporary folder that is deleted afterwards. Nothing is stored.")
    st.header("Status")
    st.caption(f"OCR models loaded in {engine['load_ocr_s']} s, Module 1 in {engine['load_m1_s']} s.")
    st.caption(f"Memory now {rss_mb()} MB, peak {peak_mb()} MB (limit about 3000 MB). One visitor at a time.")

for key, default in (("m1_last", None), ("m1_hist", []), ("pdf_last", None), ("pdf_hist", [])):
    st.session_state.setdefault(key, default)

tab_photo, tab_pdf = st.tabs(["📷 Camera photo → Module 1", "📄 Land-record PDF → Modules 2 + 3"])

with tab_photo:
    st.write("Finds the page in a camera photo, crops and straightens it, flags blank pages and reads QR codes.")
    up = st.file_uploader(f"Photo of a document (JPG or PNG, up to {MAX_PHOTO_MB} MB)", type=["jpg", "jpeg", "png"], key="photo_up")
    if up is not None and st.button("Run Module 1 on the photo", type="primary"):
        if up.size > MAX_PHOTO_MB * 1024 * 1024:
            st.session_state["m1_last"] = None
            st.error(f"File lớn hơn {MAX_PHOTO_MB} MB.")
        else:
            res = guarded_photo(up.getvalue(), engine)
            if res is not None:
                st.session_state["m1_last"], st.session_state["m1_hist"] = res, []
    with st.expander("No photo at hand? Try a synthetic one"):
        c1, c2, c3 = st.columns(3)
        kind = c1.radio("Photo", ["Tilted page", "Blank page", "No paper (plain table)"])
        tilt = c2.slider("Tilt (degrees)", -25, 25, 12)
        repeats = c3.number_input("Repeat runs", 1, 10, 1, key="photo_repeats")
        if st.button("Run synthetic photo"):
            import cv2

            ok, buf = cv2.imencode(".jpg", make_synthetic_photo(kind, tilt), [cv2.IMWRITE_JPEG_QUALITY, 90])
            data, history, last = buf.tobytes(), [], None
            for k in range(int(repeats)):
                last = guarded_photo(data, engine)
                if last is None:
                    break
                if "stats" in last:
                    history.append({"run": k + 1, **{x: last["stats"][x] for x in ("megapixels", "skew_angle", "module1_s", "total_s", "rss_now_mb", "peak_rss_mb")}})
            if last is not None:
                st.session_state["m1_last"], st.session_state["m1_hist"] = last, history
    if st.session_state["m1_hist"]:
        st.table(st.session_state["m1_hist"])
    if st.session_state["m1_last"] is not None:
        show_module1(st.session_state["m1_last"])

with tab_pdf:
    st.write("Reads a scanned land-record PDF page by page (OCR), then splits it into documents and extracts the fields of each record.")
    up = st.file_uploader(f"Scanned land-document PDF (up to {MAX_PDF_MB} MB; at most the page limit in the sidebar)", type=["pdf"], key="pdf_up")
    if up is not None and st.button("Run OCR and extraction on the PDF", type="primary"):
        if up.size > MAX_PDF_MB * 1024 * 1024:
            st.session_state["pdf_last"] = None
            st.error(f"File lớn hơn {MAX_PDF_MB} MB.")
        else:
            res = guarded_pdf(up.getvalue(), max_pages, engine)
            if res is not None:
                st.session_state["pdf_last"], st.session_state["pdf_hist"] = res, []
    with st.expander("No PDF at hand? Try a synthetic one"):
        st.write("Builds a drawn PDF with no personal data. Land-like text may give a record; unrelated text gives 0 records and shows the unrecognized-document message.")
        c1, c2, c3 = st.columns(3)
        n_pages = c1.number_input("Pages", 1, HARD_MAX_PAGES, 3)
        repeats = c2.number_input("Repeat runs", 1, 10, 1, key="pdf_repeats")
        text_kind = c3.radio("Text", ["Land-like text", "Unrelated text"])
        if st.button("Run synthetic PDF"):
            data, history, last = make_synthetic_pdf(int(n_pages), text_kind == "Land-like text"), [], None
            for k in range(int(repeats)):
                last = guarded_pdf(data, max_pages, engine)
                if last is None:
                    break
                if "stats" in last:
                    history.append({"run": k + 1, **{x: last["stats"][x] for x in ("pages", "total_s", "per_page_s", "rss_now_mb", "peak_rss_mb", "work_dir_deleted")}})
            if last is not None:
                st.session_state["pdf_last"], st.session_state["pdf_hist"] = last, history
    if st.session_state["pdf_hist"]:
        st.table(st.session_state["pdf_hist"])
    if st.session_state["pdf_last"] is not None:
        show_pdf(st.session_state["pdf_last"], max_pages)
