import os
import re
import json
import glob
from docx import Document


# ============================================================
# 1. TRÍCH XUẤT TEXT TỪ DOCX
# ============================================================

def extract_vbhn_elements(path):
    """Đọc file docx và trích xuất text cùng thông tin bold/in đậm."""
    doc = Document(path)
    elements = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        is_bold = any(run.bold for run in para.runs if run.text.strip())
        elements.append({"text": text, "bold": is_bold})
    return elements


# ============================================================
# 2. METADATA CẤP VĂN BẢN
# ============================================================

def extract_doc_level_meta(elements):
    """Trích xuất metadata tổng thể của Văn bản hợp nhất ở các đoạn đầu."""
    meta = {
        "so_hieu_goc": None,
        "ten_van_ban": None,
        "sua_doi_boi": [],
    }
    for el in elements[:25]:
        text = el["text"]

        m = re.search(r"số\s+([\d/A-Za-z]+)", text, re.IGNORECASE)
        if not meta["so_hieu_goc"] and re.search(r"(Bộ luật|Luật|Nghị định|Thông tư).*số\s+[\d/]", text):
            if m:
                meta["so_hieu_goc"] = m.group(1)

        if not meta["ten_van_ban"] and el["bold"] and len(text) > 3:
            meta["ten_van_ban"] = text

        if re.match(r"^\d+\.\s+(Luật|Nghị định|Thông tư)", text):
            meta["sua_doi_boi"].append(text)

    return meta


# ============================================================
# 3. PHÂN BIỆT ĐIỀU CẤP CAO NHẤT vs ĐIỀU TRÍCH DẪN LỒNG
# ============================================================

def is_top_level_dieu(text):
    """
    Trích dẫn lồng (vd Điều 219 sửa đổi Luật BHXH, trích nguyên văn
    "Điều 54. ..." của luật đó) luôn mở đầu bằng ngoặc kép kiểu Việt
    " (U+201C) hoặc ngoặc kép thường ". Loại các trường hợp này ra
    để không gán nhầm nội dung luật khác vào văn bản đang parse.
    """
    if not re.match(r"^Điều\s+\d+\.", text):
        return False
    if text.startswith(chr(0x201C)) or text.startswith('"'):
        return False
    return True


# ============================================================
# 4. CHUNK THEO CHƯƠNG -> ĐIỀU
# ============================================================

def parse_vbhn_to_chunks(elements, file_info, doc_meta, uu_tien=True):
    """Phân đoạn văn bản thành các chunk theo Chương -> Điều (chỉ Điều cấp cao nhất)."""
    chunks = []
    current_chuong = None
    current_dieu = None
    buffer = []

    def flush():
        if current_dieu and buffer:
            chunk_content = f"{current_dieu}\n" + "\n".join(buffer)
            match = re.match(r"Điều\s+(\d+)\.", current_dieu)
            dieu_so = match.group(1) if match else "unknown"
            chunks.append({
                "chunk_id": f"{file_info['doc_name']}_Dieu_{dieu_so}",
                "doc_name": file_info["doc_name"],
                "doc_type": "luat_hop_nhat",
                "so_hieu_goc": doc_meta["so_hieu_goc"],
                "ten_van_ban": doc_meta["ten_van_ban"],
                "cac_luat_sua_doi": doc_meta["sua_doi_boi"],
                "chuong": current_chuong,
                "article": current_dieu,
                "content": chunk_content,
                "uu_tien": uu_tien,   # VBHN luôn được ưu tiên khi trùng nội dung với nghị định gốc
            })

    for el in elements:
        text = el["text"]

        # Tiêu đề Chương (vd: Chương II, Chương XI)
        if el["bold"] and re.match(r"^Chương\s+[IVXLCDM]+", text):
            flush()
            current_chuong = text
            current_dieu = None   # reset — tránh gán nội dung mồ côi vào Điều cũ
            buffer = []

        # Điều cấp cao nhất (loại trừ trích dẫn lồng)
        elif el["bold"] and is_top_level_dieu(text):
            flush()
            current_dieu = text
            buffer = []

        # Nội dung thường — kể cả "Điều X." bị trích dẫn lồng cũng rơi vào đây
        else:
            if current_dieu is not None:
                buffer.append(text)
            # current_dieu is None -> đang ở đoạn mở đầu/tiêu đề Chương, bỏ qua

    flush()
    return chunks


# ============================================================
# 5. XỬ LÝ 1 FILE
# ============================================================

def process_vbhn_file(file_path):
    """Xử lý 1 file VBHN docx, trả về (metadata, list_chunks)."""
    doc_name = os.path.basename(file_path)
    file_info = {"doc_name": doc_name}

    elements = extract_vbhn_elements(file_path)
    doc_meta = extract_doc_level_meta(elements)
    chunks = parse_vbhn_to_chunks(elements, file_info, doc_meta)

    return doc_meta, chunks


# ============================================================
# 6. GỘP VỚI DATASET PDF (nếu có), LOẠI TRÙNG THEO so_hieu_goc
# ============================================================

def merge_with_pdf_dataset(vbhn_chunks, pdf_json_path, output_path):
    """
    Gộp chunk từ VBHN (docx) với dataset đã có từ PDF (dataset_rag_full.json).
    Nguyên tắc: nếu 1 nghị định/luật gốc trong dataset PDF đã có bản VBHN
    hợp nhất tương ứng (dựa vào so_hieu_goc trùng số hiệu văn bản gốc),
    thì loại bỏ chunk PDF gốc đó ra khỏi kết quả cuối — ưu tiên VBHN vì
    đã tính hết sửa đổi/bổ sung, tránh chatbot trả lời theo bản đã lỗi thời.
    """
    if not os.path.exists(pdf_json_path):
        print(f"⚠️ Không tìm thấy {pdf_json_path}, chỉ ghi riêng dữ liệu VBHN.")
        final_chunks = vbhn_chunks
    else:
        with open(pdf_json_path, "r", encoding="utf-8") as f:
            pdf_chunks = json.load(f)

        # Tập số hiệu văn bản gốc đã có bản VBHN thay thế
        so_hieu_da_hop_nhat = {
            c["so_hieu_goc"] for c in vbhn_chunks if c.get("so_hieu_goc")
        }

        def bi_thay_the(pdf_chunk):
            doc_name = pdf_chunk.get("doc_name", "")
            for so_hieu in so_hieu_da_hop_nhat:
                # so khớp lỏng: các chữ số của số hiệu văn bản gốc có xuất hiện trong tên file không
                so_don_gian = re.sub(r"[^0-9]", "", so_hieu)
                if so_don_gian and so_don_gian in re.sub(r"[^0-9]", "", doc_name):
                    return True
            return False

        pdf_chunks_loc = [c for c in pdf_chunks if not bi_thay_the(c)]
        loai_bo = len(pdf_chunks) - len(pdf_chunks_loc)
        print(f"🗑️ Loại {loai_bo} chunk PDF trùng với văn bản đã có bản VBHN hợp nhất.")

        final_chunks = pdf_chunks_loc + vbhn_chunks

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)

    print(f"✅ Tổng số chunk cuối cùng: {len(final_chunks)}")
    print(f"File đầu ra: {os.path.abspath(output_path)}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    # --- Sửa 3 đường dẫn này cho khớp máy bạn ---
    base_dir = r"C:\Users\Admin\Downloads\VBPLHN"
    input_dir = base_dir
    pdf_json_path = os.path.join(base_dir, "dataset_rag_full.json")
    output_path = os.path.join(base_dir, "dataset_rag_final.json")
    # ---------------------------------------------

    docx_files = glob.glob(os.path.join(input_dir, "**", "*.docx"), recursive=True)
    print(f"🔍 Tìm thấy {len(docx_files)} file .docx trong thư mục.")

    all_vbhn_chunks = []
    for idx, file_path in enumerate(docx_files, 1):
        file_name = os.path.basename(file_path)
        print(f"[{idx}/{len(docx_files)}] Đang xử lý: {file_name}")
        try:
            doc_meta, chunks = process_vbhn_file(file_path)
            all_vbhn_chunks.extend(chunks)
            print(f"    -> {len(chunks)} chunk, sửa đổi bởi {len(doc_meta['sua_doi_boi'])} văn bản")
        except Exception as e:
            print(f"    ❌ Lỗi khi xử lý {file_name}: {e}")

    print(f"\n📦 Tổng chunk từ VBHN docx: {len(all_vbhn_chunks)}")
    merge_with_pdf_dataset(all_vbhn_chunks, pdf_json_path, output_path)


if __name__ == "__main__":
    main()