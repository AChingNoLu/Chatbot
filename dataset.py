import os
import json
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f" Đã load {len(data)} chunk từ {json_path}")
    return data


# ============================================================
# 2. GHÉP TEXT ĐƯA VÀO MODEL
# ============================================================

def build_embedding_text(chunk):
    """
    Ghép thêm ngữ cảnh (tên văn bản, chương, điều) vào trước nội dung
    trước khi encode -> giúp model hiểu chunk thuộc văn bản/chương nào,
    tăng chất lượng retrieval so với chỉ encode content thuần.
    """
    parts = []
    if chunk.get("ten_van_ban"):
        parts.append(chunk["ten_van_ban"])
    if chunk.get("chuong"):
        parts.append(chunk["chuong"])
    parts.append(chunk["content"])
    return "\n".join(parts)


# ============================================================
# 3. SINH EMBEDDING THEO BATCH (tránh tràn RAM/VRAM)
# ============================================================

def embed_chunks(data, model, batch_size=8, max_length=2048):
    texts = [build_embedding_text(c) for c in data]
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Đang tạo embedding"):
        batch = texts[i:i + batch_size]
        result = model.encode(
            batch,
            batch_size=batch_size,
            max_length=max_length,
        )
        all_embeddings.append(result["dense_vecs"])

    return np.vstack(all_embeddings)


# ============================================================
# 4. LƯU KẾT QUẢ — TÁCH RIÊNG EMBEDDING (.npy) VÀ METADATA (.json)
#    Lý do tách riêng: file .npy load nhanh hơn nhiều so với nhúng
#    embedding (mảng float) trực tiếp vào JSON.
# ============================================================

def save_embeddings(data, embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    meta_path = os.path.join(output_dir, "metadata.json")

    np.save(emb_path, embeddings)

    # Bỏ trường "content" dài ra khỏi metadata riêng nếu muốn nhẹ hơn,
    # nhưng ở đây giữ nguyên để tiện debug/retrieval trả về text gốc.
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu embeddings: {emb_path}  shape={embeddings.shape}")
    print(f"Đã lưu metadata:   {meta_path}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    # --- Sửa đường dẫn cho khớp máy bạn ---
    input_json = r"C:\Users\Admin\Downloads\VBPLHN\dataset_rag_final.json"
    output_dir = r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output"
    # ----------------------------------------

    data = load_dataset(input_json)

    print(" Đang load model bge-m3 (lần đầu sẽ tự tải về, mất vài phút)...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True
    )  # use_fp16=False nếu chạy CPU

    embeddings = embed_chunks(data, model, batch_size=8)

    save_embeddings(data, embeddings, output_dir)


if __name__ == "__main__":
    main()
