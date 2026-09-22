import os
import json
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


# =========================
# CẤU HÌNH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(
    BASE_DIR,
    "dataset_rag_final.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "embeddings_output"
)


# =========================
# LOAD DATASET
# =========================

def load_dataset(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Đã load {len(data)} chunk"
    )

    return data


# =========================
# TẠO TEXT ĐỂ EMBEDDING
# =========================

def build_embedding_text(chunk):

    parts = []

    if chunk.get("loai_van_ban"):
        parts.append(
            "Loại văn bản: " +
            chunk["loai_van_ban"]
        )

    if chunk.get("so_hieu"):
        parts.append(
            "Số hiệu: " +
            chunk["so_hieu"]
        )

    if chunk.get("ten_van_ban"):
        parts.append(
            "Tên văn bản: " +
            chunk["ten_van_ban"]
        )

    if chunk.get("ngay_co_hieu_luc"):
        parts.append(
            "Ngày có hiệu lực: " +
            chunk["ngay_co_hieu_luc"]
        )

    if chunk.get("hieu_luc"):
        parts.append(
            "Hiệu lực: " +
            chunk["hieu_luc"]
        )

    if chunk.get("dieu"):
        parts.append(
            chunk["dieu"]
        )

    parts.append(
        chunk.get("content", "")
    )

    return "\n".join(parts)


# =========================
# TẠO EMBEDDING
# =========================

def embed_chunks(
    data,
    model,
    batch_size=8
):

    texts = []

    for chunk in data:
        texts.append(
            build_embedding_text(chunk)
        )

    all_embeddings = []

    for i in tqdm(
        range(
            0,
            len(texts),
            batch_size
        ),
        desc="Đang tạo embedding"
    ):

        batch = texts[
            i:i + batch_size
        ]

        result = model.encode(
            batch,
            batch_size=batch_size,
            max_length=2048
        )

        all_embeddings.append(
            result["dense_vecs"]
        )

    return np.vstack(
        all_embeddings
    )


# =========================
# LƯU KẾT QUẢ
# =========================

def save_embeddings(
    data,
    embeddings
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    embedding_file = os.path.join(
        OUTPUT_DIR,
        "embeddings.npy"
    )

    metadata_file = os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    )

    np.save(
        embedding_file,
        embeddings
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nĐã lưu:")
    print(
        embedding_file
    )

    print(
        metadata_file
    )

    print(
        "Shape:",
        embeddings.shape
    )


# =========================
# MAIN
# =========================

def main():

    data = load_dataset(
        JSON_PATH
    )

    print(
        "Đang load BGE-M3..."
    )

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False
    )

    embeddings = embed_chunks(
        data,
        model,
        batch_size=8
    )

    save_embeddings(
        data,
        embeddings
    )


if __name__ == "__main__":
    main()