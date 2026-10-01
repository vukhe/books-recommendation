import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

BOOK_FILE = DATA_DIR / "books_processed.csv"
EMB_FILE = DATA_DIR / "content_embeddings.npy"
INDEX_FILE = DATA_DIR / "faiss_content.index"
MODEL_NAME_FILE = DATA_DIR / "sbert_model_name.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def main():
    print("Start train_content.py")

    if not BOOK_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {BOOK_FILE}")

    df = pd.read_csv(BOOK_FILE)
    print("Loaded books:", df.shape)

    required_cols = ["title", "author", "genres", "description"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("")
    df["author"] = df["author"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["description"] = df["description"].fillna("")

    df["content"] = (
        df["title"].astype(str) + " " +
        df["author"].astype(str) + " " +
        df["genres"].astype(str) + " " +
        df["description"].astype(str)
    )

    texts = df["content"].tolist()

    print("Loading Sentence-BERT model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    all_embeddings = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print("Start encoding...")
    print("Total books:", len(texts))
    print("Total batches:", total_batches)

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding batches"):
        batch_texts = texts[i:i + BATCH_SIZE]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")

    print("Embedding shape:", embeddings.shape)

    print("Saving embeddings...")
    np.save(EMB_FILE, embeddings)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with open(MODEL_NAME_FILE, "wb") as f:
        pickle.dump(MODEL_NAME, f)

    print("Done train_content.py")
    print("Saved:", EMB_FILE)
    print("Saved:", INDEX_FILE)
    print("Saved:", MODEL_NAME_FILE)

if __name__ == "__main__":
    main()