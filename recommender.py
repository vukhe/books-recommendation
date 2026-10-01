import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import faiss
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

BOOK_FILE = DATA_DIR / "books_processed.csv"
INTER_FILE = DATA_DIR / "interactions.csv"

LIGHTGCN_MODEL = DATA_DIR / "lightgcn_model.pth"
LIGHTGCN_INFO = DATA_DIR / "lightgcn_info.pkl"

CONTENT_EMB = DATA_DIR / "content_embeddings.npy"
FAISS_INDEX = DATA_DIR / "faiss_content.index"


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()

        self.user_emb = nn.Embedding(
            num_users,
            emb_dim
        )

        self.item_emb = nn.Embedding(
            num_items,
            emb_dim
        )


class HybridRecommender:

    def __init__(self):

        print("Loading data...")

        self.books = pd.read_csv(BOOK_FILE)

        self.interactions = pd.read_csv(
            INTER_FILE
        )

        with open(
            LIGHTGCN_INFO,
            "rb"
        ) as f:

            info = pickle.load(f)

        self.num_users = info["num_users"]
        self.num_items = info["num_items"]
        self.emb_dim = info["emb_dim"]

        print("Loading LightGCN...")

        self.cf_model = LightGCN(
            self.num_users,
            self.num_items,
            self.emb_dim
        )

        self.cf_model.load_state_dict(
            torch.load(
                LIGHTGCN_MODEL,
                map_location="cpu"
            )
        )

        self.cf_model.eval()

        print("Loading embeddings...")

        self.content_embeddings = np.load(
            CONTENT_EMB
        )

        self.faiss_index = faiss.read_index(
            str(FAISS_INDEX)
        )

        print("Done.")

    def content_recommend(
        self,
        book_id,
        top_k=10
    ):

        rows = self.books[
            self.books["bookId"] == book_id
        ]

        if len(rows) == 0:
            return pd.DataFrame()

        idx = rows.index[0]

        query = self.content_embeddings[
            idx
        ].reshape(1, -1)

        scores, ids = self.faiss_index.search(
            query,
            top_k + 1
        )

        ids = ids[0]

        ids = [
            i
            for i in ids
            if i != idx
        ]

        return self.books.iloc[
            ids[:top_k]
        ]

    def cf_recommend(
        self,
        user_id,
        top_k=10
    ):

        if user_id >= self.num_users:
            return pd.DataFrame()

        with torch.no_grad():

            user_emb = self.cf_model.user_emb(
                torch.LongTensor(
                    [user_id]
                )
            )

            item_emb = (
                self.cf_model
                .item_emb
                .weight
            )

            scores = torch.matmul(
                user_emb,
                item_emb.T
            ).squeeze()

            scores = scores.numpy()

        seen = self.interactions[
            self.interactions["user_id"]
            == user_id
        ]["item_id"].tolist()

        scores[seen] = -9999

        top_items = np.argsort(
            scores
        )[::-1][:top_k]

        return self.books[
            self.books["item_id"]
            .isin(top_items)
        ]

    def hybrid_recommend(
        self,
        user_id,
        book_id,
        top_k=10
    ):

        rows = self.books[
            self.books["bookId"] == book_id
        ]

        if len(rows) == 0:
            return pd.DataFrame()

        idx = rows.index[0]

        query = self.content_embeddings[
            idx
        ].reshape(1, -1)

        content_scores, _ = (
            self.faiss_index.search(
                query,
                len(self.books)
            )
        )

        content_scores = (
            content_scores[0]
        )

        with torch.no_grad():

            user_emb = self.cf_model.user_emb(
                torch.LongTensor(
                    [user_id]
                )
            )

            item_emb = (
                self.cf_model
                .item_emb
                .weight
            )

            cf_scores = torch.matmul(
                user_emb,
                item_emb.T
            ).squeeze()

            cf_scores = cf_scores.numpy()

        cf_scores = (
            cf_scores
            - cf_scores.min()
        ) / (
            cf_scores.max()
            - cf_scores.min()
            + 1e-8
        )

        user_interactions = len(
            self.interactions[
                self.interactions[
                    "user_id"
                ]
                == user_id
            ]
        )

        if user_interactions < 5:

            final_score = (
                0.8 * content_scores
                + 0.2 * cf_scores
            )

        else:

            final_score = (
                0.4 * content_scores
                + 0.6 * cf_scores
            )

        final_score[idx] = -9999

        top_ids = np.argsort(
            final_score
        )[::-1][:top_k]

        return self.books.iloc[
            top_ids
        ]