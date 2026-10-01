import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

INTER_FILE = DATA_DIR / "interactions.csv"
MODEL_FILE = DATA_DIR / "lightgcn_model.pth"
INFO_FILE = DATA_DIR / "lightgcn_info.pkl"

EPOCHS = 20
BATCH_SIZE = 1024
EMB_DIM = 64
LR = 0.001
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        pos = self.item_emb(pos_items)
        neg = self.item_emb(neg_items)

        pos_score = torch.sum(u * pos, dim=1)
        neg_score = torch.sum(u * neg, dim=1)

        loss = -torch.mean(
            torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
        )

        return loss

def main():
    print("Start train_lightgcn.py")

    if not INTER_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INTER_FILE}")

    interactions = pd.read_csv(INTER_FILE)

    print("Loaded interactions:", interactions.shape)
    print(interactions.head())

    num_users = int(interactions["user_id"].max()) + 1
    num_items = int(interactions["item_id"].max()) + 1

    print("num_users:", num_users)
    print("num_items:", num_items)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    user_item = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        emb_dim=EMB_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        train_data = interactions.sample(
            frac=1,
            random_state=RANDOM_SEED + epoch
        ).reset_index(drop=True)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for start in tqdm(range(0, len(train_data), BATCH_SIZE)):
            batch = train_data.iloc[start:start + BATCH_SIZE]

            users_np = batch["user_id"].values.astype(np.int64)
            pos_np = batch["item_id"].values.astype(np.int64)

            neg_np = []

            for u in users_np:
                neg = np.random.randint(0, num_items)

                while neg in user_item.get(int(u), set()):
                    neg = np.random.randint(0, num_items)

                neg_np.append(neg)

            users = torch.LongTensor(users_np).to(device)
            pos_items = torch.LongTensor(pos_np).to(device)
            neg_items = torch.LongTensor(neg_np).to(device)

            loss = model(users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_FILE)

    with open(INFO_FILE, "wb") as f:
        pickle.dump(
            {
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": EMB_DIM
            },
            f
        )

    print("Saved model:", MODEL_FILE)
    print("Saved info:", INFO_FILE)

if __name__ == "__main__":
    main()