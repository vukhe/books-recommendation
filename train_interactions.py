import pandas as pd
import numpy as np
import random

df = pd.read_csv("../data/books_processed.csv")

df = df.dropna(subset=["bookId", "title", "genres"])
df = df.drop_duplicates(subset=["bookId"]).reset_index(drop=True)

df["item_id"] = range(len(df))

num_users = 5000
interactions = []

for user_id in range(num_users):
    liked_genre = random.choice(df["genres"].dropna().sample(100).tolist())

    candidates = df[
        df["genres"].astype(str).str.contains(
            liked_genre.split(",")[0].replace("[", "").replace("'", ""),
            case=False,
            na=False
        )
    ]

    if len(candidates) < 10:
        candidates = df.sample(50)

    sampled = candidates.sample(
        min(random.randint(10, 30), len(candidates))
    )

    for _, row in sampled.iterrows():
        interactions.append({
            "user_id": user_id,
            "item_id": row["item_id"],
            "bookId": row["bookId"],
            "rating": random.choice([4, 5])
        })

inter_df = pd.DataFrame(interactions)

df.to_csv("../data/books_processed.csv", index=False, encoding="utf-8-sig")
inter_df.to_csv("../data/interactions.csv", index=False, encoding="utf-8-sig")

print("Books:", len(df))
print("Interactions:", len(inter_df))