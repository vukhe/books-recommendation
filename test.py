from recommender import HybridRecommender

print("Loading recommender...")

rec = HybridRecommender()

print("Loaded!")

book_id = rec.books.iloc[0]["bookId"]

print("Book ID:", book_id)

result = rec.hybrid_recommend(
    user_id=1,
    book_id=book_id,
    top_k=5
)

print(result[[
    "title",
    "author",
    "rating"
]])