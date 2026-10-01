import streamlit as st
import pandas as pd
import plotly.express as px
from recommender import HybridRecommender

st.set_page_config(
    page_title="Hybrid Book Recommender",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return HybridRecommender()

rec = load_recommender()
books = rec.books.copy()
interactions = rec.interactions.copy()

if "cart" not in st.session_state:
    st.session_state.cart = {}

if "selected_detail_book" not in st.session_state:
    st.session_state.selected_detail_book = None

st.sidebar.title("📚 Menu")

page = st.sidebar.radio(
    "Chọn chức năng",
    [
        "Trang chủ",
        "Gợi ý Hybrid",
        "Content-Based",
        "Collaborative Filtering",
        "Cold Start",
        "🛒 Giỏ hàng",
        "Analytics"
    ]
)

def show_book_card(book):
    cover = book.get("coverImg", "")

    if pd.notna(cover) and str(cover).startswith("http"):
        st.image(cover, use_container_width=True)
    else:
        st.write("Không có ảnh")

    st.markdown(f"**{book.get('title', '')}**")
    st.caption(book.get("author", ""))
    st.write(f"⭐ {book.get('rating', '')}")

def show_book_detail(book):
    st.subheader(book.get("title", ""))

    col1, col2 = st.columns([1, 3])

    with col1:
        cover = book.get("coverImg", "")
        if pd.notna(cover) and str(cover).startswith("http"):
            st.image(cover, use_container_width=True)
        else:
            st.write("Không có ảnh")

    with col2:
        st.write(f"**Tác giả:** {book.get('author', '')}")
        st.write(f"**Rating:** {book.get('rating', '')}")
        st.write(f"**Thể loại:** {book.get('genres', '')}")
        st.write(f"**Book ID:** {book.get('bookId', '')}")

        desc = str(book.get("description", ""))
        st.markdown("### Mô tả")
        st.write(desc if desc and desc != "nan" else "Không có mô tả.")

def show_recommendations(df_result, prefix="book"):
    if df_result is None or len(df_result) == 0:
        st.warning("Không tìm thấy gợi ý.")
        return

    cols = st.columns(5)

    for i, (_, book) in enumerate(df_result.iterrows()):
        with cols[i % 5]:
            show_book_card(book)

            if st.button(
                "📖 Xem chi tiết",
                key=f"{prefix}_{book.get('bookId', i)}_{i}"
            ):
                st.session_state.selected_detail_book = book.to_dict()
                st.rerun()

if st.session_state.selected_detail_book is not None:
    with st.expander("📖 Chi tiết sách", expanded=True):
        show_book_detail(st.session_state.selected_detail_book)

        if st.button("Đóng chi tiết"):
            st.session_state.selected_detail_book = None
            st.rerun()

if page == "Trang chủ":
    st.title("📚 Hybrid Book Recommendation System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Số lượng sách", len(books))

    with col2:
        st.metric("Số lượng interactions", len(interactions))

    with col3:
        st.metric("Số lượng users", interactions["user_id"].nunique())

    st.markdown("""
    ### Mô hình sử dụng

    - **Content-Based Filtering**: Sentence-BERT + FAISS  
    - **Collaborative Filtering**: LightGCN  
    - **Cold Start**: Gợi ý theo thể loại  
    - **Hybrid Recommendation**: Kết hợp Content score và Collaborative score  
    - **Giỏ hàng**: Lưu sách theo từng user_id và gợi ý dựa trên giỏ hàng
    """)

    st.subheader("Một số sách trong dataset")

    sample_books = books.sample(10, random_state=42)
    show_recommendations(sample_books, prefix="home")

elif page == "Gợi ý Hybrid":
    st.title("🔥 Hybrid Recommendation")

    user_id = st.number_input(
        "Nhập user_id",
        min_value=0,
        max_value=int(interactions["user_id"].max()),
        value=1
    )

    search_text = st.text_input("Tìm kiếm sách", "")

    if search_text:
        filtered_books = books[
            books["title"].astype(str).str.contains(
                search_text,
                case=False,
                na=False
            )
        ]
    else:
        filtered_books = books

    if len(filtered_books) == 0:
        st.warning("Không tìm thấy sách.")
    else:
        selected_title = st.selectbox(
            "Chọn sách",
            filtered_books["title"].dropna().tolist()
        )

        selected_book = filtered_books[
            filtered_books["title"] == selected_title
        ].iloc[0]

        col1, col2 = st.columns([1, 3])

        with col1:
            show_book_card(selected_book)

        with col2:
            st.subheader(selected_book["title"])
            st.write(f"**Tác giả:** {selected_book.get('author', '')}")
            st.write(f"**Rating:** {selected_book.get('rating', '')}")
            st.write(f"**Thể loại:** {selected_book.get('genres', '')}")

            desc = str(selected_book.get("description", ""))
            st.write(desc[:700] + "..." if len(desc) > 700 else desc)

        if st.button("Gợi ý sách Hybrid"):
            result = rec.hybrid_recommend(
                user_id=int(user_id),
                book_id=selected_book["bookId"],
                top_k=10
            )

            st.subheader("Kết quả gợi ý Hybrid")
            show_recommendations(result, prefix="hybrid")

elif page == "Content-Based":
    st.title("🧠 Content-Based Recommendation")

    search_text = st.text_input("Tìm kiếm sách", "")

    if search_text:
        filtered_books = books[
            books["title"].astype(str).str.contains(
                search_text,
                case=False,
                na=False
            )
        ]
    else:
        filtered_books = books

    if len(filtered_books) == 0:
        st.warning("Không tìm thấy sách.")
    else:
        selected_title = st.selectbox(
            "Chọn sách để tìm sách tương tự",
            filtered_books["title"].dropna().tolist()
        )

        selected_book = filtered_books[
            filtered_books["title"] == selected_title
        ].iloc[0]

        show_book_card(selected_book)

        if st.button("Gợi ý theo nội dung"):
            result = rec.content_recommend(
                book_id=selected_book["bookId"],
                top_k=10
            )

            st.subheader("Sách tương tự theo nội dung")
            show_recommendations(result, prefix="content")

elif page == "Collaborative Filtering":
    st.title("👥 Collaborative Filtering - LightGCN")

    user_id = st.number_input(
        "Nhập user_id",
        min_value=0,
        max_value=int(interactions["user_id"].max()),
        value=1
    )

    if st.button("Gợi ý theo người dùng"):
        result = rec.cf_recommend(
            user_id=int(user_id),
            top_k=10
        )

        st.subheader(f"Gợi ý cho user_id = {user_id}")
        show_recommendations(result, prefix="cf")

elif page == "Cold Start":
    st.title("❄️ Cold Start Recommendation")

    st.write("Dành cho người dùng mới chưa có lịch sử tương tác.")

    common_genres = [
        "Fantasy",
        "Romance",
        "Young Adult",
        "Fiction",
        "Science Fiction",
        "Mystery",
        "Historical",
        "Classics",
        "Horror",
        "Nonfiction",
        "Business",
        "Biography"
    ]

    selected_genre = st.selectbox(
        "Bạn thích thể loại nào?",
        common_genres
    )

    if st.button("Gợi ý cho người dùng mới"):
        result = books[
            books["genres"].astype(str).str.contains(
                selected_genre,
                case=False,
                na=False
            )
        ].copy()

        result["rating"] = pd.to_numeric(
            result["rating"],
            errors="coerce"
        )

        result = result.sort_values(
            by="rating",
            ascending=False
        ).head(10)

        st.subheader(f"Sách thể loại {selected_genre}")
        show_recommendations(result, prefix="cold")

elif page == "🛒 Giỏ hàng":
    st.title("🛒 Giỏ hàng theo User")

    user_id = st.number_input(
        "Chọn user_id",
        min_value=0,
        max_value=int(interactions["user_id"].max()),
        value=1
    )

    user_id = int(user_id)

    if user_id not in st.session_state.cart:
        st.session_state.cart[user_id] = []

    search_text = st.text_input("Tìm kiếm sách để thêm vào giỏ", "")

    if search_text:
        filtered_books = books[
            books["title"].astype(str).str.contains(
                search_text,
                case=False,
                na=False
            )
        ]
    else:
        filtered_books = books

    if len(filtered_books) == 0:
        st.warning("Không tìm thấy sách.")
    else:
        selected_title = st.selectbox(
            "Chọn sách",
            filtered_books["title"].dropna().tolist()
        )

        selected_book = filtered_books[
            filtered_books["title"] == selected_title
        ].iloc[0]

        col1, col2 = st.columns([1, 3])

        with col1:
            show_book_card(selected_book)

        with col2:
            st.subheader(selected_book["title"])
            st.write(f"**Tác giả:** {selected_book.get('author', '')}")
            st.write(f"**Rating:** {selected_book.get('rating', '')}")
            st.write(f"**Thể loại:** {selected_book.get('genres', '')}")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("➕ Thêm vào giỏ hàng"):
                    book_id = selected_book["bookId"]

                    if book_id not in st.session_state.cart[user_id]:
                        st.session_state.cart[user_id].append(book_id)
                        st.success("Đã thêm sách vào giỏ hàng.")
                    else:
                        st.warning("Sách này đã có trong giỏ hàng.")

            with col_btn2:
                if st.button("📖 Xem chi tiết sách đang chọn"):
                    st.session_state.selected_detail_book = selected_book.to_dict()
                    st.rerun()

    st.subheader(f"Giỏ hàng của user_id = {user_id}")

    cart_ids = st.session_state.cart[user_id]

    if len(cart_ids) == 0:
        st.info("Giỏ hàng đang trống.")
    else:
        cart_books = books[
            books["bookId"].isin(cart_ids)
        ]

        show_recommendations(cart_books, prefix="cart")

        remove_title = st.selectbox(
            "Chọn sách để xóa khỏi giỏ hàng",
            cart_books["title"].tolist()
        )

        col_remove1, col_remove2 = st.columns(2)

        with col_remove1:
            if st.button("🗑 Xóa khỏi giỏ hàng"):
                remove_book = cart_books[
                    cart_books["title"] == remove_title
                ].iloc[0]

                st.session_state.cart[user_id].remove(
                    remove_book["bookId"]
                )

                st.success("Đã xóa khỏi giỏ hàng.")
                st.rerun()

        with col_remove2:
            if st.button("🧹 Xóa toàn bộ giỏ hàng"):
                st.session_state.cart[user_id] = []
                st.success("Đã xóa toàn bộ giỏ hàng.")
                st.rerun()

        st.subheader("🎯 Gợi ý dựa trên giỏ hàng")

        if st.button("Gợi ý từ giỏ hàng"):
            all_recs = []

            for book_id in cart_ids:
                result = rec.hybrid_recommend(
                    user_id=user_id,
                    book_id=book_id,
                    top_k=5
                )

                if result is not None and len(result) > 0:
                    all_recs.append(result)

            if len(all_recs) > 0:
                rec_df = pd.concat(all_recs)
                rec_df = rec_df.drop_duplicates(subset=["bookId"])
                rec_df = rec_df[
                    ~rec_df["bookId"].isin(cart_ids)
                ]

                st.write("Sách được đề xuất dựa trên các sách trong giỏ hàng:")
                show_recommendations(rec_df.head(10), prefix="cart_rec")
            else:
                st.warning("Không tìm thấy gợi ý từ giỏ hàng.")

elif page == "Analytics":
    st.title("📊 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Phân phối rating")

        rating_df = books.copy()
        rating_df["rating"] = pd.to_numeric(
            rating_df["rating"],
            errors="coerce"
        )

        fig = px.histogram(
            rating_df,
            x="rating",
            nbins=30,
            title="Rating Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top sách rating cao")

        top_books = books.copy()
        top_books["rating"] = pd.to_numeric(
            top_books["rating"],
            errors="coerce"
        )

        top_books = top_books.sort_values(
            by="rating",
            ascending=False
        ).head(10)

        st.dataframe(
            top_books[["title", "author", "rating"]]
        )

    st.subheader("Thông tin interactions")

    st.write(interactions.head())

    user_count = interactions.groupby("user_id").size().reset_index(name="count")

    fig2 = px.histogram(
        user_count,
        x="count",
        nbins=30,
        title="User Interaction Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top user có nhiều interaction")

    top_users = user_count.sort_values(
        by="count",
        ascending=False
    ).head(10)

    st.dataframe(top_users)