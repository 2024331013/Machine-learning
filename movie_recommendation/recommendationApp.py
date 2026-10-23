import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


def clean_title(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("model.joblib")


def search(title, movies, vectorizer, tfidf, n=5):
    query_vec = vectorizer.transform([clean_title(title)])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    idx = np.argsort(-similarity)[:n]
    return movies.iloc[idx]


@st.cache_data(show_spinner="Finding recommendations...")
def find_similar_movies(movie_id, n=10):
    ratings = model["ratings"]
    movies = model["movies"]

    # users who gave this movie 5 stars
    similar_users = ratings[
        (ratings["movieId"] == movie_id) & (ratings["rating"] >= 5)
    ]["userId"].unique()
    if len(similar_users) == 0:
        return pd.DataFrame()

    # what those users loved, as a fraction of them
    similar_recs = (
        ratings[ratings["userId"].isin(similar_users)]["movieId"].value_counts()
        / len(similar_users)
    )
    similar_recs = similar_recs[similar_recs > 0.1]

    # how much ALL users love those same movies
    all_users = ratings[ratings["movieId"].isin(similar_recs.index)]
    all_recs = all_users["movieId"].value_counts() / all_users["userId"].nunique()

    rec = pd.concat([similar_recs, all_recs], axis=1)
    rec.columns = ["similar", "all"]
    rec["score"] = rec["similar"] / rec["all"]
    rec = rec.drop(index=movie_id, errors="ignore").sort_values("score", ascending=False)

    out = rec.head(n).merge(movies, left_index=True, right_on="movieId")
    out = out[["title", "genres", "score"]].reset_index(drop=True)
    out.index += 1
    out["score"] = out["score"].round(2)
    return out


try:
    model = load_model()
except FileNotFoundError:
    st.error("model.joblib not found. Run `python build_model.py` first.")
    st.stop()

st.title("🎬 Movie Recommender")
st.caption("Type a movie you like and get movies loved by people with the same taste.")

query = st.text_input("Movie title", placeholder="Example: Toy Story")

if query.strip():
    matches = search(query, model["movies"], model["vectorizer"], model["tfidf"])
    choice = st.selectbox(
        "Which one did you mean?",
        options=matches["movieId"].tolist(),
        format_func=lambda mid: matches.loc[matches["movieId"] == mid, "title"].iloc[0],
    )
    n = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Recommend", type="primary"):
        recs = find_similar_movies(int(choice), n)
        if recs.empty:
            st.warning("Not enough 5-star ratings for this movie to recommend from.")
        else:
            st.subheader("You might also like")
            st.dataframe(recs, use_container_width=True)
