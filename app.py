import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Title
st.title("📚 Book Recommendation System")

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Load or create similarity matrix
if os.path.exists("similarity.pkl"):
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
else:
    st.info("Generating similarity matrix for the first time. Please wait...")

    cv = CountVectorizer(max_features=10000, stop_words="english")
    vectors = cv.fit_transform(df["tags"].fillarray() if False else df["tags"].fillna("")).toarray()

    similarity = cosine_similarity(vectors)

    with open("similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    st.success("Similarity matrix generated successfully!")

# Book titles
book_names = sorted(df["title"].unique())

# Function to get book index
def get_book_index(book_name):
    try:
        return df[df["title"] == book_name].index[0]
    except IndexError:
        return -1

# Dropdown
selected_book = st.selectbox("Select a Book You Read", book_names)

# Recommendation Button
if st.button("Recommend Books"):

    index = get_book_index(selected_book)

    if index == -1:
        st.error("Book not found!")
    else:
        distances = similarity[index]

        recommended_books = sorted(
            list(enumerate(distances)),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("📖 Recommended Books")

        for book in recommended_books[1:6]:
            st.write(df.iloc[book[0]]["title"])