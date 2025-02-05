import os
os.system("pip install scipy")
import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# ---------------------------
# Helper Functions
# ---------------------------

def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', "I'", text)
    text = re.sub(r'&amp;', 'and', text)
    return text

def load_data():
    # Change these paths as necessary
    anime = pd.read_csv('anime.csv')
    rating = pd.read_csv('rating.csv')

    # Basic cleaning and preprocessing
    anime.dropna(axis=0, inplace=True)
    rating.drop_duplicates(keep='first', inplace=True)

    fulldata = pd.merge(anime, rating, on="anime_id", suffixes=[None, "_user"])
    fulldata = fulldata.rename(columns={"rating_user": "user_rating"})
    return anime, fulldata

def preprocess_for_collaborative(fulldata):
    data = fulldata.copy()
    data["user_rating"].replace(to_replace=-1, value=np.nan, inplace=True)
    data = data.dropna(axis=0)

    # Filter users with >= 50 ratings for robustness
    selected_users = data["user_id"].value_counts()
    data = data[data["user_id"].isin(selected_users[selected_users >= 50].index)]
    
    # Clean anime names
    data["name"] = data["name"].apply(text_cleaning)
    
    data_pivot = data.pivot_table(index="name", columns="user_id", values="user_rating").fillna(0)
    return data_pivot

def build_knn_model(data_pivot):
    data_matrix = csr_matrix(data_pivot.values)
    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(data_matrix)
    return model_knn

def get_collaborative_recommendations(anime_name, data_pivot, model_knn, anime):
    # Ensure the anime exists in the pivot table
    if anime_name not in data_pivot.index:
        return None
    
    query_idx = data_pivot.index.get_loc(anime_name)
    distances, indices = model_knn.kneighbors(data_pivot.iloc[query_idx, :].values.reshape(1, -1), n_neighbors=6)
    
    recs = []
    for i in range(1, len(distances.flatten())):  # Skip the first as it's the same anime
        rec_name = data_pivot.index[indices.flatten()[i]]
        # Retrieve average rating from original anime dataset
        rating_val = anime[anime["name"] == rec_name]["rating"].values
        avg_rating = rating_val[0] if len(rating_val) > 0 else None
        recs.append((rec_name, avg_rating))
    return recs

def preprocess_for_content(fulldata):
    rec_data = fulldata.copy()
    rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
    rec_data.reset_index(drop=True, inplace=True)
    return rec_data

def build_content_model(rec_data):
    # Using genres for content-based filtering
    genres = rec_data["genre"].str.split(", | , | ,").astype(str)
    tfv = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 3),
        stop_words="english"
    )
    tfv_matrix = tfv.fit_transform(genres)
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    # Map anime names to indices for easy lookup
    rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()
    return sig, rec_indices

def get_content_recommendations(title, sig, rec_indices, anime):
    if title not in rec_indices:
        return None
    idx = rec_indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)[1:11]
    anime_indices = [i[0] for i in sig_scores]
    
    recs = []
    for i in anime_indices:
        rec_title = anime.iloc[i]["name"]
        rec_rating = anime.iloc[i]["rating"]
        recs.append((rec_title, rec_rating))
    return recs

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Anime Recommendation System")
    st.write("Welcome to the Anime Recommender. Choose a method and enter an anime title to get recommendations.")

    # Load and preprocess data
    st.info("Loading data and building models. This may take a moment...")
    anime, fulldata = load_data()

    # Build collaborative filtering model
    data_pivot = preprocess_for_collaborative(fulldata)
    knn_model = build_knn_model(data_pivot)
    
    # Build content-based model
    rec_data = preprocess_for_content(fulldata)
    sig, rec_indices = build_content_model(rec_data)
    
    st.success("Models are ready!")

    # Sidebar for selecting recommendation type
    rec_type = st.sidebar.selectbox("Select Recommendation Type", ("Collaborative Filtering", "Content-Based"))

    user_input = st.text_input("Enter the Anime Title:", "Naruto")

    if st.button("Get Recommendations"):
        if rec_type == "Collaborative Filtering":
            recs = get_collaborative_recommendations(user_input, data_pivot, knn_model, anime)
            if recs:
                st.subheader(f"Recommendations based on viewers of '{user_input}':")
                for idx, (name, rating) in enumerate(recs, 1):
                    st.write(f"{idx}. {name} (Average Rating: {rating})")
            else:
                st.error("Anime not found in our collaborative dataset.")
        else:
            recs = get_content_recommendations(user_input, sig, rec_indices, anime)
            if recs:
                st.subheader(f"Content-based recommendations similar to '{user_input}':")
                for idx, (name, rating) in enumerate(recs, 1):
                    st.write(f"{idx}. {name} (Average Rating: {rating})")
            else:
                st.error("Anime not found in our content-based dataset.")

if __name__ == '__main__':
    main()
