import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

# -------------------------------
# LOAD & PREPARE DATA (CACHED)
# -------------------------------
@st.cache_data
def load_data_and_model():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
    ]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in eval(text)]

    def convert_cast(text):
        return [i['name'] for i in eval(text)[:3]]

    def fetch_director(text):
        return [i['name'] for i in eval(text) if i['job'] == 'Director']

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['tags'] = (
        movies['overview'] + " " +
        movies['genres'].apply(lambda x: " ".join(x)) + " " +
        movies['keywords'].apply(lambda x: " ".join(x)) + " " +
        movies['cast'].apply(lambda x: " ".join(x)) + " " +
        movies['crew'].apply(lambda x: " ".join(x))
    )

    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].astype(str)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


new_df, similarity = load_data_and_model()

# -------------------------------
# RECOMMEND FUNCTION
# -------------------------------
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [new_df.iloc[i[0]].title for i in movie_list]


# -------------------------------
# UI
# -------------------------------
selected_movie = st.selectbox(
    "Select a movie",
    new_df['title'].values
)

if st.button("Recommend"):
    st.subheader("🎥 Recommended Movies")
    for movie in recommend(selected_movie):
        st.write("👉", movie)
