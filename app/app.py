"""
Streamlit application for the movie recommendation system.

Run this app with:

    streamlit run app/app.py

It loads the preprocessed movie metadata and cosine similarity matrix from the
`artifacts/` directory.  A dropdown allows the user to select a movie and the
system recommends similar movies based on content (overview, genres,
keywords, cast and crew).

If you have generated your own artefacts, ensure they are saved into
`artifacts/movies.pkl` and `artifacts/similarity.pkl` before running the app.
"""

import streamlit as st
from src.recommender import load_artefacts, recommend


@st.cache_data
def get_data():
    """Load the movies and similarity matrix once and cache them."""
    return load_artefacts('artifacts')


def main() -> None:
    st.title('🎬 Movie Recommendation System')
    st.write("Select a movie to receive content‑based recommendations.")

    movies, similarity = get_data()
    movie_list = movies['title'].values
    selected_movie = st.selectbox('Choose a movie', movie_list)

    if st.button('Recommend'):
        recommendations = recommend(selected_movie, movies, similarity, top_n=5)
        if recommendations:
            st.subheader('Recommended Movies:')
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")
        else:
            st.warning('Movie not found in the dataset.  Please ensure that you selected a valid movie.')


if __name__ == '__main__':
    main()