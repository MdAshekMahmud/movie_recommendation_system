import os
import sys

import pandas as pd
import streamlit as st

# Add project root to Python path
# This helps Streamlit find the src folder when app.py is inside app/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.recommender import load_artifacts, recommend, search_movie

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="centered"
)


@st.cache_data
def load_data():
    """
    Load saved movie dataframe and similarity matrix.
    Streamlit caches this so the app does not reload large files repeatedly.
    """

    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    movies, similarity = load_artifacts(artifacts_dir)

    return movies, similarity


def main():
    st.title("🎬 Movie Recommendation System")

    st.write(
        "This app recommends similar movies using a content-based recommendation approach. "
        "It compares movie overview, genres, keywords, cast, and director information."
    )

    try:
        movies, similarity = load_data()

    except FileNotFoundError:
        st.error(
            "Artifact files not found. Please make sure movies.pkl and similarity.pkl "
            "exist inside the artifacts folder."
        )
        st.stop()

    except Exception as error:
        st.error(f"Something went wrong while loading artifacts: {error}")
        st.stop()

    st.sidebar.header("Settings")

    top_n = st.sidebar.slider(
        "Number of recommendations", min_value=3, max_value=15, value=5, step=1
    )

    st.subheader("Find Similar Movies")

    movie_titles = movies["title"].sort_values().values

    selected_movie = st.selectbox("Select a movie", movie_titles)

    if st.button("Recommend"):
        recommendations = recommend(
            movie_name=selected_movie, movies=movies, similarity=similarity, top_n=top_n
        )

        if recommendations.empty:
            st.warning("No recommendations found for this movie.")
        else:
            st.success(f"Top {top_n} movies similar to {selected_movie}")

            for index, row in recommendations.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {index + 1}. {row['movie']}")
                    st.write(f"Similarity score: `{row['similarity_score']}`")

            with st.expander("View result as table"):
                st.dataframe(recommendations, use_container_width=True)

    st.divider()

    st.subheader("Search Movie Title")

    query = st.text_input("Type part of a movie title")

    if query:
        search_results = search_movie(query, movies, limit=20)

        if search_results.empty:
            st.info("No matching movie found.")
        else:
            st.write("Matching movie titles:")
            st.dataframe(search_results, use_container_width=True)

    st.divider()

    st.caption(
        "Built with Python, Pandas, Scikit-learn, cosine similarity, and Streamlit."
    )


if __name__ == "__main__":
    main()
