import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.recommender import load_artifacts, recommend, search_movie

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="centered"
)


@st.cache_data(show_spinner=True)
def load_data():
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    movies, similarity = load_artifacts(artifacts_dir)
    return movies, similarity


def main():
    st.title("🎬 Movie Recommendation System")

    st.write(
        "A content-based movie recommendation app using movie overview, genres, "
        "keywords, cast, director, CountVectorizer, and cosine similarity."
    )

    try:
        movies, similarity = load_data()

    except Exception as error:
        st.error("Could not load deployment artifacts.")
        st.code(str(error))
        st.stop()

    st.sidebar.header("Settings")

    top_n = st.sidebar.slider(
        "Number of recommendations", min_value=3, max_value=15, value=5, step=1
    )

    st.subheader("Recommend Similar Movies")

    movie_titles = sorted(movies["title"].values)

    selected_movie = st.selectbox("Select a movie", movie_titles)

    if st.button("Recommend"):
        recommendations = recommend(
            movie_name=selected_movie, movies=movies, similarity=similarity, top_n=top_n
        )

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.success(f"Top {top_n} movies similar to {selected_movie}")

            for index, row in recommendations.iterrows():
                st.markdown(
                    f"**{index + 1}. {row['movie']}**  \n"
                    f"Similarity score: `{row['similarity_score']}`"
                )

            with st.expander("View recommendations as table"):
                st.dataframe(recommendations, use_container_width=True)

    st.divider()

    st.subheader("Search Movie Title")

    query = st.text_input("Type part of a movie title")

    if query:
        results = search_movie(query, movies, limit=20)

        if results.empty:
            st.info("No matching movie title found.")
        else:
            st.dataframe(results, use_container_width=True)

    st.divider()

    st.caption("Built with Python, Pandas, Scikit-learn, Joblib, and Streamlit.")


if __name__ == "__main__":
    main()
