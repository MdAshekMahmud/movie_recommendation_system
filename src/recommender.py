import os
import pickle
import joblib
import pandas as pd


def load_artifacts(artifacts_dir="artifacts"):
    """
    Load movies dataframe and similarity matrix.

    The function first tries to load compressed deployment artifacts.
    If they are not found, it falls back to the original pickle files.
    """

    compressed_movies_path = os.path.join(artifacts_dir, "movies_compressed.joblib")
    compressed_similarity_path = os.path.join(
        artifacts_dir, "similarity_compressed.joblib"
    )

    pickle_movies_path = os.path.join(artifacts_dir, "movies.pkl")
    pickle_similarity_path = os.path.join(artifacts_dir, "similarity.pkl")

    if os.path.exists(compressed_movies_path) and os.path.exists(
        compressed_similarity_path
    ):
        movies = joblib.load(compressed_movies_path)
        similarity = joblib.load(compressed_similarity_path)

    elif os.path.exists(pickle_movies_path) and os.path.exists(pickle_similarity_path):
        with open(pickle_movies_path, "rb") as file:
            movies = pickle.load(file)

        with open(pickle_similarity_path, "rb") as file:
            similarity = pickle.load(file)

    else:
        raise FileNotFoundError(
            "No artifact files found. Expected compressed joblib files or pickle files inside artifacts/."
        )

    movies = movies.reset_index(drop=True)

    return movies, similarity


def search_movie(query, movies, limit=20):
    """
    Search movie titles using partial text.
    """

    query = query.strip().lower()

    results = movies[movies["title"].str.lower().str.contains(query, na=False)]

    return results[["title"]].head(limit)


def recommend(movie_name, movies, similarity, top_n=5):
    """
    Recommend similar movies based on cosine similarity.
    """

    movie_name_clean = movie_name.strip().lower()

    matches = movies[movies["title"].str.lower() == movie_name_clean]

    if matches.empty:
        return pd.DataFrame(columns=["movie", "similarity_score"])

    movie_index = matches.index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))

    sorted_scores = sorted(similarity_scores, reverse=True, key=lambda x: x[1])

    recommendations = []

    for item in sorted_scores[1 : top_n + 1]:
        recommended_movie_index = item[0]
        recommended_title = movies.iloc[recommended_movie_index]["title"]
        similarity_score = item[1]

        recommendations.append(
            {
                "movie": recommended_title,
                "similarity_score": round(float(similarity_score), 4),
            }
        )

    return pd.DataFrame(recommendations)
