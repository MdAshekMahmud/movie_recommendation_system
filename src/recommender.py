import os
import pickle
import pandas as pd


def load_artifacts(artifacts_dir="artifacts"):
    """
    Load movies dataframe and similarity matrix from the artifacts folder.
    """
    movies_path = os.path.join(artifacts_dir, "movies.pkl")
    similarity_path = os.path.join(artifacts_dir, "similarity.pkl")

    with open(movies_path, "rb") as file:
        movies = pickle.load(file)

    with open(similarity_path, "rb") as file:
        similarity = pickle.load(file)

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

    Parameters:
        movie_name: selected movie title
        movies: dataframe containing movie_id, title, tags
        similarity: cosine similarity matrix
        top_n: number of recommendations

    Returns:
        DataFrame containing recommended movie titles and similarity scores
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
