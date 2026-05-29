"""
Helper functions for recommending similar movies.

You can import the `recommend` function directly to get a list of titles
corresponding to the most similar movies, given a preprocessed dataframe and
precomputed similarity matrix.  There is also a convenience function
`load_artefacts` which reads the pickles from disk.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import os
import pickle

import pandas as pd


def load_artefacts(artefact_dir: str = 'artifacts') -> Tuple[pd.DataFrame, list[list[float]]]:
    """Load the processed movies dataframe and similarity matrix from pickle files.

    Args:
        artefact_dir: Directory containing `movies.pkl` and `similarity.pkl`.

    Returns:
        A tuple `(movies, similarity)` where `movies` is a pandas DataFrame and
        `similarity` is a 2D list or numpy array of cosine similarity scores.
    """
    movies_path = os.path.join(artefact_dir, 'movies.pkl')
    similarity_path = os.path.join(artefact_dir, 'similarity.pkl')
    with open(movies_path, 'rb') as f:
        movies = pickle.load(f)
    with open(similarity_path, 'rb') as f:
        similarity = pickle.load(f)
    return movies, similarity


def recommend(movie_title: str, movies: pd.DataFrame, similarity: Iterable[Iterable[float]], top_n: int = 5) -> List[str]:
    """Return the titles of the most similar movies.

    Args:
        movie_title: The title of the movie for which to find recommendations.
        movies: DataFrame containing at least the columns `title` and `movie_id`.
        similarity: Cosine similarity matrix aligned with the order of `movies`.
        top_n: Number of recommendations to return.

    Returns:
        A list of movie titles recommended in descending order of similarity.
        If the movie title is not found, an empty list is returned.
    """
    # Normalise the title matching to avoid case mismatches
    matches = movies[movies['title'].str.lower() == movie_title.strip().lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    # Enumerate distances and sort by similarity (largest first)
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    # Skip the first one (the movie itself)
    recommended_indices = [i for i, _ in distances[1: top_n + 1]]
    recommended_titles = movies.iloc[recommended_indices]['title'].tolist()
    return recommended_titles