"""
Utility functions for cleaning and preparing the TMDB 5000 movie data.

This module exposes a simple command line interface to generate pickle artefacts
(`movies.pkl` and `similarity.pkl`) from the raw CSV files.  It can also be
imported and used programmatically in other scripts.

Example usage as a script:

    python -m src.preprocessing --movies data/tmdb_5000_movies.csv \
                                --credits data/tmdb_5000_credits.csv \
                                --out-dir artifacts

The resulting pickles will be saved into the specified output directory.
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
from typing import Tuple

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ps = PorterStemmer()


def _parse_name_list(column: str, max_items: int | None = None) -> list[str]:
    """Parse a stringified list of dictionaries and extract the `name` field.

    Args:
        column: The raw string representation of a Python list of dicts as
            found in the TMDB CSV files.
        max_items: If provided, limit the number of names returned to the first
            `max_items` entries.  Use `None` to return all names.

    Returns:
        A list of names extracted from the dictionaries.  If parsing fails
        (e.g. due to malformed JSON), an empty list is returned.
    """
    try:
        items = ast.literal_eval(column)
    except (ValueError, SyntaxError):
        return []
    names = []
    for item in items:
        name = item.get("name")
        if name:
            names.append(name)
        if max_items and len(names) >= max_items:
            break
    return names


def _parse_director(column: str) -> list[str]:
    """Extract the director's name from the crew JSON string.

    Args:
        column: A stringified list of crew dictionaries.

    Returns:
        A list containing the director's name if present; otherwise an empty
        list.  Some movies may have multiple directors; all of them are
        returned.
    """
    try:
        crew_list = ast.literal_eval(column)
    except (ValueError, SyntaxError):
        return []
    directors = [member.get("name") for member in crew_list if member.get("job") == "Director"]
    # Remove None values
    return [d for d in directors if d]


def _remove_space(names: list[str]) -> list[str]:
    """Remove spaces within names to prevent CountVectorizer from splitting them.

    """
    return [name.replace(" ", "") for name in names]


def _stem_sentence(sentence: str) -> str:
    """Apply Porter stemming to each token in a sentence.

    Args:
        sentence: A lowercase string containing tokens separated by spaces.

    Returns:
        A new sentence string with each token replaced by its stem.
    """
    return " ".join(ps.stem(word) for word in sentence.split())


def preprocess_movies(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load, merge and clean the TMDB movie and credits datasets.

    The returned dataframe contains the columns `movie_id`, `title` and
    `tags`.  The `tags` column is a single string containing a bag of
    stemmed words representing the movie's plot and metadata.  Missing values
    and malformed entries are dropped.

    Args:
        movies_path: Path to `tmdb_5000_movies.csv`.
        credits_path: Path to `tmdb_5000_credits.csv`.

    Returns:
        A cleaned and merged pandas DataFrame with the columns `movie_id`,
        `title` and `tags`.
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    # Merge on title; movie_id appears in credits, id appears in movies
    df = movies.merge(credits, left_on="title", right_on="title")
    # Keep only necessary columns
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Convert stringified JSON columns to lists of names
    df["genres"] = df["genres"].apply(lambda x: _parse_name_list(x))
    df["keywords"] = df["keywords"].apply(lambda x: _parse_name_list(x))
    df["cast"] = df["cast"].apply(lambda x: _parse_name_list(x, max_items=3))
    df["crew"] = df["crew"].apply(_parse_director)

    # Remove spaces so vectoriser treats multi‑word names as single tokens
    df["genres"] = df["genres"].apply(_remove_space)
    df["keywords"] = df["keywords"].apply(_remove_space)
    df["cast"] = df["cast"].apply(_remove_space)
    df["crew"] = df["crew"].apply(_remove_space)

    # Combine overview and lists into a single list per movie
    df["overview"] = df["overview"].apply(lambda x: x.split())
    df["tags"] = df.apply(lambda row: row["overview"] + row["genres"] + row["keywords"] + row["cast"] + row["crew"], axis=1)
    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())
    df["tags"] = df["tags"].apply(_stem_sentence)

    # Return only the relevant columns
    return df[["movie_id", "title", "tags"]]


def create_vectorizer_and_similarity(df: pd.DataFrame) -> Tuple[CountVectorizer, list[list[float]]]:
    """Vectorise the `tags` column and compute cosine similarities.

    Args:
        df: A DataFrame with a `tags` column (stemmed text).

    Returns:
        A tuple `(cv, similarity)` where `cv` is a fitted CountVectorizer and
        `similarity` is a 2D cosine similarity matrix.
    """
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return cv, similarity


def save_artefacts(df: pd.DataFrame, similarity: list[list[float]], out_dir: str) -> None:
    """Persist the processed movie metadata and similarity matrix to disk.

    Two pickle files are written into `out_dir`: `movies.pkl` (containing the
    dataframe with columns `movie_id`, `title` and `tags`) and `similarity.pkl`
    (the cosine similarity matrix).

    Args:
        df: DataFrame containing the processed movie information.
        similarity: Cosine similarity matrix computed from the tags.
        out_dir: Directory where the pickles will be written.  The directory
            will be created if it does not already exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'movies.pkl'), 'wb') as f:
        pickle.dump(df, f)
    with open(os.path.join(out_dir, 'similarity.pkl'), 'wb') as f:
        pickle.dump(similarity, f)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the preprocessing script when run as a module."""
    parser = argparse.ArgumentParser(description="Prepare movie metadata and similarity matrix from TMDB datasets.")
    parser.add_argument('--movies', required=True, help='Path to tmdb_5000_movies.csv')
    parser.add_argument('--credits', required=True, help='Path to tmdb_5000_credits.csv')
    parser.add_argument('--out-dir', default='artifacts', help='Directory to write the pickle files')
    args = parser.parse_args(argv)

    print('Loading and processing datasets...')
    df = preprocess_movies(args.movies, args.credits)
    print(f'Dataset processed: {len(df)} movies')

    print('Vectorising and computing similarity matrix...')
    _, similarity = create_vectorizer_and_similarity(df)
    print('Similarity matrix computed')

    print(f'Saving artefacts into {args.out_dir}...')
    save_artefacts(df, similarity, args.out_dir)
    print('Done.')


if __name__ == '__main__':  # pragma: no cover
    main()