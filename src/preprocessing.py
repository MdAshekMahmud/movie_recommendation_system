import os
import ast
import pickle
import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


def convert(obj):
    names = []

    for item in ast.literal_eval(obj):
        names.append(item["name"])

    return names


def convert_cast(obj):
    names = []
    counter = 0

    for item in ast.literal_eval(obj):
        if counter != 3:
            names.append(item["name"])
            counter += 1
        else:
            break

    return names


def fetch_director(obj):
    directors = []

    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            directors.append(item["name"])
            break

    return directors


def remove_space(word_list):
    new_list = []

    for word in word_list:
        new_list.append(word.replace(" ", ""))

    return new_list


def stem(text):
    stemmed_words = []

    for word in text.split():
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)


def build_artifacts(
    movies_path="data/tmdb_5000_movies.csv",
    credits_path="data/tmdb_5000_credits.csv",
    artifacts_dir="artifacts",
):
    # Load datasets
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge datasets
    movies = movies.merge(credits, on="title")

    # Select useful columns
    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]

    # Clean missing values and duplicates
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)

    # Feature extraction
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    # Convert overview to list
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Remove spaces from multi-word names
    movies["genres"] = movies["genres"].apply(remove_space)
    movies["keywords"] = movies["keywords"].apply(remove_space)
    movies["cast"] = movies["cast"].apply(remove_space)
    movies["crew"] = movies["crew"].apply(remove_space)

    # Create tags
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    # Final dataframe
    new_df = movies[["movie_id", "title", "tags"]].copy()

    # Convert tags list to text
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

    # Lowercase
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    # Stemming
    new_df["tags"] = new_df["tags"].apply(stem)

    # Reset index
    new_df = new_df.reset_index(drop=True)

    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    # Similarity calculation
    similarity = cosine_similarity(vectors)

    # Create artifacts folder
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save artifacts
    pickle.dump(new_df, open(os.path.join(artifacts_dir, "movies.pkl"), "wb"))
    pickle.dump(similarity, open(os.path.join(artifacts_dir, "similarity.pkl"), "wb"))
    pickle.dump(cv, open(os.path.join(artifacts_dir, "count_vectorizer.pkl"), "wb"))
    pickle.dump(vectors, open(os.path.join(artifacts_dir, "vectors.pkl"), "wb"))

    print("Artifacts generated successfully.")
    print("Movies shape:", new_df.shape)
    print("Vectors shape:", vectors.shape)
    print("Similarity shape:", similarity.shape)


if __name__ == "__main__":
    build_artifacts()
