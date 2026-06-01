import os
import pickle
import joblib


def prepare_deployment_artifacts():
    artifacts_dir = "artifacts"

    movies_path = os.path.join(artifacts_dir, "movies.pkl")
    similarity_path = os.path.join(artifacts_dir, "similarity.pkl")

    movies_output_path = os.path.join(artifacts_dir, "movies_compressed.joblib")
    similarity_output_path = os.path.join(artifacts_dir, "similarity_compressed.joblib")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            "artifacts/movies.pkl not found. Run preprocessing first."
        )

    if not os.path.exists(similarity_path):
        raise FileNotFoundError(
            "artifacts/similarity.pkl not found. Run preprocessing first."
        )

    print("Loading movies.pkl...")
    with open(movies_path, "rb") as file:
        movies = pickle.load(file)

    print("Loading similarity.pkl...")
    with open(similarity_path, "rb") as file:
        similarity = pickle.load(file)

    movies = movies.reset_index(drop=True)

    print("Saving compressed movies artifact...")
    joblib.dump(movies, movies_output_path, compress=3)

    print("Saving compressed similarity artifact...")
    joblib.dump(similarity, similarity_output_path, compress=3)

    print("Deployment artifacts created successfully.")
    print("Movies artifact:", movies_output_path)
    print("Similarity artifact:", similarity_output_path)


if __name__ == "__main__":
    prepare_deployment_artifacts()
