import pickle


def test_artifacts():
    movies = pickle.load(open("artifacts/movies.pkl", "rb"))
    similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))

    movies = movies.reset_index(drop=True)

    print("Movies shape:", movies.shape)
    print("Similarity shape:", similarity.shape)

    assert "movie_id" in movies.columns, "movie_id column missing"
    assert "title" in movies.columns, "title column missing"
    assert "tags" in movies.columns, "tags column missing"

    assert (
        movies.shape[0] == similarity.shape[0]
    ), "Movies rows and similarity rows do not match"
    assert (
        movies.shape[0] == similarity.shape[1]
    ), "Movies rows and similarity columns do not match"

    print("All artifact tests passed.")


if __name__ == "__main__":
    test_artifacts()
