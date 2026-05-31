from src.recommender import load_artifacts, recommend, search_movie

movies, similarity = load_artifacts("artifacts")

print("Search result:")
print(search_movie("avatar", movies))

print("\nRecommendations:")
print(recommend("Avatar", movies, similarity, top_n=5))
