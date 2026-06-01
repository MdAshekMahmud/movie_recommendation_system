# Movie Recommendation System

A content-based movie recommendation system built with Python, Pandas, Scikit-learn, cosine similarity, and Streamlit.

The app recommends similar movies based on movie overview, genres, keywords, cast, and director information. Users can select a movie from a dropdown and receive a list of similar movies.

## Project Overview

This project uses the TMDB 5000 Movie Dataset to build a content-based recommendation system.

Unlike a rating-based recommendation system, this project does not depend on user ratings or user behavior. Instead, it compares movie content features and recommends movies with similar textual profiles.

Example:

If a user selects `Avatar`, the system recommends movies that are similar in terms of genre, story keywords, cast, director, and overview.

## Project Type

Content-Based Recommendation System

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Joblib
- Pickle
- GitHub

## Machine Learning Concepts Used

- Data cleaning
- Feature engineering
- Text preprocessing
- Stemming
- CountVectorizer
- Bag-of-words representation
- Cosine similarity
- Similarity-based recommendation

## Dataset

This project uses the TMDB 5000 Movie Dataset.

Required files:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

Place both files inside the `data/` folder:

```text
data/
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
```

The raw dataset files are not included in this repository because they are large.

## Project Structure

```text
movie_recommendation_system/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── movies_compressed.joblib
│   └── similarity_compressed.joblib
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_text_preprocessing.ipynb
│   ├── 05_vectorization.ipynb
│   ├── 06_similarity_calculation.ipynb
│   ├── 07_recommendation_function.ipynb
│   └── 08_model_saving_artifact_management.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── recommender.py
│
├── prepare_deployment_artifacts.py
├── test_artifacts.py
├── test_recommender.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How the System Works

The project follows this pipeline:

```text
Raw movie data
↓
Data cleaning
↓
Feature selection
↓
Feature engineering
↓
Text preprocessing
↓
Vectorization
↓
Cosine similarity calculation
↓
Recommendation function
↓
Streamlit web app
```

## Feature Engineering

The original dataset contains complex columns such as `genres`, `keywords`, `cast`, and `crew`.

These columns are stored as stringified lists of dictionaries.

Example:

```text
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
```

The project extracts only the useful text values.

Example:

```text
Action Adventure
```

For each movie, the system creates a combined `tags` column using:

```text
overview + genres + keywords + top 3 cast members + director
```

Example final tag:

```text
action adventure fantasy sciencefiction alien planet samworthington zoesaldana jamescameron
```

## Text Preprocessing

The `tags` column is converted to lowercase and processed using stemming.

Example:

```text
loving, loved, loves → love
running, runs → run
adventure → adventur
```

This helps reduce word variation before vectorization.

## Vectorization

The project uses `CountVectorizer` to convert movie tags into numerical vectors.

```python
CountVectorizer(max_features=5000, stop_words="english")
```

This converts text into a bag-of-words representation.

Example:

```text
action adventure space alien
```

becomes a numerical vector that can be compared mathematically.

## Similarity Calculation

The system uses cosine similarity to compare movie vectors.

Cosine similarity measures how close two movie vectors are.

A higher similarity score means the movies are more similar.

## Recommendation Logic

When a user selects a movie:

```text
1. The system finds the selected movie index.
2. It gets similarity scores for that movie.
3. It sorts all movies by similarity score.
4. It skips the selected movie itself.
5. It returns the top similar movies.
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it.

For Windows:

```bash
venv\Scripts\activate
```

For Linux or Mac:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset files

Download the TMDB 5000 Movie Dataset and place these files inside the `data/` folder:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

### 5. Generate artifacts

```bash
python src/preprocessing.py
```

This will generate:

```text
artifacts/movies.pkl
artifacts/similarity.pkl
artifacts/count_vectorizer.pkl
artifacts/vectors.pkl
```

### 6. Prepare deployment artifacts

```bash
python prepare_deployment_artifacts.py
```

This will generate:

```text
artifacts/movies_compressed.joblib
artifacts/similarity_compressed.joblib
```

### 7. Run the Streamlit app

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

## Test the Project

Test artifact files:

```bash
python test_artifacts.py
```

Test recommendation function:

```bash
python test_recommender.py
```

## Example Output

Input:

```text
Avatar
```

Output:

```text
1. Aliens
2. John Carter
3. Star Trek Into Darkness
4. Battle: Los Angeles
5. Titan A.E.
```

The exact output may vary depending on preprocessing and dataset version.

## Deployment

This project can be deployed using Streamlit Community Cloud.

Deployment entry file:

```text
app/app.py
```

The deployed app should use compressed artifacts:

```text
artifacts/movies_compressed.joblib
artifacts/similarity_compressed.joblib
```

Raw dataset files are not required during app usage.

## Limitations

This is a basic content-based recommendation system.

Current limitations:

- It does not use user ratings.
- It does not learn from user behavior.
- It does not use collaborative filtering.
- It depends heavily on text feature quality.
- It may recommend movies with similar keywords but different audience appeal.
- The similarity matrix can become large for bigger datasets.

## Future Improvements

Possible improvements:

- Add movie posters using TMDB API.
- Use TF-IDF instead of CountVectorizer.
- Use word embeddings or sentence embeddings.
- Add collaborative filtering.
- Build a hybrid recommendation system.
- Add user login and watchlist.
- Add genre filters.
- Add popularity-based fallback recommendations.
- Optimize similarity search using nearest neighbors or FAISS.

## Project Status

Completed.

The project includes:

- Data preprocessing pipeline
- Recommendation function
- Streamlit app
- Test scripts
- Deployment-ready artifacts
- GitHub-ready structure

## Author

TANVIR NIBIR
