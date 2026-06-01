# Movie Recommendation System

This repository contains a small, self‑contained example of a **content‑based movie recommendation system**. The goal of the project is to demonstrate how to build a recommendation pipeline from raw data, through feature engineering and vectorisation, to deployment as a simple web application. While the included code is ready to run, the full TMDB data files are not distributed here due to size and licensing restrictions—please follow the instructions below to download them yourself.

## Project layout

The repository is organised to separate raw data, source code, notebooks, a simple web application and generated artefacts. A typical layout after you download the data and generate the model artefacts looks like this:

```
movie_recommendation_system/
│
├── data/
│   ├── tmdb_5000_movies.csv        # raw TMDB movies dataset (download separately)
│   └── tmdb_5000_credits.csv       # raw TMDB credits dataset (download separately)
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb              # Explore dataset shape, columns, missing values, and sample rows
│   ├── 02_data_cleaning.ipynb                      # Clean missing values, duplicate values, and unnecessary columns
│   ├── 03_feature_engineering.ipynb                # Select useful features and create combined movie metadata
│   ├── 04_text_preprocessing.ipynb                 # Clean text, remove spaces, apply stemming, and prepare tags
│   ├── 05_vectorization.ipynb                      # Convert text tags into numerical vectors using CountVectorizer
│   ├── 06_similarity_calculation.ipynb             # Calculate cosine similarity between movie vectors
│   ├── 07_recommendation_function.ipynb            # Build and test the movie recommendation function
│   └── 08_model_saving_artifact_management.ipynb   # Save movies.pkl and similarity.pkl inside artifacts folder
│
├── src/
│   ├── preprocessing.py            # functions to clean and merge the TMDB data
│   ├── recommender.py              # recommendation helper functions
│   └── __init__.py                 # marks the directory as a package
│
├── app/
│   └── app.py                      # Streamlit application
│
├── artifacts/
│   ├── movies.pkl                  # processed movie metadata (generated)
│   └── similarity.pkl              # cosine similarity matrix (generated)
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # files to ignore in version control
└── README.md                       # this file
```

### Data files

This project uses the **TMDB 5000 Movie Dataset** (two CSV files). These are not included in the repository. You can download them from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). The two files you need are:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place these files in the `data/` directory before running any processing. If you prefer not to sign up to Kaggle you can often find mirrors of these files on GitHub; however these may be out of date.

### Quick start

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   ```

2. **Install the project dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data** and place the two CSV files into the `data/` directory.

4. **Generate the artefacts**. From the project root run the preprocessing script. It will read the raw CSV files, clean and merge the data, create text features (tags) and calculate cosine similarities. The resulting pickled artefacts are saved into the `artifacts/` directory.

   ```bash
   python -m src.preprocessing --movies data/tmdb_5000_movies.csv \
                               --credits data/tmdb_5000_credits.csv \
                               --out-dir artifacts
   ```

5. **Launch the application**. With the artefacts in place you can run the Streamlit app to try out the recommendation system. The app will load the pickles and present a simple dropdown for movie selection.

   ```bash
   streamlit run app/app.py
   ```

6. **Open in your browser**. When Streamlit starts it will display a local URL (usually `http://localhost:8501`). Open this address in your web browser to interact with the recommender.

### Notebooks

The `notebooks/01_dataset_understanding.ipynb` notebook walks through the exploratory data analysis performed on the TMDB 5000 Movie Dataset. It inspects the shapes of the dataframes, the column names, missing values and a few example rows. This is a good place to start if you are unfamiliar with the dataset. Use Jupyter or VS Code to open and run the notebook.

### Sample artefacts

The `artifacts/` directory in this repository contains a small, toy example `movies.pkl` and `similarity.pkl` generated from a manually crafted sample dataset of ten well‑known movies. These files allow you to run the Streamlit app without downloading the full TMDB data, but the recommendations will only use the sample movies. When you generate new artefacts from the full dataset they will overwrite these sample files.

### License

This project is provided for educational purposes. The TMDB dataset is subject to its own license; please review the terms on the Kaggle page before using the data in a production environment.
