# 🎬 Hybrid Movie Recommendation System

A personalized movie recommendation system built using the **MovieLens dataset**, combining collaborative filtering, content-based filtering, and hybrid recommendation techniques.

The project compares multiple recommendation algorithms using Precision@10, Recall@10, NDCG@10, and catalog coverage, and exposes personalized recommendations through a FastAPI backend.

## 🚀 Key Features

- **Popularity-Based Recommendations:** Recommends movies based on overall popularity.
- **Content-Based Filtering:** Uses movie genres and tags to identify similar movies.
- **Item-Based Collaborative Filtering:** Finds movies based on user rating patterns.
- **ALS Collaborative Filtering:** Uses implicit-feedback matrix factorization.
- **Hybrid Recommendation:** Combines ALS and ItemKNN rankings using rank fusion.
- **Model Evaluation:** Compares recommendation quality using ranking metrics.
- **REST API:** Provides movie search, user information, and personalized recommendations.
- **Automated Testing:** Includes unit tests and code-quality checks.

## 🧠 Recommendation Models

| Model | Approach |
| --- | --- |
| Popularity | Most popular movies |
| Content-Based | Genre similarity |
| Content-Based + Tags | Genre and tag similarity |
| ItemKNN | Item-based collaborative filtering |
| ItemKNN-Liked | ItemKNN using user-liked movies |
| ALS | Implicit-feedback matrix factorization |
| ALS-Liked | ALS with user-liked movie preferences |
| Hybrid ALS + ItemKNN | Rank-fusion ensemble |

## 📊 Model Evaluation Results

Evaluation performed on **573 users**, with the top 10 recommendations per user.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0314 | 0.0471 | 0.0500 | 0.0104 |
| Content-Based (Genres) | 0.0073 | 0.0103 | 0.0110 | 0.0953 |
| Content-Based (Genres + Tags) | 0.0047 | 0.0067 | 0.0070 | 0.1074 |
| ItemKNN | 0.0304 | 0.0537 | 0.0480 | 0.0432 |
| ItemKNN-Liked | 0.0480 | 0.0827 | 0.0752 | 0.0541 |
| ALS | 0.0375 | 0.0821 | 0.0636 | 0.0848 |
| ALS-Liked | 0.0462 | 0.0938 | 0.0797 | 0.0741 |
| **Hybrid ALS + ItemKNN** | **0.0529** | **0.0959** | **0.0833** | **0.0589** |

### Key Findings

- The Hybrid ALS + ItemKNN model achieved the best Precision@10, Recall@10, and NDCG@10 among the evaluated models.
- Compared with ItemKNN-Liked, the hybrid model improved Precision@10 by approximately 10.2%, Recall@10 by 15.9%, and NDCG@10 by 10.8%.
- ALS-Liked achieved higher catalog coverage than the hybrid model, demonstrating a trade-off between recommendation relevance and catalog diversity.
- Bootstrap confidence intervals indicated statistically significant improvements over the popularity baseline for the three ranking metrics.

## 🏗️ Project Architecture

```text
movie-recsys/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── recsys/
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── stats.py
│       └── models/
│           ├── als.py
│           ├── base.py
│           ├── content.py
│           ├── hybrid.py
│           ├── itemknn.py
│           └── popularity.py
│
├── tests/
├── pyproject.toml
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn
- **Recommendation Algorithms:** ALS, ItemKNN, Content-Based Filtering
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Testing:** Pytest
- **Code Quality:** Ruff
- **Version Control:** Git and GitHub

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nitin28061999/movie-recsys.git
cd movie-recsys
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -e ".[dev,app]"
```

### 4. Download the MovieLens dataset

Download the **MovieLens latest-small dataset** from:

<https://grouplens.org/datasets/movielens/latest/>

Extract the dataset into:

```text
data/raw/ml-latest-small/
```

### 5. Run tests

```bash
pytest
```

### 6. Run model evaluation

```bash
python -m recsys.evaluate
```

## 🌐 FastAPI Endpoints

Start the backend:

```bash
python -m uvicorn api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

| Endpoint | Description |
| --- | --- |
| `GET /` | API welcome message |
| `GET /health` | Health check |
| `GET /movies/search?q=Batman` | Search movies |
| `GET /recommendations/{user_id}?k=10` | Get personalized recommendations |
| `GET /users?limit=20` | Retrieve available users |

## 📈 Evaluation Metrics

- **Precision@K:** Measures the proportion of recommended movies that are relevant.
- **Recall@K:** Measures the proportion of relevant movies successfully recommended.
- **NDCG@K:** Evaluates ranking quality by giving higher importance to relevant movies appearing near the top.
- **Catalog Coverage:** Measures the proportion of available movies that appear in recommendations.

## 🔮 Future Improvements

- Deploy the FastAPI backend and Streamlit frontend.
- Add user authentication and personalized user profiles.
- Experiment with additional hybrid ranking strategies.
- Improve recommendation diversity and cold-start handling.
- Add model monitoring and recommendation analytics.

## 👨‍💻 Author

Nitin Gupta

- GitHub: <https://github.com/nitin28061999>
- Project Repository: <https://github.com/nitin28061999/movie-recsys>

---

⭐ If you find this project useful, consider giving the repository a star.
