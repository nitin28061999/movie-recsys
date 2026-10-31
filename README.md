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

Evaluated on 573 users (chronological split, test set; models fit on train + validation,
hyperparameters tuned on validation only). A recommendation counts as a hit when the user
later rated that movie 4 or higher. Differences use paired bootstrap 95% confidence
intervals over users.

- The Hybrid ALS + ItemKNN model scored highest on Precision@10 (0.0529), Recall@10 (0.0959)
  and NDCG@10 (0.0833). Versus the popularity baseline that is about 68% higher precision,
  about 2x recall and about 67% higher NDCG, all statistically significant.
- Versus the strongest single model, ItemKNN-Liked, the hybrid's gains in Precision@10
  (+0.0049) and Recall@10 (+0.0132) are statistically significant but small; the NDCG@10
  gain (+0.0081) is not (CI [-0.0002, +0.0169]). ItemKNN-Liked and ALS-Liked are
  statistically tied.
- Training only on movies users liked (rating >= 4) mattered more than model choice:
  ItemKNN-Liked beat plain ItemKNN significantly on all three metrics.
- The content-based models (genres, genres + tags) scored significantly below popularity,
  so they are reported as baselines and are not part of the hybrid. They do reach about 10%
  of the catalog versus about 1% for popularity.
- All models recommend from a small slice of the catalog (under 11%), so popularity bias
  is a known limitation. ALS-Liked has higher coverage (7.4%) than the hybrid (5.9%).

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
│       ├── train.py
│       ├── tune.py
│       └── models/
│           ├── als.py
│           ├── base.py
│           ├── content.py
│           ├── hybrid.py
│           ├── itemknn.py
│           └── popularity.py
│
├── reports/
├── tests/
├── Dockerfile
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

### 4. Dataset

The **MovieLens latest-small** dataset is downloaded automatically into
`data/raw/ml-latest-small/` the first time you run the evaluation or the API.

### 5. Run tests

```bash
pytest
```

### 6. Run model evaluation

```bash
python -m recsys.evaluate               # validation split (used while developing)
python -m recsys.evaluate --split test  # final numbers: fit on train + validation, score on test
```

**Evaluation protocol:** each user's ratings are split chronologically (oldest 80% train,
next 10% validation, newest 10% test); a recommendation counts as a hit when the user later
rated that movie 4 or higher; hyperparameters were tuned on the validation split only
(`python -m recsys.tune als`, `python -m recsys.tune knn`).

## 🌐 FastAPI Endpoints

Start the backend:

```bash
python -m uvicorn api.main:app --port 8001 --reload
```

The first start trains the model on all ratings and saves it to `models/`; later starts load
it from disk. To train ahead of time, run `python -m recsys.train` (it retrains automatically
when the hyperparameters in `src/recsys/config.py` change).

API documentation:

```text
http://127.0.0.1:8001/docs
```

| Endpoint | Description |
| --- | --- |
| `GET /` | API welcome message |
| `GET /health` | Health check |
| `GET /movies/search?q=Batman` | Search movies |
| `GET /recommendations/{user_id}?k=10` | Get personalized recommendations |
| `GET /users?limit=20` | Retrieve available users |

## 🖥️ Streamlit App

With the API running, start the frontend in a second terminal:

```bash
streamlit run app/streamlit_app.py
```

The app expects the API at `http://127.0.0.1:8001`. To point it elsewhere, set the
`API_URL` environment variable first (for example `$env:API_URL = "https://your-api-host"`
in PowerShell).

## 🐳 Docker

The image installs only the API dependencies and trains the model at build time, so containers
start without downloading data:

```bash
docker build -t movie-recsys-api .
docker run -p 8000:8000 movie-recsys-api
```

## 🚀 Deploy the API to Render

The repository includes a `render.yaml` Blueprint for the FastAPI backend. It builds
the existing Dockerfile, checks `/health`, and uses Render's `PORT` environment variable.

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Review the `cinematch-api` web service and deploy it.
4. When the deploy finishes, open `https://<your-service>.onrender.com/health` and
   `https://<your-service>.onrender.com/docs`.

Point the Streamlit app at the deployed API by setting `API_URL` to the service URL
in the frontend host's environment variables or Streamlit secrets. For a local
PowerShell session:

```powershell
$env:API_URL = "https://<your-service>.onrender.com"
streamlit run app/streamlit_app.py
```

The free Render service can sleep after 15 minutes without traffic, so its first
request after sleeping may take about a minute. The model is trained during the
Docker image build, so the service does not need to download MovieLens data at startup.

## 📈 Evaluation Metrics

- **Precision@K:** Measures the proportion of recommended movies that are relevant.
- **Recall@K:** Measures the proportion of relevant movies successfully recommended.
- **NDCG@K:** Evaluates ranking quality by giving higher importance to relevant movies appearing near the top.
- **Catalog Coverage:** Measures the proportion of available movies that appear in recommendations.

## 🔮 Future Improvements

- Deploy the Streamlit frontend and connect it to the Render API.
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
