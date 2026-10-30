"""Central configuration: paths, seeds, evaluation settings."""
import os
from pathlib import Path

# RECSYS_HOME overrides the project root (the Docker image sets it to /app, because a
# pip-installed package no longer sits next to data/ and models/).
ROOT = Path(os.environ.get("RECSYS_HOME") or Path(__file__).resolve().parents[2])
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIRNAME = "ml-latest-small"

SEED = 42
TOP_K = 10
RELEVANCE_THRESHOLD = 4.0  # ratings >= this count as a "hit" in evaluation

VAL_FRAC = 0.1
TEST_FRAC = 0.1
MIN_USER_INTERACTIONS = 5  # users below this stay entirely in train

# Current best hyperparameters, chosen on the VALIDATION split via `python -m recsys.tune`.
# Update these after each tuning round; evaluate.py uses them for both variants of a model.
ALS_PARAMS = {"factors": 64, "regularization": 3.0, "alpha": 3.0}
KNN_PARAMS = {"n_neighbors": 200, "shrinkage": 0.0}
