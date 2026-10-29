
"""FastAPI service for the MovieLens hybrid recommender."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from recsys import config
from recsys.data import download_movielens, load_movies, load_ratings
from recsys.models import (
    ALSRecommender,
    HybridRecommender,
    ItemKNNRecommender,
)


def build_recommender():
    """Fit the recommendation models on the available ratings history."""
    download_movielens()

    ratings = load_ratings()
    movies = load_movies()

    als = ALSRecommender(
        **config.ALS_PARAMS,
        min_rating=config.RELEVANCE_THRESHOLD,
    )
    knn = ItemKNNRecommender(
        **config.KNN_PARAMS,
        min_rating=config.RELEVANCE_THRESHOLD,
    )

    als.fit(ratings)
    knn.fit(ratings)

    hybrid = HybridRecommender(
        [(als, 1.0), (knn, 1.0)],
        prefit=True,
        name="hybrid-als+knn",
    )

    return hybrid, movies, ratings


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load data and fit models once when the API starts."""
    application.state.recommender, application.state.movies, application.state.ratings = (
        build_recommender()
    )
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="MovieLens hybrid recommender using ALS and ItemKNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/movies/search")
def search_movies(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Search movie titles by a case-insensitive substring."""
    movies = app.state.movies
    matches = movies[
        movies["title"].str.contains(q, case=False, na=False, regex=False)
    ].head(limit)

    return {
        "count": len(matches),
        "results": [
            {
                "movieId": int(row.movieId),
                "title": row.title,
                "genres": row.genres,
            }
            for row in matches.itertuples(index=False)
        ],
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50),
):
    """Return recommendations for a MovieLens user ID."""
    if user_id < 1:
        raise HTTPException(status_code=400, detail="user_id must be positive")

    recommender = app.state.recommender
    movie_lookup = app.state.movies.set_index("movieId")

    movie_ids = recommender.recommend(user_id, k)
    results = []

    for movie_id in movie_ids:
        if movie_id not in movie_lookup.index:
            continue

        movie = movie_lookup.loc[movie_id]

        results.append(
            {
                "movieId": int(movie_id),
                "title": movie["title"],
                "genres": movie["genres"],
            }
        )

    return {
        "userId": user_id,
        "model": recommender.name,
        "count": len(results),
        "recommendations": results,
    }


@app.get("/users")
def list_users(
    limit: int = Query(default=20, ge=1, le=100),
):
    """List sample users available in the ratings dataset."""
    user_ids = sorted(app.state.ratings["userId"].unique())[:limit]

    return {
        "count": len(user_ids),
        "userIds": [int(user_id) for user_id in user_ids],
    }
