"""FastAPI service for the MovieLens hybrid recommender."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from recsys.train import load_or_train


def build_recommender():
    """Load the trained recommender, training it first if no saved model exists."""
    return load_or_train()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load data and fit models once when the API starts."""
    recommender, movies, ratings = build_recommender()
    state = application.state
    state.recommender, state.movies, state.ratings = recommender, movies, ratings
    # Built once here instead of on every request.
    state.movie_lookup = movies.set_index("movieId")
    # Users at least one component can personalise for; everyone else gets popularity.
    state.personalized_users = {
        user
        for model, _ in recommender.components
        for user in model.user_index_  # pyright: ignore[reportAttributeAccessIssue]
    }
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="MovieLens hybrid recommender using ALS and ItemKNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Confirm the service is up and point to the interactive docs."""
    return {
        "message": "Movie Recommendation API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Liveness check for hosting platforms and monitors."""
    return {"status": "ok"}


@app.get("/movies/search")
def search_movies(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Search movie titles by a case-insensitive substring."""
    movies = app.state.movies
    matches = movies[movies["title"].str.contains(q, case=False, na=False, regex=False)].head(limit)

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
    """Return recommendations for a MovieLens user ID.

    Users with no usable rating history get the most popular unseen movies instead;
    `personalized` and `strategy` say which of the two the caller received.
    """
    if user_id < 1:
        raise HTTPException(status_code=400, detail="user_id must be positive")

    recommender = app.state.recommender
    movie_lookup = app.state.movie_lookup
    personalized = user_id in app.state.personalized_users

    results = []
    for movie_id in recommender.recommend(user_id, k):
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
        "personalized": personalized,
        "strategy": "hybrid" if personalized else "popularity-fallback",
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
