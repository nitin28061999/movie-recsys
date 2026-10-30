"""CineMatch Streamlit Movie Recommendation Application."""

import os
import re

import requests
import streamlit as st

# Override with the API_URL environment variable when the backend runs elsewhere.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001").rstrip("/")

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #10131f 0%, #171b2e 100%);
        color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background: #111522;
        border-right: 1px solid #292f43;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(110deg, #252c49, #42245b);
        border: 1px solid #41476a;
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 36px;
        margin-bottom: 8px;
        color: white;
    }

    .hero p {
        color: #d5d8e8;
        font-size: 15px;
    }

    .movie-card {
        background: #20263a;
        border: 1px solid #343c55;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .movie-title {
        font-size: 17px;
        font-weight: 650;
        color: #ffffff;
    }

    .movie-meta {
        color: #aeb7d0;
        font-size: 13px;
        margin-top: 6px;
    }

    div[data-testid="stMetric"] {
        background: #20263a;
        border: 1px solid #343c55;
        padding: 14px;
        border-radius: 12px;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 700;
        margin: 14px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- API HELPERS --------------------

def api_get(endpoint, params=None, timeout=20):
    """Send GET request to FastAPI backend."""

    response = requests.get(
        f"{API_URL}{endpoint}",
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


# -------------------- WIKIPEDIA POSTER FETCHING --------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_wikipedia_poster(movie_title):
    """Fetch a movie poster thumbnail from Wikipedia."""

    try:
        match = re.search(r"\((\d{4})\)\s*$", movie_title)
        year = match.group(1) if match else ""

        clean_title = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            movie_title,
        ).strip()

        # Convert MovieLens titles:
        # Matrix, The -> The Matrix

        clean_title = re.sub(
            r",\s*(The|A|An)$",
            lambda m: f" {m.group(1)}",
            clean_title,
            flags=re.IGNORECASE,
        )

        search_query = f"{clean_title} film {year}".strip()

        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": search_query,
                "gsrlimit": 5,
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 342,
                "format": "json",
            },
            headers={
                "User-Agent": "CineMatchMovieRecommender/1.0"
            },
            timeout=8,
        )

        response.raise_for_status()

        pages = response.json().get(
            "query", {}
        ).get("pages", {})

        sorted_pages = sorted(
            pages.values(),
            key=lambda page: page.get("index", 999),
        )

        for page in sorted_pages:
            thumbnail = page.get(
                "thumbnail", {}
            ).get("source")

            if thumbnail:
                return thumbnail

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        return None

    return None


# -------------------- RESPONSE EXTRACTION --------------------

def extract_list(data, possible_keys):
    """Extract lists from API responses."""

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in possible_keys:

            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


# -------------------- MOVIE TITLE HELPER --------------------

def get_movie_details(movie_data):
    """Extract movie metadata, including title, score, genres, poster and ID."""

    if isinstance(movie_data, dict):

        title = movie_data.get("title", "Unknown movie")

        score = movie_data.get("score")

        genres = movie_data.get("genres")

        poster = (
            movie_data.get("poster_url")
            or movie_data.get("image_url")
            or movie_data.get("poster")
        )

        movie_id = movie_data.get(
            "movieId",
            movie_data.get("movie_id", movie_data.get("id")),
        )

    else:

        title = str(movie_data)
        score = None
        genres = None
        poster = None
        movie_id = None

    return title, score, genres, poster, movie_id


# -------------------- MOVIE CARD --------------------

def display_movie_card(
    movie_item,
    position=None,
    show_score=False,
):
    """Render a movie card with its poster, metadata and optional score."""

    title, score, genres, poster, movie_id = (
        get_movie_details(movie_item)
    )

    # Fetch Wikipedia poster if API poster is unavailable.

    if not (
        isinstance(poster, str)
        and poster.startswith(("https://", "http://"))
    ):
        poster = get_wikipedia_poster(title)

    with st.container(border=True):

        image_col, details_col = st.columns(
            [1, 4],
            vertical_alignment="center",
        )

        with image_col:

            if poster:
                st.image(poster, width=120)

            else:
                st.markdown("🎬")
                st.caption("Poster unavailable")

        with details_col:

            if position is not None:
                st.markdown(f"### {position}. {title}")
            else:
                st.markdown(f"### 🎞️ {title}")

            if show_score and score is not None:
                st.caption(
                    f"⭐ Recommendation score: {score}"
                )

            if genres:
                st.caption(f"Genres: {genres}")

            if movie_id is not None:
                st.caption(f"Movie ID: {movie_id}")


# -------------------- SIDEBAR --------------------

st.sidebar.title("🎬 CineMatch")

st.sidebar.caption(
    "Hybrid Movie Recommendation Engine"
)

st.sidebar.markdown("---")

st.sidebar.subheader("About the Project")

st.sidebar.write(
    "Discover movies through collaborative filtering: ALS and item-based "
    "kNN, blended by rank fusion."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Recommendation Models")

st.sidebar.markdown(
    """
    - **Serving:** Hybrid ALS + ItemKNN (rank fusion)
    - **Also evaluated:** Popularity, Content-Based, ItemKNN, ALS
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built with Python, FastAPI and Streamlit"
)


# -------------------- HERO HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Discover Your Next Favourite Movie</h1>

        <p>
            Personalized movie recommendations powered by a
            hybrid recommendation engine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- API HEALTH CHECK --------------------

try:
    health = api_get(
        "/health",
        timeout=5,
    )

    API_HEALTHY = (
        isinstance(health, dict)
        and health.get("status") == "ok"
    )

except (
    requests.RequestException,
    ValueError,
):
    API_HEALTHY = False

if not API_HEALTHY:
    st.error(
        f"Backend API is unavailable at {API_URL}. "
        "Start FastAPI and refresh this page."
    )
    st.stop()

st.success("Backend connected successfully")




# -------------------- LOAD USERS --------------------

try:
    with st.spinner("Loading available users..."):
        users_data = api_get(
            "/users",
            params={"limit": 100},
        )

    # Extract user IDs directly from the API response.
    if isinstance(users_data, dict):
        users = users_data.get("userIds", [])

        # Support alternative response formats.
        if not users:
            users = users_data.get(
                "users",
                users_data.get("results", users_data.get("data", [])),
            )
    elif isinstance(users_data, list):
        users = users_data
    else:
        users = []

    user_ids = []

    for user in users:
        if isinstance(user, dict):
            user_id = user.get(
                "userId",
                user.get("user_id", user.get("id")),
            )
        else:
            user_id = user

        if user_id is not None:
            user_ids.append(int(user_id))

    if not user_ids:
        st.error(f"No user IDs found. API response: {users_data}")
        st.stop()

except requests.RequestException as error:
    st.error(f"Could not load users: {error}")
    st.stop()

except (ValueError, TypeError) as error:
    st.error(f"Invalid users API response: {error}")
    st.stop()


# -------------------- USER SELECTION --------------------

st.sidebar.subheader("Your Profile")

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_ids,
)

number_of_movies = st.sidebar.slider(
    "Recommendations to display",
    min_value=5,
    max_value=20,
    value=10,
)


# -------------------- DASHBOARD METRICS --------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Available Users",
        len(user_ids),
    )

with col2:

    st.metric(
        "Selected User",
        selected_user,
    )

with col3:

    st.metric(
        "Recommendations",
        number_of_movies,
    )


# -------------------- MAIN TABS --------------------

recommendation_tab, search_tab = st.tabs(
    [
        "✨ Personalized Recommendations",
        "🔎 Search Movies",
    ]
)


# -------------------- RECOMMENDATIONS --------------------

with recommendation_tab:

    st.markdown(
        '<div class="section-heading">'
        'Recommended For You'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Personalized movie suggestions for User {selected_user}."
    )

    if st.button(
        "✨ Generate Recommendations",
        type="primary",
        use_container_width=True,
        key="recommend_button",
    ):

        try:

            with st.spinner(
                "Finding movies you may enjoy..."
            ):

                recommendations_data = api_get(
                    f"/recommendations/{selected_user}",
                    params={
                        "k": number_of_movies
                    },
                    timeout=30,
                )

            recommendations = extract_list(
                recommendations_data,
                [
                    "recommendations",
                    "results",
                    "movies",
                    "data",
                ],
            )

            if not recommendations:

                st.info(
                    "No recommendations found for this user."
                )

            else:

                st.success(
                    f"Found {len(recommendations)} recommendations."
                )

                if (
                    isinstance(recommendations_data, dict)
                    and recommendations_data.get("personalized") is False
                ):
                    st.info(
                        "This user has no usable rating history, so these "
                        "are popular movies rather than personalised picks."
                    )

                for index, movie in enumerate(
                    recommendations,
                    start=1,
                ):

                    display_movie_card(
                        movie,
                        position=index,
                        show_score=True,
                    )

        except requests.RequestException as error:

            st.error(
                f"Recommendation request failed: {error}"
            )

        except ValueError as error:

            st.error(
                f"Invalid recommendation response: {error}"
            )


# -------------------- MOVIE SEARCH --------------------

with search_tab:

    st.markdown(
        '<div class="section-heading">'
        'Explore Movies'
        '</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search for a movie",
        placeholder=(
            "Try Batman, Toy Story, or Star Wars..."
        ),
        key="movie_search",
    )

    if st.button(
        "🔎 Search Movies",
        type="primary",
        use_container_width=True,
        key="search_button",
    ):

        if not query.strip():

            st.warning(
                "Please enter a movie title."
            )

        else:

            try:

                with st.spinner(
                    "Searching the movie database..."
                ):

                    results_data = api_get(
                        "/movies/search",
                        params={
                            "q": query.strip(),
                            "limit": 10,
                        },
                    )

                movies = extract_list(
                    results_data,
                    [
                        "results",
                        "movies",
                        "data",
                    ],
                )

                if not movies:

                    st.info(
                        "No matching movies found."
                    )

                else:

                    st.success(
                        f"Found {len(movies)} matching movies."
                    )

                    for movie in movies:

                        display_movie_card(
                            movie,
                            show_score=False,
                        )

            except requests.RequestException as error:

                st.error(
                    f"Movie search failed: {error}"
                )

            except ValueError as error:

                st.error(
                    f"Invalid search response: {error}"
                )


# -------------------- FOOTER --------------------

st.markdown("---")

st.caption(
    "CineMatch | Hybrid Movie Recommendation System | "
    "Powered by MovieLens, FastAPI and Streamlit"
)
