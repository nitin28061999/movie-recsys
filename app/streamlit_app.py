"""CineMatch Streamlit Movie Recommendation Application."""

import hashlib
import os
import re
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

def get_api_url():
    """Read the backend URL from an environment variable or Streamlit secrets."""

    url = os.environ.get("API_URL")
    if url:
        return url.rstrip("/")
    try:
        url = st.secrets.get("API_URL")
        return (url or "http://127.0.0.1:8001").rstrip("/")
    except (StreamlitSecretNotFoundError, KeyError):
        return "http://127.0.0.1:8001"

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = get_api_url()


# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 44px;
        border-radius: 18px;
        background: linear-gradient(115deg, rgba(9,9,9,.96) 5%, rgba(35,13,17,.91) 54%, rgba(89,8,16,.88)),
                    radial-gradient(circle at 82% 22%, #e50914 0%, #4a1117 35%, #111 72%);
        border: 1px solid #51242a;
        box-shadow: 0 18px 50px rgba(0,0,0,.34);
        margin: 10px 0 26px;
    }

    .hero h1 {
        max-width: 760px;
        font-size: clamp(32px, 4vw, 54px);
        line-height: 1.04;
        letter-spacing: -1.8px;
        margin: 8px 0 12px;
        color: #fff;
    }

    .hero .hero-kicker {
        color: #ff5962;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2.4px;
        margin: 0;
    }

    .hero .hero-copy {
        color: #d0c8c9;
        font-size: 16px;
        max-width: 620px;
        margin: 0;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1c1c1c, #131313);
        border: 1px solid #303030;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }

    [data-testid="stMetricValue"] {
        color: #fff;
        font-weight: 750;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #ff4b55;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 760;
        letter-spacing: -.4px;
        margin: 20px 0 6px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #303030;
        background: linear-gradient(160deg, #191919, #111111);
        transition: transform .18s ease, border-color .18s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #8f2229;
        transform: translateY(-3px);
    }

    [data-testid="stButton"] button[kind="primary"] {
        font-weight: 750;
        box-shadow: 0 8px 24px rgba(229,9,20,.2);
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


# -------------------- MOVIE POSTERS (TMDB) --------------------

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"


def get_tmdb_key():
    """Read the TMDB key from the TMDB_API_KEY env var or Streamlit secrets (None if unset)."""

    key = os.environ.get("TMDB_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("TMDB_API_KEY")
    except (StreamlitSecretNotFoundError, KeyError):  # no secrets file or no such key
        return None


def clean_movielens_title(movie_title):
    """Turn 'Matrix, The (1999)' into ('The Matrix', '1999')."""

    match = re.search(r"\((\d{4})\)\s*$", movie_title)
    year = match.group(1) if match else None

    title = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_title)
    title = re.sub(r"\s*\([^)]*\)", "", title).strip()  # drop alternate titles

    article = re.match(r"^(.*),\s*(The|A|An)$", title, flags=re.IGNORECASE)

    if article:
        title = f"{article.group(2)} {article.group(1)}"

    return title, year


@st.cache_data(show_spinner=False, max_entries=500)
def make_local_poster(movie_title, genres=""):
    """Create original, colorful poster art when no external poster is available."""

    width, height = 480, 720
    seed = int(hashlib.sha256(f"{movie_title}|{genres}".encode()).hexdigest()[:8], 16)
    palettes = [
        ((120, 15, 28), (12, 15, 29), (240, 69, 75)),
        ((26, 75, 104), (13, 17, 39), (77, 199, 220)),
        ((123, 67, 17), (28, 16, 28), (255, 184, 76)),
        ((75, 31, 106), (15, 14, 36), (197, 107, 239)),
        ((17, 104, 76), (10, 21, 29), (87, 222, 153)),
    ]
    top, bottom, accent = palettes[seed % len(palettes)]
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    for y in range(height):
        mix = y / height
        color = tuple(round(a * (1 - mix) + b * mix) for a, b in zip(top, bottom))
        draw.line((0, y, width, y), fill=color)

    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((190, 90, 650, 540), fill=(*accent, 46))
    glow_draw.ellipse((-210, 330, 280, 820), fill=(*accent, 24))
    image = Image.alpha_composite(image.convert("RGBA"), glow)
    draw = ImageDraw.Draw(image)

    try:
        font_brand = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        font_meta = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font_brand = ImageFont.load_default(size=20)
        font_title = ImageFont.load_default(size=42)
        font_meta = ImageFont.load_default(size=19)

    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=18, outline=(255, 255, 255, 72), width=2)
    draw.rounded_rectangle((50, 54, 58, 110), radius=4, fill=(*accent, 255))
    draw.text((74, 65), "CINEMATCH", font=font_brand, fill=(255, 255, 255, 235))
    draw.text((50, 385), "TONIGHT'S PICK", font=font_meta, fill=(*accent, 255))

    title, year = clean_movielens_title(movie_title)
    words = title.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font_title)[2] > width - 100 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 4:
        lines = lines[:3] + [lines[3].rstrip(".,:;!") + "…"]

    y = 430
    for line in lines:
        draw.text((50, y), line, font=font_title, fill=(255, 255, 255, 255), stroke_width=1)
        y += 52
    footer = "  •  ".join((genres or "Movie night").split("|")[:3])
    if year:
        footer = f"{year}  •  {footer}"
    draw.text((50, height - 88), footer[:42], font=font_meta, fill=(238, 229, 229, 235))

    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(movie_title, api_key):
    """Look up a poster on TMDB. Returns None when there is no key or no match.

    The key is an argument so the cache is not poisoned by lookups made without one.
    """

    if not api_key:
        return None

    title, year = clean_movielens_title(movie_title)

    params = {"query": title, "include_adult": "false"}
    headers = {}

    if year:
        params["year"] = year

    if api_key.startswith("eyJ"):  # v4 "API Read Access Token"
        headers["Authorization"] = f"Bearer {api_key}"
    else:  # v3 API key
        params["api_key"] = api_key

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()

        for result in response.json().get("results", []):
            if result.get("poster_path"):
                return f"{TMDB_IMAGE_BASE}{result['poster_path']}"

    except (requests.RequestException, ValueError):
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
    """Render a compact title card with a real or locally generated cover."""

    title, score, genres, poster, movie_id = (
        get_movie_details(movie_item)
    )

    # Look up a TMDB poster if the API did not supply one.

    if not (
        isinstance(poster, str)
        and poster.startswith(("https://", "http://"))
    ):
        poster = get_poster(title, get_tmdb_key())
    if not poster:
        poster = make_local_poster(title, genres or "")

    with st.container(border=True, gap="small"):
        if position is not None:
            st.badge(f"#{position} for you", color="red")
        st.image(poster, width="stretch")
        st.markdown(f"**{title}**")
        if show_score and score is not None:
            st.caption(f"Recommendation score: {score}")
        if genres:
            st.caption(" · ".join(genres.split("|")[:3]))
        if movie_id is not None:
            st.caption(f"MovieLens ID {movie_id}")


def display_movie_shelf(movies, show_score=False):
    """Lay out results in poster rows that adapt to narrow screens."""

    for offset in range(0, len(movies), 5):
        columns = st.columns(5, gap="small")
        for index, (column, movie) in enumerate(zip(columns, movies[offset : offset + 5])):
            with column:
                display_movie_card(
                    movie,
                    position=offset + index + 1,
                    show_score=show_score,
                )


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
    '<div class="hero">'
    '<p class="hero-kicker">YOUR NEXT MOVIE NIGHT STARTS HERE</p>'
    "<h1>Stories picked for you.</h1>"
    '<p class="hero-copy">Discover your next favourite with recommendations shaped by movie fans like you.</p>'
    "</div>",
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

                display_movie_shelf(recommendations, show_score=True)

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

                    display_movie_shelf(movies, show_score=False)

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

if get_tmdb_key():
    st.caption(
        "This product uses the TMDB API but is not endorsed or certified by TMDB."
    )
