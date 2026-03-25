import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ---------- Base ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e0d5;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0f0f18; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ---------- Hero ---------- */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(220,160,60,0.18), transparent);
    border-bottom: 1px solid rgba(220,160,60,0.12);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #dca03c;
    background: rgba(220,160,60,0.1);
    border: 1px solid rgba(220,160,60,0.3);
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 900;
    line-height: 1.05;
    color: #f0e6d3;
    margin: 0 0 0.6rem;
    letter-spacing: -0.02em;
}
.hero h1 span { color: #dca03c; }
.hero p {
    font-size: 1.05rem;
    color: #8a8070;
    font-weight: 300;
    margin: 0;
}

/* ---------- Search Box ---------- */
[data-testid="stSelectbox"] > div > div {
    background: #14141e !important;
    border: 1.5px solid rgba(220,160,60,0.25) !important;
    border-radius: 12px !important;
    color: #e8e0d5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.1rem 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #dca03c !important;
    box-shadow: 0 0 0 3px rgba(220,160,60,0.1) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6a6258 !important;
    margin-bottom: 0.5rem !important;
}

/* ---------- Button ---------- */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #dca03c, #c8873a) !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2.5rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100%;
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #e8b050, #dca03c) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(220,160,60,0.35) !important;
}

/* ---------- Selected Movie Card ---------- */
.movie-card-hero {
    background: linear-gradient(135deg, #14141e 0%, #1a1428 100%);
    border: 1px solid rgba(220,160,60,0.2);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.movie-card-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #dca03c, transparent);
}
.movie-title-big {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0e6d3;
    margin: 0 0 0.5rem;
}
.movie-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
    align-items: center;
}
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #c8bfb0;
}
.rating-pill {
    background: rgba(220,160,60,0.15);
    border: 1px solid rgba(220,160,60,0.35);
    color: #dca03c;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.25rem 0.85rem;
    border-radius: 100px;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.movie-overview {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #a09488;
    font-weight: 300;
    margin-top: 0.8rem;
}
.tagline {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    color: #5a5248;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}

/* ---------- Section Heading ---------- */
.section-heading {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1.5rem;
}
.section-heading h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #f0e6d3;
    margin: 0;
    font-weight: 700;
}
.section-heading .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(220,160,60,0.3), transparent);
}

/* ---------- Recommendation Cards ---------- */
.rec-card {
    background: #11111a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    height: 100%;
    transition: all 0.25s ease;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.rec-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(220,160,60,0.4), transparent);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}
.rec-card:hover {
    background: #16161f;
    border-color: rgba(220,160,60,0.2);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
.rec-card:hover::after { transform: scaleX(1); }
.rec-num {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: #dca03c;
    opacity: 0.6;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}
.rec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e0d5;
    margin-bottom: 0.6rem;
    line-height: 1.3;
}
.rec-overview {
    font-size: 0.82rem;
    color: #6a6258;
    line-height: 1.6;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.rec-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.star-row { color: #dca03c; font-size: 0.8rem; letter-spacing: 0.05em; }
.pop-badge {
    font-size: 0.72rem;
    color: #5a5248;
    font-weight: 400;
}

/* ---------- Stats Row ---------- */
.stat-box {
    background: #11111a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #dca03c;
}
.stat-label {
    font-size: 0.75rem;
    color: #5a5248;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 4rem;
    color: #3a3228;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}

/* ---------- Divider ---------- */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2420; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_data():
    # Load CSV instead of pkl
    df = pd.read_csv("Movies.csv")

    # Clean data
    df['overview'] = df['overview'].fillna('')
    df['title'] = df['title'].fillna('')

    # TF-IDF creation (replaces tfidf_matrix.pkl)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(df['overview'])

    # Create indices (replaces indices.pkl)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, indices, tfidf_matrix
df, indices, tfidf_matrix = load_data()


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(str(genre_str))
        if isinstance(genres, list):
            return [g['name'] for g in genres if isinstance(g, dict)]
        return []
    except Exception:
        words = re.findall(r"'name':\s*'([^']+)'", str(genre_str))
        return words

def star_string(rating):
    try:
        r = float(rating)
        full = int(r / 2)
        half = 1 if (r / 2 - full) >= 0.5 else 0
        empty = 5 - full - half
        return "★" * full + ("½" if half else "") + "☆" * empty
    except Exception:
        return "—"

def recommend(title, n=9):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    similar_idx = [i for i in similar_idx if i < len(df)]
    return df.iloc[similar_idx][['title','overview','genres','tagline','vote_average','popularity']].to_dict('records')

def get_movie_info(title):
    if title not in indices:
        return None
    idx = indices[title]
    if idx < len(df):
        return df.iloc[idx].to_dict()
    return None


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Content-Based Recommender</div>
    <h1>Cine<span>Match</span></h1>
    <p>Discover your next obsession — powered by 45,000+ films</p>
</div>
""", unsafe_allow_html=True)


# ── Stats Row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-num">45K+</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
with c2:
    avg_rating = df['vote_average'].mean()
    st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_rating:.1f}</div><div class="stat-label">Avg Rating</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><div class="stat-num">TF-IDF</div><div class="stat-label">Algorithm</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="stat-num">∞</div><div class="stat-label">Discoveries</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
all_titles = sorted(df['title'].dropna().unique().tolist())

col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected_movie = st.selectbox(
        "Search for a movie",
        options=[""] + all_titles,
        index=0,
        format_func=lambda x: "🎬  Type or select a movie title..." if x == "" else x,
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)  # align button
    find_btn = st.button("Find Movies", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
if selected_movie and (find_btn or selected_movie):

    movie_info = get_movie_info(selected_movie)

    # Selected movie details
    if movie_info:
        title = movie_info.get('title', selected_movie)
        overview = movie_info.get('overview', 'No overview available.')
        tagline = movie_info.get('tagline', '')
        rating = movie_info.get('vote_average', 0)
        popularity = movie_info.get('popularity', 0)
        genres = parse_genres(movie_info.get('genres', ''))

        genre_badges = " ".join([f'<span class="badge">{g}</span>' for g in genres]) if genres else ""
        tagline_html = f'<div class="tagline">"{tagline}"</div>' if tagline and str(tagline) != 'nan' else ""

        try:
            pop_val = f"{float(popularity):.1f}"
        except Exception:
            pop_val = str(popularity)

        st.markdown(f"""
        <div class="movie-card-hero">
            <div class="movie-title-big">🎬 {title}</div>
            {tagline_html}
            <div class="movie-meta">
                <span class="rating-pill">★ {rating}</span>
                {genre_badges}
                <span class="badge">Popularity {pop_val}</span>
            </div>
            <div class="movie-overview">{overview}</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    recs = recommend(selected_movie, n=9)

    if recs:
        st.markdown("""
        <div class="section-heading">
            <h2>You Might Also Like</h2>
            <div class="line"></div>
        </div>
        """, unsafe_allow_html=True)

        for row_start in range(0, len(recs), 3):
            cols = st.columns(3, gap="medium")
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx >= len(recs):
                    break
                rec = recs[idx]
                rec_title = rec.get('title', 'Unknown')
                rec_overview = rec.get('overview', 'No overview available.')
                rec_rating = rec.get('vote_average', 0)
                rec_genres = parse_genres(rec.get('genres', ''))
                rec_pop = rec.get('popularity', 0)
                stars = star_string(rec_rating)

                genre_str = " · ".join(rec_genres[:3]) if rec_genres else "—"
                try:
                    pop_str = f"Popularity: {float(rec_pop):.1f}"
                except Exception:
                    pop_str = ""

                with col:
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-num">№ {idx + 1:02d}</div>
                        <div class="rec-title">{rec_title}</div>
                        <div style="font-size:0.72rem; color:#5a5248; margin-bottom:0.5rem;">{genre_str}</div>
                        <div class="rec-overview">{rec_overview}</div>
                        <div class="rec-footer">
                            <span class="star-row">{stars} <span style="color:#5a5248; font-family: DM Sans;">{rec_rating}</span></span>
                            <span class="pop-badge">{pop_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this title. Try another movie!")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color: #3a3228;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4;">🎞️</div>
        <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #4a4238; margin-bottom: 0.5rem;">
            Start with any film you love
        </div>
        <div style="font-size: 0.85rem; color: #2a2220;">
            Search for a movie above and hit <em>Find Movies</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · TF-IDF Content-Based Filtering · 45K+ Movies
</div>
""", unsafe_allow_html=True)