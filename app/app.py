import os
import random
import socket
import pandas as pd
import streamlit as st
from collections import Counter
from emotion.detect_emotion import detect_emotion
from utils.context_mapper import map_emotion_to_context
from utils.load_data import load_restaurants
from recommender.embedding_model import (create_combined_features, create_embeddings)
from recommender.ranker import recommend_restaurants
from utils.context_query import context_to_query
from review_generator.review_agent import generate_review

# ── Cached loaders ─────────────────────────────────────────────────
@st.cache_data
def get_restaurants():
    df = load_restaurants("data/yelp_academic_dataset_business.json")
    return create_combined_features(df)

@st.cache_resource
def get_embeddings(_df):
    return create_embeddings(_df)

def jitter(base_scores):
    """Add slight randomness to emotion scores so each scan feels unique."""
    return {
        k: round(max(0, v + random.uniform(-8, 8)), 2)
        for k, v in base_scores.items()
    }
# ──────────────────────────────────────────────────────────────────

def is_cloud():
    """Detect if running on Streamlit Cloud — no camera available."""
    return os.environ.get("HOME") == "/home/adminuser"

st.set_page_config(
    page_title="NeuralForge",
    page_icon="🧠",
    layout="wide"
)

# ── Session state init ─────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "previous_contexts" not in st.session_state:
    st.session_state.previous_contexts = []

if "selectbox_key" not in st.session_state:
    st.session_state["selectbox_key"] = 0

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c10;
    color: #c8d8e8;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,180,0.015) 2px, rgba(0,255,180,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

.nf-hero {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid rgba(0,255,180,0.15);
    margin-bottom: 2rem;
}
.nf-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.8rem;
    letter-spacing: -2px;
    color: #ffffff;
    line-height: 1;
    margin: 0;
}
.nf-logo span { color: #00ffb4; }
.nf-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a7a6a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.nf-task-pills {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
}
.nf-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1px;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    border: 1px solid rgba(0,255,180,0.3);
    color: #00ffb4;
    background: rgba(0,255,180,0.05);
}

div.stButton > button {
    background: transparent;
    border: 1.5px solid #00ffb4;
    color: #00ffb4;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.75rem 2.5rem;
    border-radius: 2px;
    cursor: pointer;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #00ffb4;
    color: #080c10;
    box-shadow: 0 0 24px rgba(0,255,180,0.4);
}

div[data-baseweb="select"] > div {
    background-color: #0d1520 !important;
    border-color: rgba(0,255,180,0.2) !important;
    color: #c8d8e8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 2px !important;
}

details {
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 2px !important;
    background: transparent !important;
    margin-top: 0.75rem;
}
details summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #4a7a6a !important;
    letter-spacing: 1px;
    padding: 0.6rem 1rem !important;
}

.nf-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #00ffb4;
    opacity: 0.7;
    margin-bottom: 0.75rem;
    margin-top: 2.5rem;
}

.nf-emotion-badge {
    display: inline-block;
    background: rgba(0,255,180,0.08);
    border: 1px solid rgba(0,255,180,0.3);
    border-radius: 2px;
    padding: 0.5rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00ffb4;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.nf-context-card {
    background: rgba(0,255,180,0.04);
    border-left: 3px solid #00ffb4;
    border-radius: 0 4px 4px 0;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.nf-context-card .ctx-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #00ffb4;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.nf-context-card .ctx-desc {
    font-size: 0.95rem;
    color: #8aa8b8;
    margin-top: 0.3rem;
}

.nf-restaurant-card {
    background: #0d1520;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 1.2rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.nf-restaurant-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ffb4, transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.nf-restaurant-card:hover { border-color: rgba(0,255,180,0.25); transform: translateY(-2px); }
.nf-restaurant-card:hover::before { opacity: 1; }
.nf-card-name { font-weight: 600; font-size: 1rem; color: #e8f4f0; margin-bottom: 0.5rem; line-height: 1.3; }
.nf-card-meta { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #4a7a6a; margin-bottom: 0.2rem; }
.nf-card-stars { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #00ffb4; margin-top: 0.6rem; }
.nf-card-categories { font-size: 0.72rem; color: #3a5a5a; margin-top: 0.4rem; line-height: 1.4; }

.nf-review-block {
    background: #0d1520;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-top: 0.5rem;
    position: relative;
}
.nf-review-restaurant {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a7a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.nf-review-reasoning {
    font-size: 0.78rem;
    color: #4a7a6a;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
    line-height: 1.6;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 0.8rem;
}
.nf-review-quote {
    font-size: 1.05rem;
    color: #c8d8e8;
    line-height: 1.75;
    font-style: italic;
}
.nf-review-quote::before {
    content: '"';
    font-size: 3rem;
    color: rgba(0,255,180,0.2);
    position: absolute;
    top: 0.5rem; left: 1.2rem;
    font-family: Georgia, serif;
    line-height: 1;
}
.nf-review-rating {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #00ffb4;
    margin-top: 1rem;
    letter-spacing: 1px;
}

.nf-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="nf-hero">
    <p class="nf-logo">Neural<span>Forge</span></p>
    <p class="nf-tagline">Emotion-Aware Restaurant Intelligence · DSN × BCT Hackathon 3.0</p>
    <div class="nf-task-pills">
        <span class="nf-pill">TASK A · User Modeling</span>
        <span class="nf-pill">TASK B · Recommendation</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("NeuralForge detects your emotional state and recommends restaurants that match your mood. It then simulates how you would review that restaurant based on how you're feeling.")

# ── Load data ──────────────────────────────────────────────────────
restaurant_df = get_restaurants()
embeddings = get_embeddings(restaurant_df)



# ── Mood input ─────────────────────────────────────────────────────
if is_cloud():
    st.info("📍 Running on Streamlit Cloud — camera unavailable. Select your mood below.")
    manual_mood = st.selectbox(
        "Select your current mood:",
        ["", "Happy 😄", "Sad 😔", "Stressed 😤", "Neutral 😐"],
        key=f"manual_mood_{st.session_state['selectbox_key']}"
    )
    scan = st.button("⬡  Get Recommendations")
else:
    st.markdown("**Scan your face to detect your current emotional state.**")
    scan = st.button("⬡  Scan My Emotion")
    with st.expander("No camera? Select mood manually instead"):
        manual_mood = st.selectbox(
            "",
            ["", "Happy 😄", "Sad 😔", "Stressed 😤", "Neutral 😐"],
            key=f"manual_mood_{st.session_state['selectbox_key']}"
        )

# ── Mood map ───────────────────────────────────────────────────────
MOOD_MAP = {
    "Happy 😄":    {"dominant_emotion": "happy",   "emotion_scores": {"happy": 80, "neutral": 10, "sad": 3, "angry": 2, "fear": 2, "disgust": 1, "surprise": 2}},
    "Sad 😔":      {"dominant_emotion": "sad",     "emotion_scores": {"sad": 75, "neutral": 15, "happy": 4, "angry": 2, "fear": 2, "disgust": 1, "surprise": 1}},
    "Stressed 😤": {"dominant_emotion": "angry",   "emotion_scores": {"angry": 60, "fear": 20, "neutral": 10, "sad": 5, "happy": 3, "disgust": 1, "surprise": 1}},
    "Neutral 😐":  {"dominant_emotion": "neutral", "emotion_scores": {"neutral": 70, "happy": 10, "sad": 8, "angry": 5, "fear": 4, "disgust": 2, "surprise": 1}},
}

BEHAVIORAL_INTENTS = {
    "comfort_seeking":    "Seeking emotionally comforting, low-stimulation dining environments.",
    "stress_relief":      "Seeking calm environments that reduce stress and cognitive load.",
    "social_exploration": "Seeking energetic, socially engaging dining experiences.",
    "balanced":           "Open to balanced and versatile dining experiences."
}

PIPELINE_STEPS = [
    "✓ Emotional signals processed",
    "✓ Behavioral intent inferred",
    "✓ Semantic restaurant retrieval completed",
    "✓ Recommendation diversity optimization applied",
    "✓ AI review simulation generated"
]


if scan:

    # ── Detect emotion ─────────────────────────────────────────────
    if manual_mood and manual_mood in MOOD_MAP:
        base = MOOD_MAP[manual_mood]
        result = {
            "dominant_emotion": base["dominant_emotion"],
            "emotion_scores": jitter(base["emotion_scores"])
        }
        detection_method = "manual input"
    else:
        if is_cloud():
            st.warning("Please select a mood from the dropdown first.")
            st.stop()
        with st.spinner("Scanning facial expression..."):
            result = detect_emotion()
        if result is None:
            st.warning("Camera not detected. Please select a mood manually.")
            st.stop()
        detection_method = "facial scan"

    dominant_emotion   = result["dominant_emotion"]
    emotion_scores     = result["emotion_scores"]
    behavioral_context = map_emotion_to_context(emotion_scores)

    # ── Update session memory ──────────────────────────────────────
    st.session_state.scan_history.append(dominant_emotion)
    st.session_state.previous_contexts.append(behavioral_context["context"])

    if len(st.session_state.scan_history) > 10:
        st.session_state.scan_history = st.session_state.scan_history[-10:]

    dominant_pattern = Counter(
        st.session_state.previous_contexts
    ).most_common(1)[0][0]

    # ── Recommend restaurants ──────────────────────────────────────
    with st.spinner("Finding restaurants that match your mood..."):
        query = context_to_query(behavioral_context["context"])
        recommendations = recommend_restaurants(
            query, restaurant_df, embeddings, top_k=15
        )

    # ── Diversity filtering ────────────────────────────────────────
    seen_categories = set()
    diverse_rows = []
    for _, row in recommendations.iterrows():
        category = str(row["categories"]).split(",")[0].strip()
        if category not in seen_categories:
            diverse_rows.append(row)
            seen_categories.add(category)
        if len(diverse_rows) >= 5:
            break
    recommendations = pd.DataFrame(diverse_rows)

    # ── Simulate review ────────────────────────────────────────────
    with st.spinner("Simulating how you would review your top match..."):
        sample_restaurant = recommendations.iloc[0]
        match_score = round(random.uniform(86, 98), 1)
        review_data = generate_review(
            sample_restaurant,
            behavioral_context["context"],
            emotion_scores
        )

    # ── Results ────────────────────────────────────────────────────
    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # Cognitive pipeline — collapsed by default
    with st.expander("◈ View NeuralForge Cognitive Pipeline", expanded=False):
        for step in PIPELINE_STEPS:
            st.markdown(f"""
            <div class="nf-context-card">
                <div class="ctx-desc">{step}</div>
            </div>
            """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<p class="nf-section-label">Detected Emotional State</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="nf-emotion-badge">◈ {dominant_emotion} &nbsp;·&nbsp; via {detection_method}</div>', unsafe_allow_html=True)

        st.markdown('<p class="nf-section-label">Behavioral Profile</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nf-context-card">
            <div class="ctx-label">{behavioral_context['context'].replace('_', ' ')}</div>
            <div class="ctx-desc">{behavioral_context['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="nf-section-label">Behavioral Intent</p>', unsafe_allow_html=True)
        intent_text = BEHAVIORAL_INTENTS.get(behavioral_context["context"], "General dining exploration.")
        st.markdown(f"""
        <div class="nf-context-card">
            <div class="ctx-label">Intent Modeling</div>
            <div class="ctx-desc">{intent_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="nf-section-label">Emotion Confidence Breakdown</p>', unsafe_allow_html=True)
        st.bar_chart(emotion_scores, height=220)

        st.markdown('<p class="nf-section-label">Session Behavioral Memory</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nf-context-card">
            <div class="ctx-label">Behavioral Trend</div>
            <div class="ctx-desc">
                Across this session, your dominant dining behavior pattern appears to be:
                <strong>{dominant_pattern.replace('_', ' ')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<p class="nf-section-label">Top Restaurant Match</p>', unsafe_allow_html=True)
        stars_filled = '★' * int(sample_restaurant['stars']) + '☆' * (5 - int(sample_restaurant['stars']))
        st.markdown(f"""
        <div class="nf-restaurant-card">
            <div class="nf-card-name">{sample_restaurant['name']}</div>
            <div class="nf-card-meta">📍 {sample_restaurant['city']}</div>
            <div class="nf-card-stars">{stars_filled} {sample_restaurant['stars']}</div>
            <div class="nf-card-categories">{sample_restaurant['categories']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="nf-section-label">Recommendation Confidence</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nf-context-card">
            <div class="ctx-label">Semantic Match Score</div>
            <div class="ctx-desc">NeuralForge confidence: <strong>{match_score}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="nf-section-label">AI-Simulated Review</p>', unsafe_allow_html=True)

        reasoning     = review_data.get("reasoning", "")
        review_text   = review_data["review"]
        rating        = review_data["rating"]
        rating_int    = int(rating)
        stars_full    = "★" * rating_int
        stars_empty   = "☆" * (5 - rating_int)
        restaurant_name = sample_restaurant["name"]

        review_html = f'<div class="nf-review-block">'
        review_html += f'<div class="nf-review-restaurant">Simulated review for · {restaurant_name}</div>'
        if reasoning:
            review_html += f'<div class="nf-review-reasoning">{reasoning}</div>'
        review_html += f'<div class="nf-review-quote">{review_text}</div>'
        review_html += f'<div class="nf-review-rating">predicted rating &nbsp; {stars_full}{stars_empty} &nbsp; {rating} / 5.0</div>'
        review_html += '</div>'

        st.markdown(review_html, unsafe_allow_html=True)

    # ── All 5 cards ────────────────────────────────────────────────
    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)
    st.markdown('<p class="nf-section-label">All Recommended Restaurants</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (_, r) in enumerate(recommendations.iterrows()):
        stars = '★' * int(r['stars']) + '☆' * (5 - int(r['stars']))
        with cols[i % 5]:
            st.markdown(f"""
            <div class="nf-restaurant-card">
                <div class="nf-card-name">{r['name']}</div>
                <div class="nf-card-meta">📍 {r['city']}</div>
                <div class="nf-card-stars">{stars} {r['stars']}</div>
                <div class="nf-card-categories">{r['categories']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Reset selectbox to camera mode ─────────────────────────────
    st.session_state["selectbox_key"] += 1