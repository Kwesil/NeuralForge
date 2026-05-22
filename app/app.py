import streamlit as st
from emotion.detect_emotion import detect_emotion
from utils.context_mapper import map_emotion_to_context
from utils.load_data import load_restaurants
from recommender.recommend import filter_restaurants
from recommender.embedding_model import (create_combuned_features, create_embeddings)
from recommender.ranker import recommend_restaurants
from utils.context_query import context_to_query

st.set_page_config(page_title="NeuralForge", layout="centered")

st.title("NeuralForge")
st.subheader("Emotion-Aware Restaurant Recommendation Agent")
    
restaurant_df = load_restaurants(
    "data/yelp_academic_dataset_business.json"
)

restaurant_df = create_combuned_features(
    restaurant_df
)

embeddings = create_embeddings(
    restaurant_df
)

if st.button("Analyze Emotion"):
    
    with st.spinner("Analysing emotional context..."):

        result = detect_emotion()

        dominant_emotion = result["dominant_emotion"]
        emotion_scores = result["emotion_scores"]

        behavioral_context = map_emotion_to_context(emotion_scores)

        query = context_to_query(
            behavioral_context["context"]
        )
        recommendations = recommend_restaurants(
            query,
            restaurant_df,
            embeddings
        )

    st.success(f"Detected Emotion: {dominant_emotion}")

    st.write("### Emotion Confidence Scores")
    st.json(emotion_scores)

    st.write("### Behavioral Context")
    st.info(behavioral_context["context"])

    st.write(behavioral_context["description"])

    st.write("### Recommended Restaurants")

    for _, restaurant in recommendations.iterrows():

        st.markdown(f"""
        ### 🍽️{restaurant['name']}

        **City:** {restaurant['city']}
        **Rating:** ⭐{restaurant['stars']}
        **Categories:** {restaurant['categories']}
        """)