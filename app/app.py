import streamlit as st
from emotion.detect_emotion import detect_emotion
from utils.context_mapper import map_emotion_to_context

st.set_page_config(page_title="NeuralForge", layout="centered")

st.title("NeuralForge")
st.subheader("Emotion-Aware Restaurant Recommendation Agent")
    
recommendations = {
    "happy": ["Sushi Spot", "BBQ House"],
    "sad": ["Comfort Bites", "Warm Soup Cafe"],
    "angry": ["Quiet Garden Reataurant"],
    "neutral": ["City Grill"]
}

if st.button("Analyze Emotion"):
    
    with st.spinner("Analysing emotional context..."):

        result = detect_emotion()

        dominant_emotion = result["dominant_emotion"]
        emotion_scores = result["emotion_scores"]

        behavioral_context = map_emotion_to_context(emotion_scores)

    st.success(f"Detected Emotion: {dominant_emotion}")

    st.write("### Emotion Confidence Scores")
    st.json(emotion_scores)

    st.write("### Behavioral Context")
    st.info(behavioral_context["context"])

    st.write(behavioral_context["description"])