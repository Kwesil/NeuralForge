import streamlit as st
from emotion.detect_emotion import detect_emotion

st.title("NeuralForge")
st.subheader("Emotion-Aware Restaurant Recommendation Agent")
    
recommendations = {
    "happy": ["Sushi Spot", "BBQ House"],
    "sad": ["Comfort Bites", "Warm Soup Cafe"],
    "angry": ["Quiet Garden Reataurant"],
    "neutral": ["City Grill"]
}

if st.button("Analyze Emotion"):
    emotion = detect_emotion()

    st.success(f"Detected Emotion: {emotion}")

    recs = recommendations.get(emotion, ["City Grill"])
    
    st.write("recommendation Restaurants:")

    for r in recs:
        st.write(f"- {r}")