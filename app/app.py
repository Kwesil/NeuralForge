import streamlit as st
from emotion.detect_emotion import detect_emotion

st.title("NeuralForge")
st.subheader("Emotion-Aware Restaurant Recommendation Agent")

if st.button("Analyze Emotion"):
    emotion = detect_emotion()

    st.success(f"Detected Emotion: {emotion}")
    