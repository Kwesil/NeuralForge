from deepface import DeepFace
import cv2
import numpy as np
import time

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

def detect_emotion(num_frames=5, delay=0.3):
    """
    Captures multiple webcam frames and averages emotion scores for more stable emotion detection.
    """

    cap = cv2.VideoCapture(0)

    emotion_totals = {emotion: 0 for emotion in EMOTIONS}
    successful_frames = 0

    for _ in range(num_frames):

        ret, frame = cap.read()

        if not ret:
            return None
        
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                emotions = result[0]["emotion"]
            else:
                emotions = result["emotion"]
            
            for emotion in EMOTIONS:
                emotion_totals[emotion] += emotions.get(emotion, 0)

            successful_frames += 1    

        except Exception as e:
            print(f"Emotion detection error: {e}")

        time.sleep(delay)

    cap.release()

    # Fallback if no frames worked
    if successful_frames == 0:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {}
        }
    
    # Average scores
    averaged_scores = {
        emotion: round(score / successful_frames, 2)
        for emotion, score in emotion_totals.items()
    }

    dominant_emotion = max(
        averaged_scores,
        key=averaged_scores.get
    )

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": averaged_scores
    }