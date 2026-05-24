from deepface import DeepFace
import cv2
import numpy as np
import time

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def detect_emotion_from_image(image_file):
    """
    Detects emotion from a Streamlit camera_input image (UploadedFile).
    Used on Streamlit Cloud where cv2.VideoCapture is unavailable.
    """
    bytes_data = image_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        emotions = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
        averaged_scores = {e: round(emotions.get(e, 0), 2) for e in EMOTIONS}
        dominant_emotion = max(averaged_scores, key=averaged_scores.get)
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": averaged_scores
        }
    except Exception as e:
        print(f"[detect_emotion] Image analysis error: {e}")
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {e: 0 for e in EMOTIONS}
        }


def detect_emotion(num_frames=5, delay=0.3):
    """
    Captures multiple webcam frames and averages emotion scores.
    Used locally where cv2.VideoCapture is available.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        return None

    emotion_totals = {emotion: 0 for emotion in EMOTIONS}
    successful_frames = 0

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            emotions = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
            for emotion in EMOTIONS:
                emotion_totals[emotion] += emotions.get(emotion, 0)
            successful_frames += 1
        except Exception as e:
            print(f"[detect_emotion] Frame error: {e}")
        time.sleep(delay)

    cap.release()

    if successful_frames == 0:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {e: 0 for e in EMOTIONS}
        }

    averaged_scores = {
        emotion: round(score / successful_frames, 2)
        for emotion, score in emotion_totals.items()
    }
    dominant_emotion = max(averaged_scores, key=averaged_scores.get)

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": averaged_scores
    }