from deepface import DeepFace
import cv2

def detect_emotion():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    if not ret:
        return None
    
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']
    except Exception:
        emotion = "neutral"

    cap.release()

    return emotion