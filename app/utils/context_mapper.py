def map_emotion_to_context(emotion_scores):
    """
    Converts raw emotion scores into behavioral contexts.
    """

    happy = emotion_scores.get("happy", 0)
    neutral = emotion_scores.get("neutral", 0)
    sad = emotion_scores.get("sad", 0)
    angry = emotion_scores.get("angry", 0)
    fear = emotion_scores.get("fear", 0)
    surprise = emotion_scores.get("surprise", 0)

    # Comfort-seeking behavior
    if sad > 35 or (neutral > 40 and sad > 20):
        return {
            "context": "comfort_seeking",
            "description": "User may prefer comforting and relaxing experience."
        }
    
    # Stress-relief behavior
    elif angry > 25 or fear > 25:
        return {
            "context": "stress_relief",
            "description": "User may prefer calm environments and low-friction experiences."
        }
    
    # Social-exploration behaviour
    elif happy > 45 or surprise > 30:
        return {
            "context": "social_exploration",
            "description": "User may be open to energetic or adventurpus experiences."
        }
    
    # Default balanced state
    else:
        return {
            "context": "balanced",
            "description": "User appears emotionally balanced and open to general recommendations"
        }