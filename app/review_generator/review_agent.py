import os
import json
import random
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

CONTEXT_DESCRIPTIONS = {
    "comfort_seeking": "sad or emotionally low, seeking comfort and warmth",
    "stress_relief": "stressed or anxious, needing calm and low-friction experiences",
    "social_exploration": "happy and energetic, open to lively social experiences",
    "balanced": "emotionally neutral and open to general dining experiences"
}

STYLE_VARIATIONS = [
    "Write from the angle of someone who visited for the first time.",
    "Write from the angle of someone who is a regular customer.",
    "Write from the angle of someone who came with friends.",
    "Write from the angle of someone who came alone after a long day.",
    "Write from the angle of someone who nearly did not come but is glad they did.",
]

def generate_review(restaurant, context, emotion_scores=None):
    """
    Uses Gemini 2.0 Flash to generate a contextually-aware simulated
    Nigerian user review based on the user's detected emotional state.

    Args:
        restaurant: pandas Series or dict with restaurant details
        context: one of comfort_seeking | stress_relief | social_exploration | balanced
        emotion_scores: optional dict of raw emotion scores from DeepFace

    Returns:
        dict with 'reasoning' (str), 'review' (str) and 'rating' (float)
    """

    emotional_state = CONTEXT_DESCRIPTIONS.get(context, "neutral")
    style_variation = random.choice(STYLE_VARIATIONS)

    emotion_detail = ""
    if emotion_scores:
        top_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: float(x[1]),
            reverse=True
        )[:3]

        formatted_emotions = []
        for e, s in top_emotions:
            try:
                score = float(s)
            except:
                score = 0.0
            formatted_emotions.append(f"{e}: {score:.1f}%")

        emotion_detail = (
            "Top detected emotions: "
            + ", ".join(formatted_emotions)
            + "."
        )

    prompt = f"""You are simulating a real restaurant review written by a Nigerian customer.
Vary your writing style, sentence structure, and Pidgin usage each time — no two reviews should sound the same.

The customer was feeling {emotional_state} when they visited this restaurant.
{emotion_detail}
{style_variation}

Restaurant details:
- Name: {restaurant['name']}
- City: {restaurant['city']}
- Categories: {restaurant['categories']}
- Yelp Rating: {restaurant['stars']} stars
- Review Count: {restaurant.get('review_count', 'N/A')}

First explain WHY this restaurant emotionally matches the user. Then write a short authentic 2-3 sentence review this customer might leave. Also provide a predicted rating out of 5.
The review tone should reflect the customer's emotional state — someone seeking comfort will appreciate warmth and coziness more than someone in a social mood.

IMPORTANT: Write in authentic Nigerian English and naturally mix in Nigerian Pidgin. Use expressions like:
'the food dey sweet', 'e be like say', 'I no go lie', 'the vibes was on point', 'e don do am',
'this place try', 'guy the service na 10/10', 'abeg', 'e shock me', 'na so e be'.
Do NOT force Pidgin into every sentence — blend it naturally so it sounds like a real Nigerian reviewing on Yelp.

Respond ONLY with valid JSON in this exact format, no markdown, no backticks:
{{
  "reasoning": "one sentence explaining why this restaurant fits the user's emotional state",
  "review": "the 2-3 sentence Nigerian review",
  "rating": 4.5
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        if "review" not in result or "rating" not in result:
            raise ValueError("Missing keys in Gemini response")

        result["rating"] = round(min(5.0, max(1.0, float(result["rating"]))), 1)
        result.setdefault("reasoning", "")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[review_agent] Parse error: {e}")
        return _fallback_review(context)

    except Exception as e:
        print(f"[review_agent] Gemini call failed: {e}")
        return _fallback_review(context)


def _fallback_review(context):
    """Returns a randomised static Nigerian review if the Gemini call fails."""
    fallbacks = {
        "comfort_seeking": [
            ("I no go lie, this place dey give the right energy when you just need to relax. The food was warm and e hit different.", 4.5),
            ("Abeg this spot saved me that evening. The warmth of the food and the calm atmosphere na exactly wetin I needed.", 4.4),
        ],
        "stress_relief": [
            ("Guy if you need somewhere to decompress, this place na the answer. Peaceful vibes and the service was smooth.", 4.2),
            ("E be like say dem designed this place for people wey wan reset. I entered stressed, I left feeling like myself again.", 4.3),
        ],
        "social_exploration": [
            ("This place was on point! Brought my crew and we had the best time — the energy dey contagious.", 4.6),
            ("E shock me how lively this spot was. The food, the vibes, the whole experience na 10/10.", 4.7),
        ],
        "balanced": [
            ("Solid spot overall, I no go complain. Good food, decent service — would recommend to anyone.", 4.0),
            ("The experience was straight — nothing wey go make you vex, everything just dey flow well.", 4.1),
        ]
    }
    options = fallbacks.get(context, fallbacks["balanced"])
    review, rating = random.choice(options)
    return {"reasoning": "", "review": review, "rating": rating}