import random

def filter_restaurants(df, context):
    keywords = []
    if context == "comfort_seeking":
        keywords = [
            "Cafe",
            "Coffee",
            "Soup",
            "Ramen",
            "Tea"
        ]

    elif context == "stress_relief":
        keywords = [
            "Garden",
            "Healthy",
            "Vegatarian",
            "Seafood"
        ]
    
    elif context == "social_exploration":
        keywords = [
            "American",
            "Italian",
            "Burgers"
        ]

    else: 
        # Fallback safety
        keywords = [
            "Restaurant"
        ]

    filtered = df[
        df["categories"].str.contains(
            "|".join(keywords),
            case=False,
            na=False
        )
    ]

    if len(filtered) == 0:
        filtered = df

    recommendations = filtered.sample(
        min(5, len(filtered))
    )

    return recommendations.to_dict(orient="records")