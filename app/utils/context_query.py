def context_to_query(context):
    queries = {
        "comfort_seeking": "quiet comforting ramen cafe warm atmosphere",
        "stress_relief": "healthy peaceful relaxing restaurant calm ambience",
        "social_exploration": "trendy energetic fun social dining esxperience",
        "balanced": "popular highly rated restaurant"
    }

    return queries.get(
        context,
        "popular restaurant"
    )