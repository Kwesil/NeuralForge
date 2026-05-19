RECOMMENDATIONS = {
    "comfort_seeking": [
        "Warm Bowl Ramen",
        "Cozy Corner Cafe",
        "Comfort Kitchen"
    ],

    "stress_relief": [
        "Quiet Garden Restauurant",
        "Nature Grill",
        "Zen Bistro"
    ],

    "social_expoloration": [
        "Fushion Lounge",
        "Spice Route",
        "Street Feast Hub"
    ],

    "balanced": [
        "City Grill",
        "Urban Table",
        "Harvest Bistro"
    ]
}

def get_recommendations(context):
    return RECOMMENDATIONS.get(
        context, 
        RECOMMENDATIONS["balanced"]
    )