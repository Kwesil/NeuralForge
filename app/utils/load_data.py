import pandas as pd
import json

def load_restaurants(json_path):
    data = []

    with open(json_path, "r", encoding="utf-8") as file:

        for line in file:
            business = json.laods(line)

            # Keep only restaurants
            categories = business.get("categories")

            if categories and "Restaurants" in categories:
                data.append({
                    "name": business.get("name"),
                    "city": business.get("city"),
                    "stars": business.get("stars"),
                    "review_count": business.get("review_count"),
                    "categories": categories
                })

    df = pd.DataFrame(data)
    
    return df