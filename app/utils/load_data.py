import pandas as pd
import json
import gdown
import os

def load_restaurants(json_path):
    if not os.path.exists(json_path):
        os.makedirs("data", exist_ok=True)
        gdown.download(
            id="1NxI4SqK1qLXXh9_PXjLI7uzPm-O2LaOI",
            output=json_path,
            quiet=False
        )

    data = []
    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            business = json.loads(line)
            categories = business.get("categories")
            if categories and "Restaurants" in categories:
                data.append({
                    "name": business.get("name"),
                    "city": business.get("city"),
                    "stars": business.get("stars"),
                    "review_count": business.get("review_count"),
                    "categories": categories
                })

    return pd.DataFrame(data)