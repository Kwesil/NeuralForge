from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from recommender.embedding_model import embedding_model

def recommend_restaurants(user_query, df, embeddings, top_k=5):
    query_embedding = embedding_model.encode([user_query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    recommendations = df.iloc[top_indices]

    return recommendations