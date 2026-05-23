from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-l6-v2"
)

def create_combined_features(df):
    df["combined_features"] = (
        df["categories"].fillna("") + " " + df["city"].fillna("")
    )

    return df

def create_embeddings(df):
    embeddings = embedding_model.encode(
        df["combined_features"].tolist(),
        show_progress_bar=True
    )

    return embeddings
