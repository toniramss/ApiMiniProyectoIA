# cosineSimilarity.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityRecommender:
    def __init__(self, csv_url: str):
        # Basado en tu código: leer dataset + features + escalado
        self.df = pd.read_csv(csv_url)

        self.features = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
            "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol"
        ]

        X = self.df[self.features].copy()

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

    def recommend(self, wine_id: int, top_n: int = 3):
        # Obtener índice real a partir del ID (igual que tú)
        idx = self.df.index[self.df["wine_id"] == wine_id][0]

        # Cosine similarity (igual que tú)
        sims = cosine_similarity(
            self.X_scaled[idx].reshape(1, -1),
            self.X_scaled
        ).flatten()

        # Añadimos score y sacamos top recomendaciones (igual que tú)
        df_rec = self.df.copy()
        df_rec["cosine_sim"] = sims

        df_rec["cosine_sim"] = (sims * 100).round(2)

        recs = (
            df_rec[df_rec["wine_id"] != wine_id]
            .sort_values("cosine_sim", ascending=False)
            .head(top_n)
        )

        return recs
