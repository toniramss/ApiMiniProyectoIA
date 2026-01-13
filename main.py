import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from knn_model import run_knn, knn_curve
from cosine_similarity import CosineSimilarityRecommender

app = FastAPI(title="Wine Quality ML API")

# CORS para Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego limita en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del dataset
df = pd.read_csv("data/wine_quality_with_id.csv")

model = CosineSimilarityRecommender("data/wine_quality_with_id.csv")


@app.get("/knn")
def knn_endpoint(
    k: int = Query(..., ge=1, le=40, description="Número de vecinos")
):
    return run_knn(k)


@app.get("/knn/curve")
def knn_curve_endpoint(
    min_k: int = Query(1, ge=1, le=40),
    max_k: int = Query(25, ge=1, le=40)
):
    # Seguridad: si vienen al revés
    if min_k > max_k:
        min_k, max_k = max_k, min_k

    return {
        "min_k": min_k,
        "max_k": max_k,
        "points": knn_curve(min_k=min_k, max_k=max_k)
    }


@app.get("/wines")
def get_wines(limit: int = 20):
    """
    Devuelve N vinos aleatorios (mezclando tintos y blancos)
    """
    return (
        df.sample(frac=1)          # baraja todo el dataset
          .head(limit)             # coge solo N
          .to_dict(orient="records")
    )


@app.get("/wines/{wine_id}/similar")
def get_similar_wines(
    wine_id: int,
    top_n: int = Query(3, ge=1, le=50)
):
    
    try:
        recs = model.recommend(wine_id=wine_id, top_n=top_n)
    except IndexError:
        # Esto pasa si el wine_id no existe en el df y tu línea [0] revienta
        raise HTTPException(status_code=404, detail=f"wine_id {wine_id} no existe")

    # Devuelve solo lo que ya generas tú en recs
    return recs.to_dict(orient="records")