from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from knn_model import run_knn, knn_curve

app = FastAPI(title="Wine Quality ML API")

# CORS para Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego limita en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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