from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import pickle, numpy as np

app = FastAPI(title="KMeans Product Segmentation API", version="1.0.0")

# paths to artifacts (make sure scaler.pkl and kmeans.pkl are in project root)
# in main.py
from pathlib import Path

ART_DIR = Path(__file__).resolve().parent  # same folder as main.py
SCALER_PATH = ART_DIR / "scaler.pkl"
MODEL_PATH  = ART_DIR / "kmeans.pkl"


_scaler = None
_model = None
_features = ["Price","Sales","Revenue","Ratings","Review Count","FBA Fees","Weight"]

def load_artifacts():
    global _scaler, _model
    if _scaler is None:
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
    if _model is None:
        with open(MODEL_PATH, "rb") as f: _model = pickle.load(f)

class Product(BaseModel):
    price: float
    sales: float
    revenue: float
    ratings: float = Field(..., ge=0, le=5)
    review_count: float
    fba_fees: float
    weight: float

def to_vector(p: Product) -> np.ndarray:
    return np.array([
        p.price, p.sales, p.revenue, p.ratings,
        p.review_count, p.fba_fees, p.weight
    ], dtype=float).reshape(1, -1)

@app.get("/ping")
def ping() -> Dict[str, Any]:
    load_artifacts()
    return {"status": "ok", "features": _features}

@app.post("/predict")
def predict(product: Product) -> Dict[str, Any]:
    load_artifacts()
    X = to_vector(product)
    Xs = _scaler.transform(X)
    cluster = int(_model.predict(Xs)[0])
    return {"cluster": cluster}

class BatchRequest(BaseModel):
    items: List[Product]

@app.post("/predict-batch")
def predict_batch(req: BatchRequest) -> Dict[str, Any]:
    load_artifacts()
    X = np.vstack([to_vector(p) for p in req.items])
    Xs = _scaler.transform(X)
    clusters = _model.predict(Xs).astype(int).tolist()
    return {"clusters": clusters, "count": len(clusters)}
