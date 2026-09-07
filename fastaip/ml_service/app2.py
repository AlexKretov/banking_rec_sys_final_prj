"""FastAPI-сервис рекомендаций товаров T-ECD."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

ARTIFACT = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "artifacts" / "model.joblib"))
app = FastAPI(title="T-ECD recommender", version="2.0")
Instrumentator().instrument(app).expose(app)
REQUESTS = Counter("recommendation_requests_total", "Recommendation requests", ["status"])
LATENCY = Histogram("recommendation_latency_seconds", "Recommendation request latency")
CPU = Gauge("process_cpu_percent", "CPU usage percent")
MEMORY = Gauge("process_memory_percent", "Memory usage percent")
_bundle: dict[str, Any] | None = None


class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=5, ge=1, le=100)
    domain: str | None = None
    subdomain: str | None = None
    os: str | None = None


class Recommendation(BaseModel):
    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise HTTPException(503, f"Модель не найдена: {ARTIFACT}. Сначала запустите tecd_pipeline.py")
        _bundle = joblib.load(ARTIFACT)
    return _bundle


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok" if ARTIFACT.exists() else "model_not_trained", "model_path": str(ARTIFACT)}


@app.post("/predict", response_model=RecommendationResponse)
def predict(request: RecommendationRequest) -> RecommendationResponse:
    started = time.perf_counter()
    try:
        bundle = get_bundle()
        users = pd.Index(bundle["users"])
        user_code = users.get_indexer([request.user_id])[0]
        if user_code < 0:
            raise HTTPException(404, "user_id отсутствует в обучающем периоде T-ECD")
        als = bundle["als"]
        # implicit принимает строку user-items; обучающая матрица для фильтрации в
        # артефакт не сохраняется, поэтому filter_already_liked_items=False.
        ids, scores = als.recommend(user_code, None, N=request.top_k, filter_already_liked_items=False)
        item_ids = bundle["items"][ids]

        ranker = bundle.get("ranker")
        features = bundle.get("features", [])
        if ranker is not None:
            frame = pd.DataFrame({"als_score": scores})
            context = {"domain": request.domain, "subdomain": request.subdomain, "os": request.os}
            for column in features:
                if column not in frame:
                    frame[column] = context.get(column, 0 if column in {"price", "count"} else "unknown")
            scores = ranker.predict_proba(frame[features])[:, 1]
            order = np.argsort(-scores)
            item_ids, scores = item_ids[order], scores[order]

        result = [Recommendation(item_id=int(item), score=float(score)) for item, score in zip(item_ids, scores)]
        REQUESTS.labels("success").inc()
        return RecommendationResponse(user_id=request.user_id, recommendations=result)
    except HTTPException:
        REQUESTS.labels("error").inc()
        raise
    finally:
        LATENCY.observe(time.perf_counter() - started)
        CPU.set(psutil.cpu_percent())
        MEMORY.set(psutil.virtual_memory().percent)
