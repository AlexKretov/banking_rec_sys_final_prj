"""Подготовка T-ECD и обучение двухуровневой рекомендательной системы.

Модуль намеренно работает с локальной выборкой Parquet: полный T-ECD (2.81 ТБ)
не следует загружать в память pandas целиком.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DOMAINS = ("marketplace", "retail", "offers")
POSITIVE_ACTIONS = {"click", "add_to_cart", "add-to-cart", "cart", "order", "purchase", "transition"}


def find_dataset_root(path: str | Path) -> Path:
    """Находит каталог, содержащий users.pq, при любом local_dir downloader-а."""
    path = Path(path)
    candidates = [path, path / "dataset" / "small", path / "dataset" / "full"]
    for candidate in candidates:
        if (candidate / "users.pq").exists():
            return candidate
    found = list(path.glob("**/users.pq"))
    if not found:
        raise FileNotFoundError(f"users.pq не найден в {path}")
    return found[0].parent


def _event_files(root: Path, domains: Iterable[str]) -> list[tuple[str, Path]]:
    result = []
    for domain in domains:
        for file in sorted((root / domain / "events").glob("*.pq")):
            result.append((domain, file))
    if not result:
        raise FileNotFoundError("Не найдены файлы <domain>/events/*.pq")
    # Читаем домены вперемешку по дням: иначе max_rows мог бы целиком
    # израсходоваться первым (обычно marketplace) доменом.
    return sorted(result, key=lambda pair: (pair[1].stem, pair[0]))


def load_events(data_dir: str | Path, domains: Iterable[str] = DOMAINS,
                max_rows: int | None = 2_000_000) -> pd.DataFrame:
    """Читает дневные партиции T-ECD и приводит их к единой схеме.

    День берётся из имени партиции, поэтому разбиение остаётся временным даже
    если в опубликованной версии файла отсутствует отдельный timestamp.
    """
    root = find_dataset_root(data_dir)
    frames, total = [], 0
    for domain, file in _event_files(root, domains):
        frame = pd.read_parquet(file)
        required = {"user_id", "item_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{file}: отсутствуют обязательные поля {sorted(missing)}")
        frame = frame.copy()
        frame["domain"] = domain
        frame["day"] = int(file.stem)
        keep = [c for c in ["user_id", "item_id", "brand_id", "action_type", "subdomain", "price", "count", "os", "domain", "day"] if c in frame]
        frame = frame[keep]
        if max_rows is not None:
            frame = frame.iloc[: max(0, max_rows - total)]
        frames.append(frame)
        total += len(frame)
        if max_rows is not None and total >= max_rows:
            break
    events = pd.concat(frames, ignore_index=True)
    events = events.dropna(subset=["user_id", "item_id"])
    return events


def temporal_split(events: pd.DataFrame, test_days: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Последние test_days партиций — тест; случайное разбиение запрещено."""
    days = np.sort(events["day"].unique())
    if len(days) <= test_days:
        raise ValueError(f"Нужно больше {test_days} уникальных дневных партиций")
    cutoff = days[-test_days]
    return events[events.day < cutoff].copy(), events[events.day >= cutoff].copy()


def positive_events(events: pd.DataFrame) -> pd.DataFrame:
    if "action_type" not in events:
        return events
    action = events.action_type.astype(str).str.lower()
    mask = action.isin(POSITIVE_ACTIONS)
    # Если кодировка action_type числовая, события всё равно являются implicit feedback.
    return events[mask] if mask.any() else events


def train(data_dir: str | Path, output_dir: str | Path = "fastaip/ml_service/artifacts",
          max_rows: int | None = 2_000_000, factors: int = 64) -> dict:
    events = load_events(data_dir, max_rows=max_rows)
    train_events, test_events = temporal_split(events)
    interactions = positive_events(train_events)

    users = pd.Index(interactions.user_id.unique())
    items = pd.Index(interactions.item_id.unique())
    user_codes = users.get_indexer(interactions.user_id)
    item_codes = items.get_indexer(interactions.item_id)
    weights = interactions.get("count", pd.Series(1.0, index=interactions.index)).fillna(1).clip(lower=1).astype(float)
    matrix = csr_matrix((weights, (user_codes, item_codes)), shape=(len(users), len(items)))

    als = AlternatingLeastSquares(factors=factors, iterations=20, regularization=.05, random_state=42)
    als.fit(matrix)

    # Сохраняем исходную двухуровневую идею проекта: ALS score + контекст -> RF.
    rows = test_events[test_events.user_id.isin(users) & test_events.item_id.isin(items)].copy()
    if rows.empty:
        raise ValueError("В тестовом периоде нет известных user/item; загрузите больше дней")
    rows["user_code"] = users.get_indexer(rows.user_id)
    rows["item_code"] = items.get_indexer(rows.item_id)
    rows["als_score"] = (als.user_factors[rows.user_code.to_numpy()] * als.item_factors[rows.item_code.to_numpy()]).sum(1)
    rows["target"] = positive_events(rows).index.to_series().reindex(rows.index).notna().astype(int)
    features = [c for c in ["als_score", "domain", "subdomain", "os", "price", "count"] if c in rows]
    numeric = [c for c in ["als_score", "price", "count"] if c in features]
    categorical = [c for c in features if c not in numeric]
    if rows.target.nunique() > 1:
        ranker = Pipeline([
            ("prepare", ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])),
            ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]).fit(rows[features], rows.target)
        report = classification_report(rows.target, ranker.predict(rows[features]), output_dict=True, zero_division=0)
    else:
        ranker, report = None, {"note": "В test-периоде только один класс; RF не обучен"}

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"als": als, "ranker": ranker, "users": users.to_numpy(), "items": items.to_numpy(), "features": features}, output / "model.joblib")
    metadata = {"dataset": "t-tech/T-ECD", "domains": list(DOMAINS), "train_until_day": int(train_events.day.max()), "test_from_day": int(test_events.day.min()), "train_interactions": len(interactions), "users": len(users), "items": len(items), "metrics": report}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--output-dir", default="fastaip/ml_service/artifacts")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.output_dir, args.max_rows), ensure_ascii=False, indent=2))
