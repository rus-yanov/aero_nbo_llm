# src/service/nbo_pipeline.py
from __future__ import annotations

from typing import Any, Mapping, Sequence, Dict, List, Optional

import pandas as pd
from catboost import Pool

from src.utils.config import ML_TRAINING_DATASET_PATH, DEFAULT_CHANNEL, DEFAULT_TOP_N
from src.ml.ranking_model import load_ranking_model
from src.utils.config import ML_TRAINING_DATASET_PATH
from src.llm.user_profile_builder import build_user_profile
from src.llm.message_generator import generate_message

def _load_feature_schema():
    """
    Восстанавливаем структуру признаков из ml_training_dataset.csv.
    target = conversion
    категория = объектные столбцы
    """
    df = pd.read_csv(ML_TRAINING_DATASET_PATH, nrows=100)
    feature_cols = [c for c in df.columns if c != "conversion"]
    cat_features = [c for c in feature_cols if df[c].dtype == "object"]
    cat_indices = [feature_cols.index(c) for c in cat_features]
    return feature_cols, cat_features, cat_indices

def _score_offers(df: pd.DataFrame):
    model = load_ranking_model()

    # узнаём колонки, на которых обучалась модель
    feature_cols, cat_features, cat_indices = _load_feature_schema()

    df = df.copy()
    X = df[feature_cols]

    pool = Pool(X, cat_features=cat_indices)
    df["p_click"] = model.predict_proba(pool)[:, 1]
    df = df.sort_values("p_click", ascending=False)
    return df


def _row_to_offer_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Приводим строку к словарю оффера, с которым удобно работать LLM.
    В реальном проекте сюда можно добавить больше полей.
    """
    return {
        "offer_id": int(row["offer_id"]),
        "offer_type": row.get("offer_type"),
        "offer_category": row.get("offer_category"),
        "cost": float(row.get("cost", 0.0)),
        "offer_AOV": float(row.get("offer_AOV", 0.0)),
        "title": row.get("title") or f"Offer {int(row['offer_id'])}",
        "product_name": row.get("product_name", ""),
        "short_description": row.get("short_description", ""),
        "conditions": row.get("conditions", ""),
    }


def _build_response(
    client_id: int,
    scored: pd.DataFrame,
    channel: str,
    provider: Optional[str],
    top_n: int,
) -> Dict[str, Any]:
    """
    Общая сборка JSON-ответа для обоих режимов (internal / online).
    """
    top = scored.head(top_n)
    if top.empty:
        return {
            "client_id": int(client_id),
            "channel": channel,
            "user_profile": "",
            "best_offer": None,
            "alternative_offers": [],
        }

    # Берём первую строку для построения профиля клиента
    profile_row = top.iloc[0]
    user_profile = build_user_profile(profile_row)

    offers_payload: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        offer = _row_to_offer_dict(row)
        message = generate_message(
            user_profile=user_profile,
            offer=offer,
            channel=channel,
            provider=provider or "dummy",
        )

        offers_payload.append(
            {
                "offer_id": offer["offer_id"],
                "p_click": float(row["p_click"]),
                "title": offer["title"],
                "short_description": offer["short_description"],
                "conditions": offer["conditions"],
                "personalized_message": message,
            }
        )

    return {
        "client_id": int(client_id),
        "channel": channel,
        "user_profile": user_profile,
        "best_offer": offers_payload[0],
        "alternative_offers": offers_payload[1:],
    }


# ---------- Режим 1. Внутренние данные (client_id) ----------

def get_nbo_response(
    client_id: int,
    top_n: int = DEFAULT_TOP_N,
    channel: str = DEFAULT_CHANNEL,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Основной сценарий:
    - читаем ml_training_dataset.csv,
    - фильтруем строки по client_id,
    - считаем p_click,
    - строим профиль и тексты сообщений,
    - возвращаем JSON-ответ для сервиса рассылок.
    """
    df_ml = pd.read_csv(ML_TRAINING_DATASET_PATH)
    client_rows = df_ml[df_ml["client_id"] == client_id]

    if client_rows.empty:
        return {
            "client_id": int(client_id),
            "channel": channel,
            "user_profile": "",
            "best_offer": None,
            "alternative_offers": [],
        }

    scored = _score_offers(client_rows)
    return _build_response(
        client_id=client_id,
        scored=scored,
        channel=channel,
        provider=provider,
        top_n=top_n,
    )


# ---------- Режим 2. Онлайн-вход (фичи приходят в запросе) ----------

def get_nbo_response_from_rows(
    rows: Sequence[Mapping[str, Any]],
    client_id: Optional[int] = None,
    top_n: int = DEFAULT_TOP_N,
    channel: str = DEFAULT_CHANNEL,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Онлайн-режим:
    - rows — список словарей, каждый соответствует паре client–offer
      с теми же полями, что и в обучающем датасете (кроме conversion).
    - client_id можно передать отдельно, либо берем из первого ряда.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    df = pd.DataFrame(list(rows))

    if client_id is None:
        if "client_id" not in df.columns:
            raise ValueError("client_id is not provided and not found in rows")
        client_id = int(df["client_id"].iloc[0])

    scored = _score_offers(df)
    return _build_response(
        client_id=client_id,
        scored=scored,
        channel=channel,
        provider=provider,
        top_n=top_n,
    )