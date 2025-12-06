# src/evaluation/comparison.py
from __future__ import annotations

from typing import Dict

import pandas as pd
from catboost import Pool

from src.utils.config import ML_TRAINING_DATASET_PATH, RANKING_MODEL_PATH
from src.ml.ranking_model import load_ranking_model
from src.ml.rule_based_baseline import baseline_predict_proba
from src.evaluation.metrics import classification_metrics


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(ML_TRAINING_DATASET_PATH)
    return df


def _get_feature_schema(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "conversion"]
    cat_features = [c for c in feature_cols if df[c].dtype == "object"]
    cat_indices = [feature_cols.index(c) for c in cat_features]
    return feature_cols, cat_indices


def evaluate_ml_model(df: pd.DataFrame) -> Dict[str, float]:
    feature_cols, cat_indices = _get_feature_schema(df)
    model = load_ranking_model()

    X = df[feature_cols]
    y = df["conversion"]

    pool = Pool(X, label=y, cat_features=cat_indices)
    y_proba = model.predict_proba(pool)[:, 1]

    return classification_metrics(y, y_proba, threshold=0.5)


def evaluate_rule_based(df: pd.DataFrame) -> Dict[str, float]:
    y = df["conversion"]
    y_proba = baseline_predict_proba(df)
    return classification_metrics(y, y_proba, threshold=0.5)


def compare_models() -> pd.DataFrame:
    """
    Возвращает таблицу с метриками:
    index = ['ml_model', 'rule_based']
    columns = ['roc_auc', 'log_loss', 'accuracy@0.5', 'average_precision']
    """
    df = _load_data()

    ml_metrics = evaluate_ml_model(df)
    rb_metrics = evaluate_rule_based(df)

    res = pd.DataFrame(
        {
            "ml_model": ml_metrics,
            "rule_based": rb_metrics,
        }
    ).T

    res.index.name = "model"
    return res