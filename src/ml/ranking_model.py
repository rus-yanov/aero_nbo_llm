# src/ml/ranking_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.utils.config import (
    ML_TRAINING_DATASET_PATH,
    RANKING_MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__) if 'get_logger' in globals() else None


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: List[str]
    cat_feature_indices: List[int]


def load_training_data() -> pd.DataFrame:
    """Читает подготовленный ML-датасет."""
    df = pd.read_csv(ML_TRAINING_DATASET_PATH)
    if logger:
        logger.info("Loaded ML training dataset: %s, shape=%s",
                    ML_TRAINING_DATASET_PATH, df.shape)
    return df


def _prepare_features(df: pd.DataFrame) -> DatasetSplit:
    """Разделяет датафрейм на признаки/таргет + train/test split."""
    target_col = "conversion"

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Категориальные фичи — все object-колонки
    cat_feature_indices = [
        i for i, col in enumerate(feature_cols) if X[col].dtype == "object"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    if logger:
        logger.info(
            "Train/test split: train=%d, test=%d, features=%d (cat=%d)",
            len(X_train),
            len(X_test),
            len(feature_cols),
            len(cat_feature_indices),
        )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=feature_cols,
        cat_feature_indices=cat_feature_indices,
    )


def train_ranking_model(df: pd.DataFrame) -> Tuple[CatBoostClassifier, DatasetSplit]:
    """Обучает CatBoost-модель ранжирования по вероятности клика."""
    split = _prepare_features(df)

    train_pool = Pool(
        split.X_train,
        label=split.y_train,
        cat_features=split.cat_feature_indices,
    )
    eval_pool = Pool(
        split.X_test,
        label=split.y_test,
        cat_features=split.cat_feature_indices,
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.1,
        iterations=300,
        random_seed=RANDOM_SEED,
        verbose=50,
    )

    if logger:
        logger.info("Start CatBoost training...")
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    if logger:
        logger.info("Training finished.")

    # сохранение модели
    RANKING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(RANKING_MODEL_PATH))
    if logger:
        logger.info("Model saved to %s", RANKING_MODEL_PATH)

    return model, split


def load_ranking_model() -> CatBoostClassifier:
    """Загружает сохранённую модель."""
    model = CatBoostClassifier()
    model.load_model(str(RANKING_MODEL_PATH))
    return model


def predict_click_proba(
    model: CatBoostClassifier,
    df_features: pd.DataFrame,
    feature_cols: List[str],
    cat_feature_indices: List[int],
) -> pd.Series:
    """Возвращает p(click=1) для набора признаков."""
    X = df_features[feature_cols]
    pool = Pool(X, cat_features=cat_feature_indices)
    proba = model.predict_proba(pool)[:, 1]
    return pd.Series(proba, index=df_features.index)