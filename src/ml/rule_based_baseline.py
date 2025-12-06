# src/ml/rule_based_baseline.py
from __future__ import annotations

import pandas as pd


def baseline_predict_proba(df: pd.DataFrame) -> pd.Series:
    """
    Очень простой rule-based baseline:
    - берем средний CTR по комбинации (price_segment, offer_type)
    - если комбо нет в статистике — используем глобальный CTR.
    На вход подаём те же строки, что и в ML-модель (с колонкой conversion).
    """

    if "conversion" not in df.columns:
        raise ValueError("DataFrame must contain 'conversion' column for baseline")

    # считаем CTR по группам
    group_ctr = (
        df.groupby(["price_segment", "offer_type"])["conversion"]
        .mean()
        .rename("group_ctr")
    )

    global_ctr = df["conversion"].mean()

    # мёрджим обратно
    df_tmp = df.merge(
        group_ctr,
        left_on=["price_segment", "offer_type"],
        right_index=True,
        how="left",
    )

    proba = df_tmp["group_ctr"].fillna(global_ctr)
    return proba