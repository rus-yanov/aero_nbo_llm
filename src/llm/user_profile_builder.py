# src/llm/user_profile_builder.py

import pandas as pd

def _fmt_money(x: float) -> str:
    return f"{x:.0f}" if pd.notna(x) else "неизвестно"

def _fmt_float(x: float) -> str:
    return f"{x:.2f}" if pd.notna(x) else "неизвестно"

def build_user_profile(row: pd.Series) -> str:
    """
    На вход: одна строка (df.loc[i]) из ML-датасета.
    На выход: короткий текстовый профиль клиента.
    """

    recency = row.get("recency_days")
    freq = row.get("frequency_90d")
    monetary = row.get("monetary_90d")
    fav_cat = row.get("favorite_category")
    price_seg = row.get("price_segment")
    discounts_used = row.get("discounts_used_90d")
    email_open = row.get("email_open_rate_30d")

    profile = f"""
Краткий профиль клиента:

• Давность последней покупки: {recency} дней.
• Частота покупок за 90 дней: {freq}.
• Сумма трат за 90 дней: {monetary}.
• Интересуется категорией: {fav_cat}.
• Сегмент цены: {price_seg}.
• Использовал скидки: {discounts_used} раз за 90 дней.
• Открываемость e-mail писем: {email_open}.

Общий вывод: сформировать аккуратное персонализированное сообщение с учётом интересов и сегмента.
"""
    return profile.strip()