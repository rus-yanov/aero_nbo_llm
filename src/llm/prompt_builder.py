import json
from typing import Mapping, Any
import numpy as np

from src.utils.config import DATA_PROMPTS, DEFAULT_CHANNEL


def load_template(channel: str | None = None) -> str:
    """
    Загружает текстовый шаблон промпта по каналу.

    channel: "push" | "email" | "sms"
    """
    if channel is None:
        channel = DEFAULT_CHANNEL

    template_path = DATA_PROMPTS / f"{channel}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def _to_native(value: Any) -> Any:
    """Переводим NumPy / pandas типы в обычные Python-типы."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_offer_json(offer: Mapping[str, Any]) -> str:
    """
    Готовим компактное JSON-представление оффера,
    которое подставляется в шаблон как {offer_data}.
    """
    native_offer = {str(k): _to_native(v) for k, v in offer.items()}
    return json.dumps(native_offer, ensure_ascii=False, indent=2)


def build_prompt(
    user_profile: str,
    offer: Mapping[str, Any],
    channel: str | None = None,
) -> str:
    """
    Собирает финальный промпт для LLM:
    - читает шаблон по каналу,
    - подставляет профиль клиента и JSON с параметрами оффера.
    """

    template = load_template(channel)
    offer_data = build_offer_json(offer)

    prompt = template.format(
        user_profile=user_profile.strip(),
        offer_data=offer_data,
    )
    return prompt