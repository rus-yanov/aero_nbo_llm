from typing import Mapping, Any

from src.llm.clients import get_llm_client, LLMProvider
from src.llm.prompt_builder import build_prompt
from src.utils.config import DEFAULT_CHANNEL


def generate_message(
    user_profile: str,
    offer: Mapping[str, Any],
    channel: str = DEFAULT_CHANNEL,
    provider: LLMProvider | str | None = None,
    max_tokens: int = 256,
) -> str:
    """
    Высокоуровневая обёртка:
    - собирает промпт,
    - вызывает выбранного LLM-провайдера,
    - возвращает готовый текст сообщения.

    Параметры:
        user_profile: текстовый профиль клиента (из User Profile Builder).
        offer: словарь с полями оффера (title, product_name, short_description, conditions, …).
        channel: "push" | "email" | "sms".
        provider: 'dummy' | 'yandex_gpt' | 'gigachat' (по умолчанию берётся из LLM_PROVIDER или dummy).
    """

    prompt = build_prompt(user_profile=user_profile, offer=offer, channel=channel)
    client = get_llm_client(provider=provider)
    message = client.generate(prompt=prompt, max_tokens=max_tokens)
    return message