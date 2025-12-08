from typing import Mapping, Any

from src.llm.clients import get_llm_client
from src.llm.prompt_builder import build_prompt
from src.utils.config import DEFAULT_CHANNEL


def generate_message(
    user_profile: str,
    offer: Mapping[str, Any],
    channel: str = DEFAULT_CHANNEL,
    provider: str | None = None,
    max_tokens: int = 256,
) -> str:
    """
    Генерирует персонализированное сообщение:
    - собирает промпт,
    - вызывает выбранного LLM-провайдера,
    - возвращает текст.

    provider:
        None — использовать значение из окружения LLM_PROVIDER
        "dummy"
        "gigachat"
        "openai"
        "gigachat_with_openai_fallback"
    """
    prompt = build_prompt(user_profile=user_profile, offer=offer, channel=channel)
    client = get_llm_client(provider=provider)
    return client.generate(prompt=prompt, max_tokens=max_tokens)