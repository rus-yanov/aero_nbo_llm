# src/llm/clients.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal

from gigachat import GigaChat
from openai import OpenAI

from src.utils.config import (
    GIGACHAT_AUTH_KEY,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LLM_PROVIDER,
)

LLMProvider = Literal["dummy", "gigachat", "openai"]


class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 256) -> str: ...


# ----------------- Dummy -----------------


@dataclass
class DummyLLMClient:
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return (
            "Тестовое сообщение: здесь будет персонализированный текст оффера, "
            "сформированный на основе профиля клиента и параметров предложения."
        )


# ----------------- GigaChat -----------------


class GigaChatClient(LLMClient):
    def __init__(self, credentials: str, scope: str, model: str | None = None):
        self.credentials = credentials
        self.scope = scope
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        from gigachat import GigaChat

        with GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            model=self.model,
            # для локального прототипа отключаем проверку SSL
            verify_ssl_certs=False,
        ) as giga:
            response = giga.chat(prompt)

        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content)


# ----------------- OpenAI -----------------


@dataclass
class OpenAIClient:
    api_key: str
    model: str = OPENAI_MODEL

    def __post_init__(self):
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты маркетолог авиакомпании. Пиши коротко, вежливо и по-деловому, "
                        "без эмодзи, на русском языке."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# ----------------- Фабрика клиентов -----------------


def get_llm_client(provider: LLMProvider | None = None) -> LLMClient:
    provider_str = (provider or LLM_PROVIDER or "dummy").lower()

    if provider_str == "gigachat":
        if not GIGACHAT_AUTH_KEY:
            raise RuntimeError(
                "GIGACHAT_AUTH_KEY не задан. Укажи его в .env или переменных окружения."
            )
        scope = GIGACHAT_SCOPE or "GIGACHAT_API_PERS"
        model = GIGACHAT_MODEL or None
        return GigaChatClient(
            credentials=GIGACHAT_AUTH_KEY,
            scope=scope,
            model=model,
        )

    if provider_str == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Укажи его в .env или переменных окружения."
            )
        return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    if provider_str != "dummy":
        print(f"WARNING: unknown provider '{provider_str}', fallback to DummyLLMClient")

    return DummyLLMClient()