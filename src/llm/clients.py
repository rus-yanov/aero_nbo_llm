from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 256) -> str:  # pragma: no cover - интерфейс
        ...


# ---------- Dummy-клиент ----------


@dataclass
class DummyLLMClient:
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        logger.warning("DummyLLMClient is used instead of real LLM provider")
        return (
            "Тестовое сообщение: здесь будет персонализированный текст оффера, "
            "сформированный на основе профиля клиента и параметров предложения."
        )


# ---------- GigaChat-клиент ----------


class GigaChatLLMClient:
    def __init__(self, credentials: Optional[str] = None):
        from gigachat import GigaChat  # импортируем только при использовании

        self._credentials = credentials or os.getenv("GIGACHAT_AUTH_KEY")
        if not self._credentials:
            raise ValueError(
                "GIGACHAT_AUTH_KEY is not set in environment, "
                "но выбран провайдер 'gigachat'"
            )

        scope = os.getenv("GIGACHAT_SCOPE")
        auth_url = os.getenv("GIGACHAT_API_URL")
        model = os.getenv("GIGACHAT_MODEL")

        # по умолчанию для локального прототипа отключаем проверку SSL-сертификата,
        # но даём возможность включить её через переменную окружения
        verify_ssl_env = os.getenv("GIGACHAT_VERIFY_SSL_CERTS")
        if verify_ssl_env is None:
            verify_ssl_certs = False  # дефолт: легче запустить прототип
        else:
            verify_ssl_certs = verify_ssl_env.lower() in ("1", "true", "yes", "on")

        kwargs = {
            "credentials": self._credentials,
            "verify_ssl_certs": verify_ssl_certs,
        }
        if scope:
            kwargs["scope"] = scope
        if auth_url:
            kwargs["auth_url"] = auth_url
        if model:
            kwargs["model"] = model

        # GigaChatSyncClient внутри сам управляет токеном и моделью
        self._client = GigaChat(**kwargs)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Для GigaChat параметр max_tokens не пробрасываем, так как
        у метода chat нет такого аргумента в текущей версии SDK.
        Ограничения по длине контролируются на стороне GigaChat.
        """
        try:
            resp = self._client.chat(prompt)
        except Exception as e:  # noqa: BLE001
            logger.error(f"GigaChat request failed: {e}")
            raise

        if not resp.choices:
            raise RuntimeError("GigaChat вернул пустой список choices")

        return resp.choices[0].message.content

# ---------- OpenAI-клиент ----------


class OpenAILLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI  # официальный SDK

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in environment, "
                "но выбран провайдер 'openai'"
            )

        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=self._api_key)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"OpenAI request failed: {e}")
            raise

        choice = resp.choices[0]
        return choice.message.content or ""


# ---------- Клиент с фолбэком ----------


class FallbackLLMClient:
    """
    Сначала пробуем primary, при любой ошибке уходим в fallback.
    """

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient,
        primary_name: str = "primary",
        fallback_name: str = "fallback",
    ):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            return self.primary.generate(prompt, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Primary LLM '{self.primary_name}' failed ({e}), "
                f"falling back to '{self.fallback_name}'"
            )
            return self.fallback.generate(prompt, max_tokens=max_tokens)


# ---------- Фабрика клиентов ----------


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    provider:
      - "dummy"
      - "gigachat"                      — только GigaChat
      - "openai"                        — только OpenAI
      - "gigachat_with_openai_fallback" — GigaChat с фолбэком на OpenAI

    если None — берём из LLM_PROVIDER или "gigachat_with_openai_fallback".
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gigachat_with_openai_fallback")

    provider = provider.lower()

    if provider == "dummy":
        return DummyLLMClient()

    if provider == "gigachat":
        return GigaChatLLMClient()

    if provider == "openai":
        return OpenAILLMClient()

    if provider in ("gigachat_with_openai_fallback", "gigachat_fallback"):
        primary = GigaChatLLMClient()
        fallback = OpenAILLMClient()
        return FallbackLLMClient(
            primary=primary,
            fallback=fallback,
            primary_name="gigachat",
            fallback_name="openai",
        )

    raise ValueError(f"Unknown LLM provider: {provider}")