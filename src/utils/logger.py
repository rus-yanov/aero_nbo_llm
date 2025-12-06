# src/utils/logger.py

from loguru import logger as loguru_logger
import sys
from pathlib import Path

from src.utils.config import (
    PROJECT_ROOT,
    ENABLE_LOGGING,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
)


# Папка для логов
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


# Чтобы не создавать несколько одинаковых sink'ов
_logger_initialized = False


def _initialize_logger():
    """Инициализация loguru (один раз за весь проект)."""
    global _logger_initialized

    if _logger_initialized:
        return

    # Убираем стандартный sink loguru
    loguru_logger.remove()

    if ENABLE_LOGGING:
        # Консоль
        loguru_logger.add(
            sys.stdout,
            level=LOG_LEVEL_CONSOLE,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
                   "<level>{message}</level>",
        )

        # Файл
        loguru_logger.add(
            LOG_FILE,
            level=LOG_LEVEL_FILE,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
    else:
        # Полное отключение логов
        loguru_logger.disable(None)

    _logger_initialized = True


def get_logger(name: str):
    """
    Возвращает экземпляр логгера для модуля.
    Использование:
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    _initialize_logger()
    return loguru_logger.bind(module=name)