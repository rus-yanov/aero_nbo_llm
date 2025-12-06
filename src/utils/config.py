import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# --- Data directories ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_PROMPTS = DATA_DIR / "prompts"

# --- Models, reports, logs ---
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure folders exist
for p in [DATA_PROCESSED, DATA_PROMPTS, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Raw datasets ---
COMMON_DATASET_PATH = DATA_RAW / "common_dataset.csv"
OFFERS_DATASET_PATH = DATA_RAW / "offers.csv"

# --- Processed datasets ---
ML_TRAINING_DATASET_PATH = DATA_PROCESSED / "ml_training_dataset.csv"

# --- Model paths ---
RANKING_MODEL_PATH = MODELS_DIR / "ranking_model.pkl"

# --- General settings ---
RANDOM_SEED = 42
TEST_SIZE = 0.2

# --- LLM settings ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dummy").strip().lower()

GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_API_URL = os.getenv(
    "GIGACHAT_API_URL",
    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_CHANNEL = "push"
DEFAULT_TOP_N = 3

# --- Logging settings ---
ENABLE_LOGGING = True
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"