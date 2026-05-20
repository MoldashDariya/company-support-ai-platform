"""Central configuration for the support assistant platform."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "company_knowledge.md"
MEMORY_DB_PATH = ROOT_DIR / "data" / "memory.db"
# Legacy JSON paths (auto-migrated on first startup)
SESSION_STORE_PATH = ROOT_DIR / "data" / "sessions.json"
LEGACY_DIALOGS_PATH = ROOT_DIR / "data" / "dialogs.json"
MEMORY_SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "1200"))

# Telegram channel
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("BOT_TOKEN", ""))

# LLM provider (OpenAI-compatible API)
LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", os.getenv("CEREBRAS_API_KEY", "")),
)
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    os.getenv(
        "OPENAI_BASE_URL",
        os.getenv("CEREBRAS_BASE_URL", "https://api.openai.com/v1"),
    ),
)
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", os.getenv("CEREBRAS_MODEL", "gpt-4o-mini")))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("AI_TEMPERATURE", "0.2")))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Company branding (used in prompts and UI)
COMPANY_NAME = os.getenv("COMPANY_NAME", "Центр Красок #1")
COMPANY_PHONE = os.getenv("COMPANY_PHONE", "+77780615000")
COMPANY_SITE = os.getenv("COMPANY_SITE", "https://centr-krasok.kz/")

# Retrieval (RAG)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", os.getenv("TOP_K_CHUNKS", "4")))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.1"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | semantic | sparse
ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() in ("1", "true", "yes")
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.35"))
CITATION_INCLUDE_URLS = os.getenv("CITATION_INCLUDE_URLS", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Semantic index (sentence-transformers + ChromaDB)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHROMA_PERSIST_DIR = ROOT_DIR / "data" / "chroma"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "company_knowledge")
CORPUS_LANGUAGE = os.getenv("CORPUS_LANGUAGE", "ru")

# Website ingestion
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "30"))
CRAWL_REQUEST_DELAY_SEC = float(os.getenv("CRAWL_REQUEST_DELAY_SEC", "0.5"))
CRAWL_TIMEOUT_SEC = float(os.getenv("CRAWL_TIMEOUT_SEC", "20"))
CRAWL_USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT",
    "CompanySupportBot/1.0 (+https://github.com; knowledge-ingestion)",
)
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "120"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))


def ingestion_seed_urls() -> list[str]:
    """Resolve crawl seeds from env or fall back to the company site."""
    raw = os.getenv("INGESTION_SEED_URLS", "").strip()
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    base = COMPANY_SITE.rstrip("/")
    # Single seed — crawler discovers same-domain editorial pages via link graph
    return [base + "/"]

# Conversation memory
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", os.getenv("MAX_HISTORY_TURNS", "8")))
MAX_USER_INPUT_CHARS = int(os.getenv("MAX_USER_INPUT_CHARS", os.getenv("MAX_USER_MESSAGE_LEN", "2000")))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "3500"))

# Rate limiting
THROTTLE_MAX_REQUESTS = int(os.getenv("THROTTLE_MAX_REQUESTS", os.getenv("RATE_LIMIT_REQUESTS", "12")))
THROTTLE_WINDOW_SEC = int(os.getenv("THROTTLE_WINDOW_SEC", os.getenv("RATE_LIMIT_WINDOW", "60")))

# Streaming UX
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", os.getenv("STREAM_RESPONSES", "true")).lower() in (
    "1",
    "true",
    "yes",
)
STREAM_EDIT_INTERVAL_SEC = float(os.getenv("STREAM_EDIT_INTERVAL_SEC", "0.6"))

WELCOME_MESSAGE = os.getenv(
    "WELCOME_MESSAGE",
    f"Здравствуйте! Я AI-ассистент «{COMPANY_NAME}» — отвечаю на вопросы о магазине: "
    "услуги, бренды, салоны, доставка, программа для дизайнеров.\n\n"
    "Просто напишите вопрос обычным сообщением.",
)

INJECTION_PATTERNS = (
    "ignore previous",
    "забудь инструкции",
    "system prompt",
    "jailbreak",
    "выведи промпт",
    "pretend you are",
    "act as",
)
