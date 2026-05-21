"""Entry point for the Company Support AI Assistant platform."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_BASE_PACKAGES = (
    ("aiogram", "aiogram"),
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
    ("rank_bm25", "rank-bm25"),
)

_SEMANTIC_PACKAGES = (
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
)


def _load_dotenv() -> None:
    """Load project .env without overriding variables already set (e.g. on Render)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env", override=False)


def _semantic_search_enabled() -> bool:
    """Whether chromadb / sentence-transformers are required (matches runtime.settings)."""
    _load_dotenv()
    return os.getenv("ENABLE_SEMANTIC_SEARCH", "false").lower() in ("1", "true", "yes")


def _check_dependencies() -> None:
    required = list(_BASE_PACKAGES)
    if _semantic_search_enabled():
        required.extend(_SEMANTIC_PACKAGES)

    missing: list[str] = []
    for module, package in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        install_lines = ["  pip install -r requirements.txt"]
        if _semantic_search_enabled():
            install_lines.append(
                "  pip install chromadb sentence-transformers  # ENABLE_SEMANTIC_SEARCH=true"
            )
        print(
            "Missing Python dependencies:\n"
            f"  {', '.join(missing)}\n\n"
            "Install dependencies:\n"
            + "\n".join(install_lines),
            file=sys.stderr,
        )
        raise SystemExit(1)


def _load_runtime():
    try:
        from runtime.bootstrap import ApplicationContext, validate_environment
        from runtime.logging_config import configure_logging
        from runtime import settings

        return ApplicationContext, validate_environment, configure_logging, settings
    except ImportError as exc:
        print(
            f"Failed to import application modules: {exc}\n\n"
            "Install dependencies:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


logger = logging.getLogger(__name__)


async def main() -> None:
    ApplicationContext, validate_environment, configure_logging, settings = _load_runtime()

    configure_logging()
    validate_environment()

    logger.info(
        "Environment loaded | retrieval=%s semantic=%s bm25=%s citations_urls=%s",
        settings.effective_retrieval_mode(),
        settings.ENABLE_SEMANTIC_SEARCH,
        settings.sparse_retrieval_enabled(),
        settings.CITATION_INCLUDE_URLS,
    )

    ctx = ApplicationContext()
    ctx.log_provider_ready()

    bot = await ctx.initialize_telegram()
    dispatcher = ctx.build_dispatcher()

    logger.info(
        "Support platform ready | corpus_fragments=%d model=%s streaming=%s",
        ctx.retriever.fragment_count,
        settings.LLM_MODEL,
        settings.ENABLE_STREAMING,
    )

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    _check_dependencies()
    asyncio.run(main())
