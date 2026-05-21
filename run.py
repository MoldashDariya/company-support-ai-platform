"""Entry point for the Company Support AI Assistant platform."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys

_REQUIRED_PACKAGES = (
    ("aiogram", "aiogram"),
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
    ("rank_bm25", "rank-bm25"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
)


def _check_dependencies() -> None:
    missing: list[str] = []
    for module, package in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(
            "Missing Python dependencies:\n"
            f"  {', '.join(missing)}\n\n"
            "Install dependencies:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt",
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
        "Environment loaded | retrieval=%s bm25=%s citations_urls=%s",
        settings.RETRIEVAL_MODE,
        settings.ENABLE_BM25,
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
