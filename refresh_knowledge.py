#!/usr/bin/env python3
"""CLI: crawl company website and rebuild the knowledge indexes."""

from __future__ import annotations

import argparse
import logging
import sys

from ingestion.pipeline import KnowledgeIngestionPipeline
from runtime.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crawl company website and rebuild ChromaDB + BM25 knowledge indexes.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        help="Seed URL to crawl (can be repeated). Defaults to INGESTION_SEED_URLS / COMPANY_SITE.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to crawl (overrides CRAWL_MAX_PAGES).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    pipeline = KnowledgeIngestionPipeline(seed_urls=args.seeds)
    if args.max_pages:
        pipeline._crawler._max_pages = args.max_pages  # noqa: SLF001 — CLI override

    result = pipeline.run()

    print("\n=== Knowledge ingestion report ===")
    print(f"Pages crawled:      {result.pages_crawled}")
    print(f"Pages cleaned:      {result.pages_cleaned}")
    print(f"Chunks indexed:     {result.chunks_indexed}")
    print(f"Duplicates removed: {result.chunks_deduplicated}")
    print(f"Crawl failures:     {result.crawl_failures}")
    print(f"Corpus file:        {result.corpus_path}")
    print(f"Chroma collection:  {result.chroma_collection}")
    print(f"Corpus hash:        {result.corpus_hash[:16]}...")

    if result.top_sections:
        print("\nTop detected sections:")
        for section in result.top_sections:
            print(f"  - {section}")

    if result.errors:
        print("\nWarnings / errors:")
        for err in result.errors[:20]:
            print(f"  - {err}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")

    if result.chunks_indexed == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
