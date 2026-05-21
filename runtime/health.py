"""Minimal HTTP health endpoints for Render Web Service port checks."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Company Support AI", docs_url=None, redoc_url=None)

_HEALTH_JSON = {"status": "ok", "service": "company-support-ai"}


@app.get("/")
async def root() -> dict[str, str]:
    return _HEALTH_JSON


@app.get("/health")
async def health() -> dict[str, str]:
    return _HEALTH_JSON
