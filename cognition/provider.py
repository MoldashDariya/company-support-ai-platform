"""OpenAI-compatible async LLM client (Cerebras, OpenAI, local proxies)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from domain.models import ChatTurn
from runtime import settings


class CerebrasLanguageModel:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

    async def generate(
        self,
        system: str,
        history: list[ChatTurn],
        user_text: str,
    ) -> str:
        chunks: list[str] = []
        async for part in self.generate_stream(system, history, user_text):
            chunks.append(part)
        return "".join(chunks)

    async def generate_stream(
        self,
        system: str,
        history: list[ChatTurn],
        user_text: str,
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": system},
            *[t.to_llm_dict() for t in history],
            {"role": "user", "content": user_text},
        ]
        stream = await self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
