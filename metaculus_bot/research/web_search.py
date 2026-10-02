"""Nimbleway and Tavily search providers shared by research and gap-fill tools."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    NIMBLEWAY_API_KEY_ENV,
    NIMBLEWAY_SEARCH_ENDPOINT,
    TAVILY_API_KEY_ENV,
    TAVILY_SEARCH_ENDPOINT,
    WEB_SEARCH_MAX_RESPONSE_BYTES,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT_S,
)
from metaculus_bot.research.http_fetch import build_session, read_body_capped
from metaculus_bot.research.provider_diagnostics import record_provider_detail

_NIMBLEWAY_EXPLANATION = "Nimbleway web search returned titles, URLs, and source descriptions."
_TAVILY_EXPLANATION = "Tavily web search returned ranked source snippets."


async def _post_json(
    endpoint: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    async with build_session(timeout_s=WEB_SEARCH_TIMEOUT_S, connector_limit=4) as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            body = await read_body_capped(
                response,
                max_bytes=WEB_SEARCH_MAX_RESPONSE_BYTES,
                label=f"{provider} search",
            )
    if body is None:
        raise ValueError(f"{provider} search response exceeded the size limit")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise ValueError(f"{provider} search response was not a JSON object")
    return decoded


async def nimbleway_search(query: str, *, focus: str = "general") -> list[dict[str, Any]]:
    api_key = os.getenv(NIMBLEWAY_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Nimbleway API key: {NIMBLEWAY_API_KEY_ENV}")
    data = await _post_json(
        NIMBLEWAY_SEARCH_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "query": query,
            "focus": focus,
            "search_depth": "standard",
            "full_content": False,
            "max_results": WEB_SEARCH_MAX_RESULTS,
            "output_format": "markdown",
        },
        provider="Nimbleway",
    )
    results = data.get("results")
    return [result for result in results if isinstance(result, dict)] if isinstance(results, list) else []


async def tavily_search(
    query: str,
    *,
    topic: str = "general",
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    api_key = os.getenv(TAVILY_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Tavily API key: {TAVILY_API_KEY_ENV}")
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "topic": topic,
        "search_depth": "basic",
        "max_results": WEB_SEARCH_MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,
    }
    if end_date is not None:
        payload["end_date"] = end_date
    data = await _post_json(
        TAVILY_SEARCH_ENDPOINT,
        headers={"Content-Type": "application/json"},
        payload=payload,
        provider="Tavily",
    )
    results = data.get("results")
    return [result for result in results if isinstance(result, dict)] if isinstance(results, list) else []


def format_search_results(provider: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = [f"### {provider} search results"]
    for result in results:
        title = str(result.get("title") or "Untitled source").strip()
        url = str(result.get("url") or "").strip()
        snippet = str(result.get("content") or result.get("description") or "").strip()
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


def _search_provider(
    name: str,
    explanation: str,
    search: Callable[..., Awaitable[list[dict[str, Any]]]],
    *,
    focus: str = "general",
    topic: str = "general",
) -> Callable[[MetaculusQuestion], Awaitable[str]]:
    async def _fetch(question: MetaculusQuestion) -> str:
        kwargs = {"focus": focus} if name == "nimbleway" else {"topic": topic}
        results = await search(question.question_text, **kwargs)
        qid = getattr(question, "id_of_question", None)
        record_provider_detail(
            qid,
            name,
            {"explanation": explanation, "counts": {"results": len(results)}},
        )
        return format_search_results(name.title(), results)

    return _fetch


def nimbleway_search_provider() -> Callable[[MetaculusQuestion], Awaitable[str]]:
    return _search_provider("nimbleway", _NIMBLEWAY_EXPLANATION, nimbleway_search)


def tavily_search_provider() -> Callable[[MetaculusQuestion], Awaitable[str]]:
    return _search_provider("tavily", _TAVILY_EXPLANATION, tavily_search)