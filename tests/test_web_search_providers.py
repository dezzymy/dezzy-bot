from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.research import web_search
from metaculus_bot.research.provider_diagnostics import pop_provider_detail


class _ResponseContext:
    async def __aenter__(self) -> _ResponseContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None


class _SessionContext:
    def __init__(self) -> None:
        self.request: tuple[str, dict, dict] | None = None

    async def __aenter__(self) -> _SessionContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, endpoint: str, *, headers: dict, json: dict) -> _ResponseContext:
        self.request = (endpoint, headers, json)
        return _ResponseContext()


@pytest.mark.asyncio
async def test_nimbleway_provider_records_short_explanation_and_results(monkeypatch: pytest.MonkeyPatch) -> None:
    search = AsyncMock(return_value=[{"title": "A result", "url": "https://example.com", "description": "Evidence."}])
    monkeypatch.setattr(web_search, "nimbleway_search", search)
    question = cast(MetaculusQuestion, type("Question", (), {"question_text": "Will this happen?", "id_of_question": 1})())

    result = await web_search.nimbleway_search_provider()(question)
    detail = pop_provider_detail(1, "nimbleway")

    assert "[A result](https://example.com)" in result
    assert "Evidence." in result
    assert detail["explanation"] == "Nimbleway web search returned titles, URLs, and source descriptions."
    assert detail["counts"] == {"results": 1}
    search.assert_awaited_once_with("Will this happen?", focus="general")


@pytest.mark.asyncio
async def test_tavily_provider_formats_content_and_records_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    search = AsyncMock(return_value=[{"title": "A result", "url": "https://example.com", "content": "A snippet."}])
    monkeypatch.setattr(web_search, "tavily_search", search)
    question = cast(MetaculusQuestion, type("Question", (), {"question_text": "Will this happen?", "id_of_question": 2})())

    result = await web_search.tavily_search_provider()(question)
    detail = pop_provider_detail(2, "tavily")

    assert "A snippet." in result
    assert detail["explanation"] == "Tavily web search returned ranked source snippets."
    assert detail["counts"] == {"results": 1}
    search.assert_awaited_once_with("Will this happen?", topic="general")


def test_format_search_results_handles_empty_results() -> None:
    assert web_search.format_search_results("Tavily", []) == ""


@pytest.mark.asyncio
async def test_nimbleway_uses_documented_endpoint_and_bearer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _SessionContext()
    monkeypatch.setenv("NIMBLEWAY_API_KEY", "nimble-key")
    monkeypatch.setattr(web_search, "build_session", lambda **_: session)
    monkeypatch.setattr(web_search, "read_body_capped", AsyncMock(return_value=b'{"results": []}'))

    await web_search.nimbleway_search("query", focus="news")

    assert session.request is not None
    endpoint, headers, payload = session.request
    assert endpoint == "https://sdk.nimbleway.com/v2/search"
    assert headers == {"Authorization": "Bearer nimble-key"}
    assert payload["query"] == "query"
    assert payload["focus"] == "news"
    assert payload["search_depth"] == "standard"


@pytest.mark.asyncio
async def test_tavily_uses_documented_endpoint_key_and_date_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _SessionContext()
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setattr(web_search, "build_session", lambda **_: session)
    monkeypatch.setattr(web_search, "read_body_capped", AsyncMock(return_value=b'{"results": []}'))

    await web_search.tavily_search("query", topic="news", end_date="2026-09-30")

    assert session.request is not None
    endpoint, headers, payload = session.request
    assert endpoint == "https://api.tavily.com/search"
    assert headers == {"Content-Type": "application/json"}
    assert payload["api_key"] == "tavily-key"
    assert payload["query"] == "query"
    assert payload["topic"] == "news"
    assert payload["end_date"] == "2026-09-30"