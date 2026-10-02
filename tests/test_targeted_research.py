"""Unit tests for targeted_research module and its prompt functions."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S
from metaculus_bot.prompts import disagreement_crux_prompt, targeted_search_prompt
from metaculus_bot.research.targeted import extract_disagreement_crux, run_targeted_search


class TestDisagreementCruxPrompt:
    """Tests for the disagreement_crux_prompt formatting function."""

    def test_formats_predictions(self):
        result = disagreement_crux_prompt("Will X happen?", ["Analysis 1 text", "Analysis 2 text"])

        assert "Forecaster 1" in result
        assert "Forecaster 2" in result
        assert "Will X happen?" in result
        assert "Analysis 1 text" in result
        assert "Analysis 2 text" in result
        assert "factual question" in result.lower()

    def test_handles_many_predictions(self):
        predictions = [f"Prediction {i}" for i in range(6)]
        result = disagreement_crux_prompt("Question?", predictions)

        for i in range(1, 7):
            assert f"Forecaster {i}" in result


class TestTargetedSearchPrompt:
    """Tests for the targeted_search_prompt formatting function."""

    def test_includes_crux_and_question(self):
        result = targeted_search_prompt("Is the treaty signed?", "Will X happen?")

        assert "Is the treaty signed?" in result
        assert "Will X happen?" in result

    def test_benchmarking_warning_present_when_true(self):
        result = targeted_search_prompt("crux", "question", is_benchmarking=True)

        assert "benchmarking" in result.lower()
        assert "data leakage" in result.lower()

    def test_benchmarking_warning_absent_when_false(self):
        result = targeted_search_prompt("crux", "question", is_benchmarking=False)

        assert "benchmarking" not in result.lower()
        assert "data leakage" not in result.lower()


class TestExtractDisagreementCrux:
    """Tests for the async extract_disagreement_crux function."""

    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_result(self):
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "The crux is X"

        result = await extract_disagreement_crux(mock_llm, "question", ["pred1", "pred2"])

        assert result == "The crux is X"
        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "pred1" in prompt_arg
        assert "pred2" in prompt_arg

    @pytest.mark.asyncio
    async def test_propagates_errors_after_broad_retry(self):
        """A fast retryable error is retried (broad, 30s-gated) then propagates.

        Round-2: the crux invoke is wrapped in invoke_with_broad_retry. A fast
        RuntimeError is broadly-retryable, so it retries the full backoff schedule
        then re-raises — the propagation contract is preserved (the error surfaces,
        not swallowed), just after retries. asyncio.sleep is patched so the
        backoffs are instant; the awaitable is invoked len(backoffs)+1 = 4 times.
        """
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")

        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            pytest.raises(RuntimeError, match="LLM timeout"),
        ):
            await extract_disagreement_crux(mock_llm, "question", ["pred1", "pred2"])

        assert mock_llm.invoke.await_count == 4

    @pytest.mark.asyncio
    async def test_slow_crux_error_not_retried(self):
        """A crux failure past the 30s gate is NOT retried — it propagates on the first attempt."""
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("slow analyzer stall")
        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 20)

        with (
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            pytest.raises(RuntimeError, match="slow analyzer stall"),
        ):
            await extract_disagreement_crux(mock_llm, "question", ["pred1", "pred2"])

        assert mock_llm.invoke.await_count == 1


class TestRunTargetedSearch:
    """Tests for the async run_targeted_search function."""

    @pytest.mark.asyncio
    async def test_searches_tavily_and_formats_results(self):
        search = AsyncMock(return_value=[{"title": "Source", "url": "https://example.com", "content": "Evidence."}])

        with patch("metaculus_bot.research.targeted.tavily_search", search):
            result = await run_targeted_search("crux text", "question text")

        assert "Evidence." in result
        assert "https://example.com" in result
        search.assert_awaited_once_with("crux text\nQuestion: question text")

    @pytest.mark.asyncio
    async def test_passes_benchmarking_flag(self):
        search = AsyncMock(return_value=[])

        with patch("metaculus_bot.research.targeted.tavily_search", search):
            await run_targeted_search("crux", "q", is_benchmarking=True)

        assert search.await_args is not None
        query = search.await_args.args[0]
        assert query.startswith("crux\nQuestion: q")
        assert "benchmarking run" in query
        assert "prediction-market odds" in query

    @pytest.mark.asyncio
    async def test_enforces_wall_clock_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stalled search request is bounded by the caller's wall timeout."""
        monkeypatch.setattr("metaculus_bot.research.targeted.NATIVE_SEARCH_WALL_TIMEOUT", 0.05)

        async def hang(_query: str) -> list[dict[str, str]]:
            await asyncio.sleep(5)
            return []

        with (
            patch("metaculus_bot.research.targeted.tavily_search", side_effect=hang),
            pytest.raises(asyncio.TimeoutError),
        ):
            await run_targeted_search("crux", "question text")
