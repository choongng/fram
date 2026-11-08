from tau_agent.messages import AssistantMessage, Usage
from tau_coding.events import PromptCacheMissEvent
from tau_coding.prompt_cache import PromptCacheMissDetector


def _response(
    *, input: int, cache_read: int = 0, model: str = "qwen", stop_reason: str = "stop"
) -> AssistantMessage:
    return AssistantMessage(
        provider="llama",
        model=model,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(input=input, cache_read=cache_read),
    )


def test_uncached_large_prompt_is_a_miss_only_after_the_provider_has_cached() -> None:
    detector = PromptCacheMissDetector(min_prompt_tokens=1_000)

    assert detector.observe(_response(input=30_000)) is None
    assert detector.observe(_response(input=200, cache_read=29_000)) is None
    assert detector.observe(_response(input=31_000)) == PromptCacheMissEvent(
        provider="llama", model="qwen", prompt_tokens=31_000
    )


def test_small_prompts_errors_and_other_models_are_not_flagged() -> None:
    detector = PromptCacheMissDetector(min_prompt_tokens=1_000)
    detector.observe(_response(input=200, cache_read=29_000))

    assert detector.observe(_response(input=999)) is None
    assert detector.observe(_response(input=31_000, stop_reason="error")) is None
    assert detector.observe(_response(input=31_000, model="other")) is None


def test_expected_miss_skips_only_the_next_response() -> None:
    detector = PromptCacheMissDetector(min_prompt_tokens=1_000)
    detector.observe(_response(input=200, cache_read=29_000))

    detector.expect_miss()

    assert detector.observe(_response(input=31_000)) is None
    assert detector.observe(_response(input=32_000)) is not None
