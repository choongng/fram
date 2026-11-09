"""`tau probe`: a few short fixed requests that warm-start a model's profile.

The probe sends three requests through the same timing path as the agent loop
(:func:`tau_agent.loop.timed_response`):

1. a prompt of about 1,500 tokens asking for one word, for prefill speed;
2. a request for a short Python function and a paragraph about it (a few
   hundred tokens of code and prose), for decode speed and chars per token;
3. the same request with ``max_tokens`` set well below its length, to see
   whether the server honours the cap.

Every request is capped, so a thinking model cannot run long: on a model that
decodes 30 tokens/s the probe takes well under a minute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tau_agent.loop import timed_response
from tau_agent.messages import AssistantMessage, UserMessage, current_timestamp_ms
from tau_agent.provider import ModelProvider
from tau_ai.model_limits import ModelLimitsProvider, RuntimeModelLimits
from tau_ai.model_profile import ModelProfile, ProfileKey
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.model_profiles import ModelProfileStore
from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    ProviderConfig,
    ProviderModelMetadata,
    load_provider_settings,
    provider_model_endpoint,
)
from tau_coding.provider_runtime import create_model_provider

PROBE_SYSTEM = "You are being measured. Follow the instruction exactly."
PREFILL_MAX_TOKENS = 16
DECODE_MAX_TOKENS = 400
CAP_MAX_TOKENS = 48
#: Asks for code and prose, like an agent's output, so the chars-per-token
#: sample is representative; it runs to a few hundred tokens.
OUTPUT_INSTRUCTION = (
    "Write a short Python function with a docstring that parses ISO-8601 dates, "
    "then a paragraph explaining it."
)

#: A provider for one probe request, built with the given ``max_tokens``.
type ProbeProviderFactory = Callable[[int], ModelProvider]


def prefill_prompt(nonce: int) -> str:
    """About 1,500 tokens of distinct sentences, led by ``nonce`` so no cache holds it."""
    lines = [f"Probe {nonce}."]
    lines.extend(
        f"Record {n}: the sensor in bay {n % 17} read {n * 13 % 997} units at minute {n}."
        for n in range(1, 101)
    )
    lines.append("Reply with the single word OK.")
    return "\n".join(lines)


def cap_honoured(message: AssistantMessage | None, cap: int) -> bool | None:
    """Whether a request capped at ``cap`` tokens was held to it; None if it cannot tell.

    More reported output than the cap means the server ignored it. A length
    stop within the cap means it applied it. A response that stopped on its
    own below the cap, or reported no usage, shows neither.
    """
    if message is None or message.usage.output <= 0:
        return None
    if message.usage.output > cap:
        return False
    if message.stop_reason == "length":
        return True
    return None


class ProbeError(RuntimeError):
    """A probe request failed, so nothing was measured or stored."""


async def _close(provider: ModelProvider) -> None:
    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        await aclose()


async def _ask(
    provider_for: ProbeProviderFactory, model: str, max_tokens: int, prompt: str
) -> AssistantMessage | None:
    provider = provider_for(max_tokens)
    try:
        message = await timed_response(
            provider, model=model, system=PROBE_SYSTEM, messages=[UserMessage(content=prompt)]
        )
    finally:
        await _close(provider)
    if message is not None and message.stop_reason in {"error", "aborted"}:
        raise ProbeError(message.error_message or f"The request ended: {message.stop_reason}.")
    return message


async def _discover_limits(
    provider_for: ProbeProviderFactory, model: str
) -> RuntimeModelLimits | None:
    provider = provider_for(PREFILL_MAX_TOKENS)
    try:
        if isinstance(provider, ModelLimitsProvider):
            return await provider.discover_model_limits(model)
        return None
    except Exception:
        return None
    finally:
        await _close(provider)


async def probe_model(
    key: ProfileKey,
    provider_for: ProbeProviderFactory,
    *,
    store: ModelProfileStore,
) -> ModelProfile:
    """Probe ``key``'s model and store what it measured as its profile's warm start."""
    profile = store.get(key)

    prefill = await _ask(
        provider_for, key.model, PREFILL_MAX_TOKENS, prefill_prompt(current_timestamp_ms())
    )
    decode = await _ask(provider_for, key.model, DECODE_MAX_TOKENS, OUTPUT_INSTRUCTION)
    capped = await _ask(provider_for, key.model, CAP_MAX_TOKENS, OUTPUT_INSTRUCTION)

    for message in (prefill, decode, capped):
        if message is not None:
            profile = profile.with_message(message, current_timestamp_ms())

    honoured = cap_honoured(capped, CAP_MAX_TOKENS)
    if honoured is not None:
        profile = replace(profile, max_tokens_honoured=honoured)

    limits = await _discover_limits(provider_for, key.model)
    if limits is not None:
        profile = replace(
            profile,
            context_window=limits.context_window,
            max_output_tokens=limits.max_output_tokens or profile.max_output_tokens,
        )

    profile = replace(profile, updated_at_ms=current_timestamp_ms())
    store.put(profile)
    return profile


def with_model_max_tokens(provider: ProviderConfig, model: str, max_tokens: int) -> ProviderConfig:
    """``provider`` with ``model``'s output cap set to ``max_tokens``."""
    metadata = provider.model_metadata.get(model) or ProviderModelMetadata()
    return replace(
        provider,
        model_metadata={**provider.model_metadata, model: replace(metadata, max_tokens=max_tokens)},
    )


async def run_probe(
    provider_name: str | None,
    model: str | None,
    *,
    paths: TauPaths | None = None,
) -> ModelProfile:
    """Probe a configured provider's model (defaults: the configured default ones)."""
    resolved = paths or TauPaths()
    settings = load_provider_settings(resolved)
    provider = settings.get_provider(provider_name or settings.default_provider)
    selected_model = model or provider.default_model
    credentials = FileCredentialStore(credentials_path(resolved))

    def provider_for(max_tokens: int) -> ModelProvider:
        return create_model_provider(
            with_model_max_tokens(provider, selected_model, max_tokens),
            credential_store=credentials,
            model=selected_model,
        )

    key = ProfileKey(
        provider=provider.name,
        endpoint=provider_model_endpoint(provider, selected_model),
        model=selected_model,
    )
    return await probe_model(key, provider_for, store=ModelProfileStore.at(resolved))
