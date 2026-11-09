"""Where Tau keeps measured per-model profiles, and how callers read them.

The measuring itself is :mod:`tau_ai.model_profile`; this module stores the
profiles as one JSON file under ``TAU_HOME`` (``model-profiles.json``), written
atomically, and offers :func:`read_model_profiles` for in-process callers. The
same JSON is what ``tau profile show --json`` prints for other programs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeGuard

from tau_agent.messages import AssistantMessage, current_timestamp_ms
from tau_ai.model_profile import DECODE_BUCKETS, AveragedValue, ModelProfile, ProfileKey
from tau_coding.catalog_loader import _atomic_write_text
from tau_coding.context_window import CHARS_PER_TOKEN
from tau_coding.paths import TauPaths

#: Version of the stored file and of ``tau profile show --json``.
MODEL_PROFILES_VERSION = 1
#: Tokens of evidence a chars-per-token ratio needs before it replaces the
#: default in an estimate: about a dozen ordinary replies, or one long thinking
#: turn. A heuristic; a few short replies can sit far from the model's real ratio.
MIN_TRUSTED_RATIO_WEIGHT = 4000


class ModelProfileStore:
    """The profiles file: load, update one profile, save atomically.

    Two processes updating the same file at once can lose one update (the
    later write wins); a profile is an estimate, so that costs one sample.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def at(cls, paths: TauPaths | None = None) -> ModelProfileStore:
        return cls((paths or TauPaths()).model_profiles_path)

    def load(self) -> dict[ProfileKey, ModelProfile]:
        """Every stored profile; an absent or unreadable file holds none."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        profiles: dict[ProfileKey, ModelProfile] = {}
        for item in data.get("profiles", []) if isinstance(data, dict) else []:
            try:
                profile = ModelProfile.from_json(item)
            except (KeyError, TypeError, ValueError):
                continue
            profiles[profile.key] = profile
        return profiles

    def get(self, key: ProfileKey) -> ModelProfile:
        """The stored profile for ``key``, or an empty one."""
        return self.load().get(key) or ModelProfile(key=key)

    def save(self, profiles: Iterable[ModelProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(profiles_json(profiles), indent=2) + "\n")

    def put(self, profile: ModelProfile) -> None:
        """Store ``profile``, replacing the one with its key."""
        profiles = self.load()
        profiles[profile.key] = profile
        self.save(profiles.values())

    def record_response(self, key: ProfileKey, message: AssistantMessage) -> ModelProfile | None:
        """Fold one finished response into ``key``'s profile; None if it gave no samples."""
        profiles = self.load()
        current = profiles.get(key) or ModelProfile(key=key)
        updated = current.with_message(message, current_timestamp_ms())
        if updated is current:
            return None
        profiles[key] = updated
        self.save(profiles.values())
        return updated


def profiles_json(profiles: Iterable[ModelProfile]) -> dict[str, Any]:
    """The documented JSON document: ``{"version": 1, "profiles": [...]}``."""
    ordered = sorted(profiles, key=lambda p: (p.key.provider, p.key.model, p.key.endpoint))
    return {
        "version": MODEL_PROFILES_VERSION,
        "profiles": [profile.to_json() for profile in ordered],
    }


def read_model_profiles(
    *,
    provider: str | None = None,
    model: str | None = None,
    paths: TauPaths | None = None,
) -> list[ModelProfile]:
    """Stored profiles, optionally only one provider's and/or one model's."""
    return [
        profile
        for profile in ModelProfileStore.at(paths).load().values()
        if (provider is None or profile.key.provider == provider)
        and (model is None or profile.key.model == model)
    ]


def trusted_ratio(value: AveragedValue | None) -> TypeGuard[AveragedValue]:
    """Whether a measured ratio rests on `MIN_TRUSTED_RATIO_WEIGHT` tokens."""
    return value is not None and value.value > 0 and value.weight >= MIN_TRUSTED_RATIO_WEIGHT


def context_chars_per_token(profile: ModelProfile | None) -> float:
    """Characters a token for context estimates: the measured combined ratio, once trusted."""
    value = profile.combined_chars_per_token if profile is not None else None
    return value.value if trusted_ratio(value) else CHARS_PER_TOKEN


def _stat(value: AveragedValue | None, unit: str) -> str:
    if value is None:
        return "-"
    return f"{value.value:.1f}{unit} (n={value.samples})"


def format_profile(profile: ModelProfile) -> str:
    """A few human-readable lines for one profile."""
    key = profile.key
    decode = ", ".join(
        f"{bucket} {_stat(profile.decode_tokens_per_second.get(bucket), ' tok/s')}"
        for bucket, _lower in DECODE_BUCKETS
    )
    honoured = profile.max_tokens_honoured
    return "\n".join(
        [
            f"{key.provider}/{key.model} @ {key.endpoint or '(no endpoint)'}",
            f"  decode: {decode}",
            f"  prefill (uncached): {_stat(profile.prefill_tokens_per_second, ' tok/s')}",
            "  chars/token: "
            f"thinking {_stat(profile.thinking_chars_per_token, '')}, "
            f"output {_stat(profile.output_chars_per_token, '')}, "
            f"combined {_stat(profile.combined_chars_per_token, '')}",
            f"  limits: context {profile.context_window or '-'}, "
            f"max output {profile.max_output_tokens or '-'}, "
            f"largest output seen {profile.largest_output_tokens or '-'}",
            "  max_tokens honoured: "
            + ("unknown" if honoured is None else "yes" if honoured else "no"),
        ]
    )
