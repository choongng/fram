from __future__ import annotations

from collections.abc import Callable

import pytest

from tau_coding import branding, oauth
from tau_coding.branding import ENTRY_POINT_GROUP, Branding, load_branding


class FakeEntryPoint:
    def __init__(self, name: str, load: Callable[[], object]) -> None:
        self.name = name
        self.load = load


def _install(monkeypatch: pytest.MonkeyPatch, *eps: FakeEntryPoint) -> None:
    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        assert group == ENTRY_POINT_GROUP
        return list(eps)

    monkeypatch.setattr(branding, "entry_points", fake_entry_points)


def _raise() -> object:
    raise ImportError("broken branding plugin")


def test_without_an_entry_point_branding_is_taus(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    assert load_branding() == Branding(
        product_name="Tau", command_name="tau", mark="τ", logo="τ = 2π"
    )


def test_entry_point_mapping_overrides_the_fields_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, FakeEntryPoint("acme", lambda: {"product_name": "Acme", "mark": "A"}))

    assert load_branding() == Branding(product_name="Acme", mark="A")


def test_broken_or_malformed_entry_points_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        FakeEntryPoint("a-broken", _raise),
        FakeEntryPoint("b-not-a-mapping", lambda: "Acme"),
        FakeEntryPoint("c-unknown-field", lambda: {"slogan": "x"}),
        FakeEntryPoint("d-good", lambda: {"product_name": "Good"}),
    )

    assert load_branding() == Branding(product_name="Good")


def test_entry_points_are_tried_in_name_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        FakeEntryPoint("zeta", lambda: {"product_name": "Zeta"}),
        FakeEntryPoint("alpha", lambda: {"product_name": "Alpha"}),
    )

    assert load_branding().product_name == "Alpha"


def test_oauth_page_escapes_the_product_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oauth, "BRANDING", Branding(product_name="A<b>&C"))

    assert "<title>A&lt;b&gt;&amp;C OAuth</title>" in oauth._oauth_html("done")
