from rich.color import Color as RichColor
from rich.segment import Segment
from rich.style import Style
from rich.terminal_theme import DEFAULT_TERMINAL_THEME
from textual.color import Color
from textual.filter import ANSIToTruecolor
from textual.renderables.background_screen import BackgroundScreen

from tau_coding.tui.terminal_background import (
    KeepDefaultBackgroundScreen,
    KeepTerminalBackground,
)

DEFAULT = RichColor.parse("default")


def test_filter_keeps_the_default_background() -> None:
    line_filter = KeepTerminalBackground(DEFAULT_TERMINAL_THEME)

    style = line_filter.truecolor_style(Style(color="red", bgcolor="default"), DEFAULT)

    assert style.bgcolor == DEFAULT
    assert style.color is not None and style.color.triplet is not None


def test_filter_converts_other_backgrounds_like_textual() -> None:
    line_filter = KeepTerminalBackground(DEFAULT_TERMINAL_THEME)
    textual_filter = ANSIToTruecolor(DEFAULT_TERMINAL_THEME)
    style = Style(color="red", bgcolor="blue")
    background = RichColor.parse("#000000")

    assert line_filter.truecolor_style(style, background) == textual_filter.truecolor_style(
        style, background
    )


def test_filter_keeps_meta_for_mouse_handling() -> None:
    # Textual's own filter keeps meta too; this guards against rebuilding the
    # style with `without_color`, which drops it and broke clicks.
    line_filter = KeepTerminalBackground(DEFAULT_TERMINAL_THEME)
    style = Style(bgcolor="default") + Style.from_meta({"@click": "noop"})

    assert line_filter.truecolor_style(style, DEFAULT).meta == {"@click": "noop"}


def test_filter_dims_against_a_default_background() -> None:
    line_filter = KeepTerminalBackground(DEFAULT_TERMINAL_THEME)

    style = line_filter.truecolor_style(
        Style(color="#ffffff", bgcolor="default", dim=True), DEFAULT
    )

    assert not style.dim
    assert style.bgcolor == DEFAULT
    assert style.color is not None and style.color.triplet is not None


def test_backdrop_tints_text_and_keeps_the_default_background() -> None:
    tint = Color.parse("#ffffff").with_alpha(0.6)
    segment = Segment("x", Style(color="#000000", bgcolor="default", bold=True))

    (tinted,) = KeepDefaultBackgroundScreen.process_segments([segment], tint)
    (textual,) = BackgroundScreen.process_segments(
        [Segment("x", Style(color="#000000", bold=True))], tint
    )

    assert tinted.style is not None and textual.style is not None
    assert tinted.style.bgcolor == DEFAULT
    assert tinted.style.color == textual.style.color
    assert tinted.style.bold


def test_backdrop_tints_other_backgrounds_like_textual() -> None:
    tint = Color.parse("#ffffff").with_alpha(0.6)
    segments = [
        Segment("a", Style(color="#000000", bgcolor="#112233")),
        Segment("b", None),
        Segment("c", Style(link="https://example.com")),
    ]

    assert list(KeepDefaultBackgroundScreen.process_segments(segments, tint)) == list(
        BackgroundScreen.process_segments(segments, tint)
    )
