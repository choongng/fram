"""Let the terminal's own background show through Textual's color conversion.

Textual converts every color without an RGB value, the terminal's default
included, to a fixed truecolor from its ANSI theme. A Tau theme with
`terminal_background` paints `ansi_default` where it would otherwise paint its
screen background, so the TUI keeps the terminal's background. That only works
if the default background reaches the terminal unconverted, which is what this
filter does; every other color is converted as before.

Modal screens tint the screen behind them by blending every cell's colors
with their own, and Textual counts the default background as black there. So
the backdrop behind a dialog would turn grey in a light theme.
`KeepDefaultBackgroundScreen` tints those cells' text but leaves their
background to the terminal.

Textual's own switch for this (`App.ansi_color`) also turns on its `:ansi`
widget styles and changes the look of the prompt and other widgets, so it is
not used.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import textual.screen
from rich.color import Color as RichColor
from rich.segment import Segment
from rich.style import Style
from textual.app import App
from textual.color import Color
from textual.filter import ANSIToTruecolor
from textual.renderables.background_screen import BackgroundScreen

_DEFAULT_BACKGROUND = Style(bgcolor="default")


class KeepTerminalBackground(ANSIToTruecolor):
    """Convert ANSI colors to truecolor, except the terminal's default background."""

    @lru_cache(1024)  # noqa: B019 - Textual's filter caches the same way.
    def truecolor_style(self, style: Style, background: RichColor) -> Style:
        # Dimming blends toward the background, which needs an RGB value.
        if background.triplet is None:
            background = RichColor.from_triplet(
                background.get_truecolor(self._terminal_theme, foreground=False)
            )
        converted = super().truecolor_style(style, background)
        bgcolor = style.bgcolor
        if bgcolor is None or not bgcolor.is_default:
            return converted
        # Adding a style replaces the background but keeps the rest, including
        # the meta that Textual's mouse handling reads.
        return converted + _DEFAULT_BACKGROUND


def install_terminal_background_filter(app: App[Any]) -> None:
    """Replace the app's ANSI-to-truecolor filter with `KeepTerminalBackground`."""
    filters = app._filters
    for index, line_filter in enumerate(filters):
        if type(line_filter) is ANSIToTruecolor:
            filters[index] = KeepTerminalBackground(
                line_filter._terminal_theme, enabled=line_filter.enabled
            )


def _has_default_background(segment: Segment) -> bool:
    style = segment.style
    return (
        not segment.control
        and style is not None
        and style.bgcolor is not None
        and style.bgcolor.is_default
    )


class KeepDefaultBackgroundScreen(BackgroundScreen):
    """Tint the screen behind a modal, leaving the default background alone."""

    @classmethod
    def process_segments(cls, segments: Iterable[Segment], color: Color) -> Iterable[Segment]:
        segments = list(segments)
        keep = [_has_default_background(segment) for segment in segments]
        # Without a background, Textual tints only the text; meta and links
        # are cleared by the tint either way.
        bare = [
            Segment(text, style.without_color + Style(color=style.color), control)
            if kept and style is not None
            else Segment(text, style, control)
            for (text, style, control), kept in zip(segments, keep, strict=True)
        ]
        # The tint yields one segment for each one it is given.
        tinted = super().process_segments(bare, color)
        for (text, style, control), kept in zip(tinted, keep, strict=True):
            if kept and style is not None:
                style += _DEFAULT_BACKGROUND
            yield Segment(text, style, control)


def install_modal_backdrop() -> None:
    """Make modal screens draw their backdrop with `KeepDefaultBackgroundScreen`.

    `Screen.render` builds the backdrop from the name it imported, so this
    swaps that name. It is global to the process, and safe to call again.
    """
    textual.screen.BackgroundScreen = KeepDefaultBackgroundScreen  # type: ignore[attr-defined]
