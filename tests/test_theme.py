"""Design-system guards: every theme stays legible, and helpers format honestly."""

import tomllib
from pathlib import Path

import pytest

from ui import theme as t

ROOT = Path(__file__).resolve().parent.parent


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", t.THEMES.values(), ids=lambda th: th.key)
def test_theme_text_and_marks_meet_contrast_targets(theme):
    tok = t.tokens(theme)
    for surface in (theme.bg, theme.s1, theme.s2):
        assert contrast(theme.text, surface) >= 7, "body text"
        assert contrast(theme.muted, surface) >= 4.5, "secondary text"
        assert contrast(theme.faint, surface) >= 4.0, "tertiary text"
        assert contrast(theme.accent_fg, surface) >= 4.5, "links / active nav"
    assert contrast(theme.on_accent, theme.accent) >= 4.5, "button label"
    assert contrast(theme.mark, theme.s1) >= 3 and contrast(theme.mark, theme.s3) >= 3, "bars"
    for role in ("good-fg", "warn-fg", "serious-fg", "bad-fg"):
        assert contrast(tok[role], theme.s1) >= 4.5, role
    # Position dots sit beside a text label, so they only need to be visible.
    for role in ("rb", "wr", "te"):
        assert contrast(tok[role], theme.s1) >= 2.5, role


def test_config_toml_matches_the_default_theme():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    flat = {"theme." + k: v for k, v in cfg["theme"].items() if not isinstance(v, dict)}
    flat.update({"theme.sidebar." + k: v for k, v in cfg["theme"]["sidebar"].items()})
    for key, value in t.THEMES[t.DEFAULT_THEME].streamlit_options().items():
        assert str(flat[key]).lower() == value.lower(), key


def test_logo_is_drawn_in_the_theme_colors():
    for theme in t.THEMES.values():
        svg = t.logo_svg(theme)
        assert svg.startswith("<svg") and theme.text in svg and theme.mark in svg
        assert "DraftAid" in svg and "DraftAid" not in t.logo_svg(theme, mark_only=True)


def test_percent_never_overstates_certainty():
    assert [t.fmt_pct(p) for p in (0, 0.004, 0.5, 0.996, 1)] == ["0%", "<1%", "50%", ">99%", "100%"]
    assert t.fmt_pct(None) == "–" and t.fmt_pct(float("nan")) == "–"


def test_small_formatters():
    assert [t.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]
    assert "+2.5 pts" in t.delta(2.5, " pts") and "da-up" in t.delta(2.5)
    assert "−3.6" in t.delta(-3.6) and "da-down" in t.delta(-3.6)
    assert "no change" in t.delta(0.01) and "no change" in t.delta(None)
    assert t.esc("<b>&") == "&lt;b&gt;&amp;" and t.esc(None) == ""
    assert "da-chip rb" in t.pos_chip("RB") and "da-chip'" in t.pos_chip("QB")


def test_table_headers_can_shorten_on_phones():
    html = t.table([(("Rest of season", "Season"), "r"), ("Bye", "c")], ["<tr><td>x</td></tr>"])
    assert "<span class='hide-m'>Rest of season</span><span class='show-m'>Season</span>" in html
    assert "<th class='c'>Bye</th>" in html
