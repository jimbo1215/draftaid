"""DraftAid design system: themes, global CSS, and small HTML building blocks.

A theme is a set of tokens (surfaces, ink, one accent). Everything on a page
is drawn from those tokens through the helpers below, so switching theme is a
token swap. Conventions the helpers hold to:

  - One sans face throughout; numbers align with tabular figures only where
    they sit in columns.
  - Color does one job at a time. The accent marks emphasis and data; status
    colors (good / warn / bad) are reserved for state and always carry a label;
    positions get three validated hues (RB / WR / TE) and neutral for the rest.
  - Text is never colored to carry meaning. A dot, bar, or border does that.

Components return HTML strings and `html()` puts them on the page. Keep each
element on one line: Markdown treats indented or blank-line-separated HTML as
code blocks.
"""

import html as _html
import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st

DEFAULT_THEME = "midnight"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"


@dataclass(frozen=True)
class Theme:
    key: str
    label: str
    blurb: str
    base: str            # Streamlit base: "dark" | "light"
    bg: str              # page
    side: str            # sidebar
    s1: str              # card surface
    s2: str              # inset / input
    s3: str              # track, neutral fill
    line: str
    line2: str
    text: str
    muted: str
    faint: str
    accent: str          # interactive fill (buttons, toggles)
    on_accent: str       # text on the accent fill
    accent_fg: str       # accent as text on a surface (links, active nav)
    mark: str            # accent as a data mark (bars, dots)

    @property
    def dark(self) -> bool:
        return self.base == "dark"

    def streamlit_options(self) -> dict:
        """The native-widget theme, so buttons and inputs match the tokens."""
        return {
            "theme.base": self.base,
            "theme.primaryColor": self.accent,
            "theme.backgroundColor": self.bg,
            "theme.secondaryBackgroundColor": self.s2,
            "theme.textColor": self.text,
            "theme.linkColor": self.accent_fg,
            "theme.borderColor": self.line2,
            "theme.sidebar.backgroundColor": self.side,
            "theme.sidebar.secondaryBackgroundColor": self.s2,
            "theme.sidebar.borderColor": self.line,
        }


THEMES = {t.key: t for t in (
    Theme("midnight", "Midnight", "Slate dark with a blue accent",
          base="dark", bg="#0E1217", side="#0B0E13", s1="#151A21", s2="#1B212A", s3="#252C37",
          line="#232A34", line2="#303947", text="#E8EBF0", muted="#9BA6B5", faint="#808B9C",
          accent="#2672CE", on_accent="#FFFFFF", accent_fg="#6DA7EC", mark="#3987E5"),
    Theme("daylight", "Daylight", "Light paper with ink text",
          base="light", bg="#F5F4F0", side="#EFEDE8", s1="#FFFFFF", s2="#F3F2EE", s3="#E9E7E1",
          line="#E3E1DA", line2="#CFCCC3", text="#14171C", muted="#5B616B", faint="#666C77",
          accent="#2672CE", on_accent="#FFFFFF", accent_fg="#1C5CAB", mark="#2A78D6"),
    Theme("dusk", "Dusk", "Deep indigo dark with a violet accent",
          base="dark", bg="#110F18", side="#0D0C13", s1="#181622", s2="#1F1D2B", s3="#2A2838",
          line="#282636", line2="#383548", text="#ECEAF5", muted="#A5A2B8", faint="#8A87A0",
          accent="#6250D6", on_accent="#FFFFFF", accent_fg="#A39AEE", mark="#9085E9"),
)}

# Mode-level palettes: fixed across themes of the same base.
_MODE = {
    "dark": {
        "good": "#0CA30C", "warn": "#FAB219", "serious": "#EC835A", "bad": "#D03B3B",
        "good_fg": "#34C759", "warn_fg": "#FAB219", "serious_fg": "#EC835A", "bad_fg": "#F07272",
        "rb": "#199E70", "wr": "#3987E5", "te": "#D95926",
        "pos": "#3987E5", "neg": "#E66767",       # diverging poles: favorable / unfavorable
    },
    "light": {
        "good": "#0CA30C", "warn": "#FAB219", "serious": "#EC835A", "bad": "#D03B3B",
        "good_fg": "#006300", "warn_fg": "#8A5A00", "serious_fg": "#A8481C", "bad_fg": "#B3261E",
        "rb": "#1BAF7A", "wr": "#2A78D6", "te": "#EB6834",
        "pos": "#2A78D6", "neg": "#E34948",
    },
}


def tokens(theme: Theme) -> dict:
    """Every CSS custom property for a theme."""
    return {
        "bg": theme.bg, "side": theme.side, "s1": theme.s1, "s2": theme.s2, "s3": theme.s3,
        "line": theme.line, "line2": theme.line2, "text": theme.text, "muted": theme.muted,
        "faint": theme.faint, "accent": theme.accent, "on-accent": theme.on_accent,
        "accent-fg": theme.accent_fg, "mark": theme.mark,
        **{k.replace("_", "-"): v for k, v in _MODE[theme.base].items()},
    }


CSS = """
<style>
:root { __TOKENS__
  --accent-dim: color-mix(in srgb, var(--mark) 14%, transparent);
  --track: color-mix(in srgb, var(--mark) 16%, var(--s3));
  --font: __FONT__;
}
/* ---------- chrome */
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.2rem !important; max-width: 1160px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebarNav"] a { border-radius: 8px; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--accent-dim); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--accent-fg) !important; }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line); }
[data-testid="stSidebarNav"] header, [data-testid="stNavSectionHeader"] {
  font-size: 11px !important; font-weight: 600; letter-spacing: .07em;
  text-transform: uppercase; color: var(--faint) !important; }
h1, h2, h3, h4 { letter-spacing: -.01em; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }
div[data-testid="stColumn"] button { min-width: 40px; }
button[kind="primary"], button[kind="primaryFormSubmit"] {
  color: var(--on-accent) !important; font-weight: 600; }
[data-testid="stExpander"] details { border-color: var(--line) !important; background: var(--s1); }
[data-testid="stExpander"] summary p { font-size: 13px; color: var(--muted); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---------- type + primitives */
.da-num { font-variant-numeric: tabular-nums; }
.da-eyebrow { font-size: 11.5px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); display: flex; gap: 8px;
  align-items: center; flex-wrap: wrap; }
.da-eyebrow .sep { color: var(--line2); }
.da-eyebrow .live { display: inline-flex; align-items: center; gap: 6px; }
.da-eyebrow .live::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--mark); box-shadow: 0 0 0 3px var(--accent-dim); }
.da-title { font-weight: 700; font-size: 30px; line-height: 1.12; letter-spacing: -.02em;
  color: var(--text); margin: 6px 0 4px; }
.da-sub { color: var(--muted); font-size: 14px; margin-bottom: 4px; max-width: 70ch; }
.da-section { display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; margin: 30px 0 10px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.da-section .h { font-size: 15.5px; font-weight: 650; letter-spacing: -.005em; margin: 0;
  padding: 0; color: var(--text); line-height: 1.3; white-space: nowrap; }
.da-section span { font-size: 12px; color: var(--faint); text-align: right; min-width: 0; }

.da-chip { display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px;
  font-weight: 650; letter-spacing: .03em; padding: 1px 7px 1px 6px; border-radius: 999px;
  line-height: 1.5; background: var(--s3); color: var(--text); white-space: nowrap; }
.da-chip i { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); flex: none; }
.da-chip.rb i { background: var(--rb); } .da-chip.wr i { background: var(--wr); }
.da-chip.te i { background: var(--te); }
.da-tag { display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
  font-weight: 600; letter-spacing: .02em; padding: 3px 9px; border-radius: 999px;
  white-space: nowrap; background: var(--s3); color: var(--text); }
.da-tag::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--faint); flex: none; }
.da-tag.accent::before { background: var(--mark); }
.da-tag.good::before { background: var(--good); }
.da-tag.warn::before { background: var(--warn); }
.da-tag.bad::before { background: var(--bad); }
.da-tag.info::before { background: var(--mark); opacity: .55; }
.da-inj { font-size: 10px; font-weight: 700; color: var(--bad-fg); letter-spacing: .03em;
  background: color-mix(in srgb, var(--bad) 14%, transparent); padding: 1px 5px;
  border-radius: 4px; }
.da-logo { width: 16px; height: 16px; vertical-align: -3px; }
.da-shot { width: 38px; height: 38px; border-radius: 50%; object-fit: cover;
  background: var(--s3); flex: 0 0 38px; border: 1px solid var(--line2); }
.da-shot.sm { width: 30px; height: 30px; flex-basis: 30px; }
.da-shot.dst { object-fit: contain; padding: 5px; }

/* ---------- stat tiles */
.da-tiles { display: flex; flex-wrap: wrap; gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin: 14px 0 6px; }
.da-tile { background: var(--s1); padding: 14px 16px 13px; flex: 1 1 170px; }
.da-tile .k { font-size: 12px; font-weight: 500; color: var(--muted); }
.da-tile .v { font-weight: 650; font-size: 27px; line-height: 1.15; margin-top: 3px;
  color: var(--text); letter-spacing: -.015em; }
.da-tile .v small { font-size: 13px; font-weight: 500; color: var(--faint); margin-left: 4px;
  letter-spacing: 0; }
.da-tile .d { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.da-tile.lead { box-shadow: inset 0 3px 0 var(--mark); }

/* ---------- meters and bars */
.da-meter { height: 8px; background: var(--track); border-radius: 999px; overflow: hidden;
  min-width: 48px; }
.da-meter i { display: block; height: 100%; background: var(--mark); border-radius: 999px; }
.da-meter.thin { height: 4px; }
.da-meter.neutral { background: var(--s3); }
.da-meter.neutral i { background: var(--faint); }
.da-odds { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.da-odds .da-meter { flex: 0 0 72px; }
.da-odds b { font-weight: 600; min-width: 42px; text-align: right;
  font-variant-numeric: tabular-nums; }
.da-split { display: flex; gap: 2px; height: 8px; }
.da-split i { display: block; height: 100%; background: var(--mark); border-radius: 999px 0 0 999px;
  min-width: 2px; }
.da-split u { display: block; height: 100%; flex: 1; background: var(--s3);
  border-radius: 0 999px 999px 0; min-width: 2px; }
.da-seeds { display: flex; gap: 2px; align-items: flex-end; height: 26px; }
.da-seeds i { flex: 1; min-width: 5px; background: var(--mark); border-radius: 2px 2px 0 0;
  min-height: 1px; }
.da-seeds i.out { background: var(--faint); opacity: .55; }

/* ---------- tables (hand-built, not dataframes) */
.da-table { width: 100%; border-collapse: collapse; font-size: 13.5px; margin: 0; border: 0; }
.da-table th, .da-table td { border: 0; background: none; }
.da-table tr { background: none !important; border: 0; }
.da-table th { font-size: 11.5px; font-weight: 600; letter-spacing: .03em; color: var(--muted);
  text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.da-table td { padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
.da-table tr:last-child td { border-bottom: 0; }
.da-table tbody tr:hover td { background: color-mix(in srgb, var(--s2) 60%, transparent); }
.da-table .r { text-align: right; }
.da-table .c { text-align: center; }
.da-table td.r, .da-table td.c { font-variant-numeric: tabular-nums; }
.da-table tr.me td { background: var(--accent-dim); }
.da-table tr.me td:first-child { box-shadow: inset 3px 0 0 var(--mark); }
.da-table tr.cut td { border-bottom: 2px solid var(--line2); }
.da-table tr.group td { font-size: 11.5px; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; color: var(--faint); padding: 16px 10px 6px;
  background: none !important; border-bottom: 1px solid var(--line); }
.da-wrap { background: var(--s1); border: 1px solid var(--line); border-radius: 12px;
  overflow-x: auto; }
.da-who { display: flex; align-items: center; gap: 10px; min-width: 0; }
.da-who .nm { font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.da-who .mt { font-size: 12px; color: var(--muted); display: flex; gap: 6px; align-items: center;
  white-space: nowrap; }
.da-slot { font-size: 11px; font-weight: 600; color: var(--muted); }
.da-strong { color: var(--text); font-weight: 600; }
.da-dim { color: var(--faint); }
.da-up::before { content: "▲ "; font-size: 9px; color: var(--good); }
.da-down::before { content: "▼ "; font-size: 9px; color: var(--bad); }
td.heat { text-align: center; font-variant-numeric: tabular-nums; font-size: 12.5px; }

/* ---------- cards */
.da-card { background: var(--s1); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 18px; margin-bottom: 10px; }
.da-card.lead { box-shadow: inset 3px 0 0 var(--mark); }
.da-wv { display: grid; grid-template-columns: 26px 1fr auto; gap: 14px; align-items: start; }
.da-rank { font-size: 20px; font-weight: 650; color: var(--faint); line-height: 1;
  padding-top: 9px; font-variant-numeric: tabular-nums; }
.da-wv .top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.da-bid { text-align: right; min-width: 92px; }
.da-bid .amt { font-size: 30px; font-weight: 700; line-height: 1; color: var(--text);
  letter-spacing: -.02em; }
.da-bid .rng { font-size: 11.5px; color: var(--faint); margin-top: 5px; }
.da-bid .lbl { font-size: 11px; font-weight: 600; letter-spacing: .05em; color: var(--muted);
  text-transform: uppercase; margin-bottom: 2px; }
.da-stats { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 10px 0 0; font-size: 12px;
  color: var(--muted); }
.da-stats b { font-weight: 600; color: var(--text); margin-left: 4px;
  font-variant-numeric: tabular-nums; }
.da-swap { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px;
  padding: 9px 11px; background: var(--s2); border-radius: 8px; font-size: 13px; }
.da-swap .lab { font-size: 10px; font-weight: 700; letter-spacing: .08em; padding: 2px 6px;
  border-radius: 4px; }
.da-swap .add { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-fg); }
.da-swap .drop { background: color-mix(in srgb, var(--bad) 14%, transparent); color: var(--bad-fg); }
.da-swap .arrow { color: var(--faint); }
.da-why { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-direction: column;
  gap: 4px; }
.da-why li { font-size: 12.5px; color: var(--muted); padding-left: 14px; position: relative;
  margin: 0; }
.da-why li::before { content: ""; position: absolute; left: 2px; top: 8px; width: 5px;
  height: 1px; background: var(--faint); }
.da-why b { color: var(--text); font-weight: 600; }

/* ---------- alerts: a status edge plus a label, never color alone */
.da-alert { display: flex; gap: 12px; align-items: center; padding: 11px 14px; border-radius: 10px;
  background: var(--s1); border: 1px solid var(--line); border-left: 3px solid var(--warn);
  margin-bottom: 8px; font-size: 13.5px; flex-wrap: wrap; }
.da-alert.bad { border-left-color: var(--bad); }
.da-alert.good { border-left-color: var(--good); }
.da-alert.info { border-left-color: var(--mark); }
.da-alert .k { font-size: 10.5px; letter-spacing: .07em; font-weight: 700;
  text-transform: uppercase; color: var(--warn-fg); min-width: 64px; }
.da-alert.bad .k { color: var(--bad-fg); }
.da-alert.good .k { color: var(--good-fg); }
.da-alert.info .k { color: var(--accent-fg); }
.da-alert .x { color: var(--muted); font-size: 12.5px; margin-left: auto;
  font-variant-numeric: tabular-nums; }

/* ---------- matchup */
.da-mu { background: var(--s1); border: 1px solid var(--line); border-radius: 12px;
  overflow: hidden; }
.da-mu .rowm { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
  padding: 18px 20px 14px; gap: 12px; }
.da-mu .side .who { font-size: 11.5px; font-weight: 600; letter-spacing: .05em;
  text-transform: uppercase; color: var(--muted); }
.da-mu .side .tm { font-weight: 600; font-size: 15px; margin-top: 2px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.da-mu .side .pts { font-size: 48px; font-weight: 700; line-height: 1.05; margin-top: 4px;
  letter-spacing: -.03em; }
.da-mu .side .pj { font-size: 12px; color: var(--muted); margin-top: 4px; }
.da-mu .side.r { text-align: right; }
.da-mu .vs { font-size: 11.5px; font-weight: 600; color: var(--faint); letter-spacing: .06em; }
.da-mu .wp { padding: 0 20px; }
.da-mu .wpl { display: flex; justify-content: space-between; padding: 8px 20px 14px;
  font-size: 12px; color: var(--muted); }
.da-mu .wpl b { color: var(--text); font-weight: 600; }

.da-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.da-sb { background: var(--s1); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px 14px; }
.da-sb.me { box-shadow: inset 3px 0 0 var(--mark); }
.da-sb .ln { display: flex; justify-content: space-between; align-items: center; padding: 5px 0;
  font-size: 13.5px; gap: 10px; }
.da-sb .ln span:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.da-sb .ln .p { font-variant-numeric: tabular-nums; }
.da-sb .ln.w span { color: var(--text); font-weight: 600; }
.da-sb .ln.l span { color: var(--muted); }
.da-sb .ft { font-size: 11px; font-weight: 600; letter-spacing: .05em; color: var(--faint);
  text-transform: uppercase; border-top: 1px solid var(--line); padding-top: 6px;
  margin-top: 4px; }

/* ---------- trade card */
.da-tr { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.da-tr .col { background: var(--s2); border-radius: 8px; padding: 10px 12px; }
.da-tr .col .h { font-size: 10.5px; letter-spacing: .08em; font-weight: 700;
  margin-bottom: 8px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.da-tr .col .h::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
.da-tr .get .h::before { background: var(--good); }
.da-tr .give .h::before { background: var(--bad); }
.da-tr .pl { display: flex; align-items: center; gap: 9px; padding: 4px 0; }
.da-fair { margin-top: 12px; }
.da-fair .track { position: relative; height: 6px; border-radius: 3px; background: var(--s3); }
.da-fair .track::before { content: ""; position: absolute; left: 50%; top: -3px; width: 1px;
  height: 12px; background: var(--line2); }
.da-fair .track i { position: absolute; top: -4px; width: 4px; height: 14px;
  background: var(--text); border-radius: 2px; }
.da-fair .lg { display: flex; justify-content: space-between; font-size: 11px;
  color: var(--faint); margin-top: 6px; }
.da-delta { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 12px; font-size: 12.5px;
  color: var(--muted); }
.da-delta b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }

/* ---------- misc */
.da-empty { border: 1px dashed var(--line2); border-radius: 12px; padding: 22px;
  color: var(--muted); font-size: 14px; background: var(--s1); }
.da-empty b { color: var(--text); }
.da-foot { font-size: 12px; color: var(--faint); margin-top: 18px; line-height: 1.6; }
.da-news a { color: var(--text) !important; text-decoration: none; font-weight: 500; }
.da-news a:hover { color: var(--accent-fg) !important; }
.da-news div { padding: 10px 0; border-bottom: 1px solid var(--line); }
.da-news span { display: block; font-size: 11.5px; color: var(--faint); margin-top: 3px; }
.da-swatch { display: flex; gap: 0; height: 34px; border-radius: 8px; overflow: hidden;
  border: 1px solid var(--line2); }
.da-swatch i { flex: 1; }
.da-m { display: none; }
.show-m { display: none; }

/* ---------- phones */
@media (max-width: 700px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 3.6rem 1rem 3rem !important; }
  div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.35rem !important; }
  div[data-testid="stColumn"] { min-width: 0 !important; }
  div[data-testid="stColumn"] button { padding: 0.3rem 0.45rem !important; }
  .da-title { font-size: 25px; }
  .da-tile { flex-basis: 140px; }
  .da-tile .v { font-size: 22px; }
  /* Waiver card: name and bid share the first row, everything else runs full width. */
  .da-wv { grid-template-columns: 1fr auto; gap: 0 12px; }
  .da-wv > .body { display: contents; }
  .da-wv .top { grid-column: 1; grid-row: 1; }
  .da-wv .da-bid { grid-column: 2; grid-row: 1; }
  .da-wv .da-stats, .da-wv .da-swap, .da-wv .da-why { grid-column: 1 / -1; }
  .da-rank { display: none; }
  .da-table .da-who .nm { max-width: 36vw; }
  .da-bid .amt { font-size: 25px; }
  .da-tr { grid-template-columns: 1fr; }
  .da-mu .side .pts { font-size: 36px; }
  .da-mu .rowm { padding: 14px 14px 10px; }
  .da-mu .wp { padding: 0 14px; }
  .da-mu .wpl { padding: 8px 14px 12px; }
  .hide-m { display: none !important; }
  .show-m { display: inline !important; }
  .da-section { flex-direction: column; align-items: flex-start; gap: 3px; }
  .da-section span { text-align: left; }
  .da-table { font-size: 13px; }
  .da-table td, .da-table th { padding: 8px 5px; }
  .da-table td:first-child, .da-table th:first-child { padding-left: 10px; }
  .da-table td:last-child, .da-table th:last-child { padding-right: 10px; }
  .da-who { gap: 8px; }
  .da-shot.sm { width: 26px; height: 26px; flex-basis: 26px; }
  .da-odds .da-meter { flex-basis: 40px; }
  .da-d { display: none !important; }
  .da-m { display: flex !important; }
  /* Draft-room rows: fixed-width action columns, text column takes the rest. */
  div[data-testid="stHorizontalBlock"]:has(.da-m) > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"]:has(.da-sg) > div[data-testid="stColumn"] {
    flex: 0 0 46px !important; min-width: 46px !important; }
  div[data-testid="stHorizontalBlock"]:has(.da-m) > div[data-testid="stColumn"]:first-child,
  div[data-testid="stHorizontalBlock"]:has(.da-sg) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important; min-width: 0 !important; }
}
</style>
"""


# ------------------------------------------------------------------ theme plumbing

def current() -> Theme:
    return THEMES.get(st.session_state.get("theme"), THEMES[DEFAULT_THEME])


def _apply_native(theme: Theme) -> bool:
    """Point Streamlit's own theme at this theme's tokens. Returns True when
    something changed (the page then needs one rerun to pick it up)."""
    try:
        from streamlit import config
        changed = False
        for key, value in theme.streamlit_options().items():
            if str(config.get_option(key) or "").lower() != value.lower():
                config.set_option(key, value)
                changed = True
        return changed
    except Exception:
        return False  # runtime theming unavailable: config.toml's default stands


def setup(saved: str | None = None):
    """Resolve the active theme, sync Streamlit's native theme to it, and
    inject the CSS. Call once per run, right after set_page_config."""
    if st.session_state.get("theme") not in THEMES:
        st.session_state["theme"] = saved if saved in THEMES else DEFAULT_THEME
    theme = current()
    if st.session_state.get("_theme_synced") != theme.key:
        st.session_state["_theme_synced"] = theme.key
        if _apply_native(theme):
            st.rerun()
    css = CSS.replace("__FONT__", FONT).replace(
        "__TOKENS__", " ".join(f"--{k}: {v};" for k, v in tokens(theme).items()))
    st.markdown(css, unsafe_allow_html=True)


def choose(key: str):
    """Switch theme for this session (the caller persists the preference)."""
    if key in THEMES and key != st.session_state.get("theme"):
        st.session_state["theme"] = key
        st.session_state.pop("_theme_synced", None)


def logo_svg(theme: Theme, mark_only: bool = False) -> str:
    """Wordmark with three rising bars, drawn in the theme's ink and accent."""
    bars = (f"<rect x='2' y='17' width='6' height='9' rx='1.5' fill='{theme.mark}' opacity='.5'/>"
            f"<rect x='10.5' y='11' width='6' height='15' rx='1.5' fill='{theme.mark}' opacity='.75'/>"
            f"<rect x='19' y='5' width='6' height='21' rx='1.5' fill='{theme.mark}'/>")
    if mark_only:
        return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 27 31' fill='none'>"
                f"{bars}</svg>")
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 142 31' fill='none'>{bars}"
            f"<text x='34' y='23' font-family=\"Inter, 'Segoe UI', system-ui, sans-serif\" "
            f"font-weight='700' font-size='20' letter-spacing='-0.4' fill='{theme.text}'>"
            f"DraftAid</text></svg>")


# ------------------------------------------------------------------ formatting

def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(v) -> str:
    return _html.escape(str(v)) if v is not None else ""


def isnum(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and pd.notna(v)


def fmt_int(v, dash: str = "–") -> str:
    return str(int(v)) if isnum(v) else dash


def fmt_pts(v, dash: str = "–") -> str:
    return f"{float(v):.1f}" if isnum(v) else dash


def fmt_pct(p, dash: str = "–") -> str:
    """Probability as a percent; the tails read as <1% / >99%, never 0 or 100
    unless it is truly settled."""
    if not isnum(p):
        return dash
    if p <= 0:
        return "0%"
    if p >= 1:
        return "100%"
    if p < 0.01:
        return "<1%"
    if p > 0.99:
        return ">99%"
    return f"{p * 100:.0f}%"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ------------------------------------------------------------------ atoms

def pos_chip(pos: str) -> str:
    cls = {"RB": " rb", "WR": " wr", "TE": " te"}.get(pos, "")
    return f"<span class='da-chip{cls}'><i></i>{esc(pos)}</span>"


def team_logo(team: str) -> str:
    if not team or team == "FA":
        return ""
    return (f"<img class='da-logo' alt='' src='https://sleepercdn.com/images/team_logos/"
            f"nfl/{esc(str(team).lower())}.png'>")


def headshot(p, small: bool = False) -> str:
    size = " sm" if small else ""
    if p.get("pos") == "DST":
        return (f"<img class='da-shot dst{size}' alt='' src='https://sleepercdn.com/images/"
                f"team_logos/nfl/{esc(str(p.get('team', '')).lower())}.png'>")
    sid = p.get("sleeper_id")
    if isinstance(sid, str) and sid:
        return (f"<img class='da-shot{size}' alt='' loading='lazy' src='https://sleepercdn.com/"
                f"content/nfl/players/thumb/{esc(sid)}.jpg'>")
    return f"<span class='da-shot{size}'></span>"


def injury_tag(inj) -> str:
    if not inj:
        return ""
    short = {"Questionable": "Q", "Doubtful": "D", "Out": "OUT", "Injury Reserve": "IR",
             "Ir": "IR", "Pup": "PUP", "Suspension": "SUS", "Day To Day": "DTD"}
    label = short.get(str(inj).title(), str(inj).upper()[:4])
    return f"<span class='da-inj' title='{esc(inj)}'>{esc(label)}</span>"


def tag(text: str, tone: str = "") -> str:
    """Pill label. tone: accent | good | warn | bad | info | '' (neutral)."""
    return f"<span class='da-tag {tone}'>{esc(text)}</span>"


def who(p, small: bool = False, meta: str | None = None) -> str:
    """Headshot + name + (position chip, team logo, team, injury) block."""
    if meta is None:
        meta = (f"{pos_chip(p.get('pos', ''))}{team_logo(p.get('team'))}"
                f"<span>{esc(p.get('team', ''))}</span>{injury_tag(p.get('injury'))}")
    return (f"<div class='da-who'>{headshot(p, small)}<div style='min-width:0'>"
            f"<div class='nm'>{esc(p.get('player', ''))}</div>"
            f"<div class='mt'>{meta}</div></div></div>")


def meter(frac, cls: str = "") -> str:
    """A ratio against its whole: accent fill on a same-hue track."""
    pct = max(0.0, min(1.0, float(frac) if isnum(frac) else 0.0)) * 100
    return f"<div class='da-meter {cls}'><i style='width:{pct:.1f}%'></i></div>"


def odds(p) -> str:
    """Percent figure with its meter, for a column of probabilities."""
    return (f"<div class='da-odds' title='{(p or 0) * 100:.1f}%'>{meter(p)}"
            f"<b>{fmt_pct(p)}</b></div>")


def split_bar(frac) -> str:
    """Two-part bar (mine | theirs) with a surface gap between the parts."""
    pct = max(0.0, min(1.0, float(frac))) * 100
    return f"<div class='da-split'><i style='width:{pct:.1f}%'></i><u></u></div>"


def seed_bars(dist: list[float], cut: int) -> str:
    """Small column chart of finishing-position odds; seeds past the playoff
    cut are drawn in neutral."""
    top = max(dist) if dist else 0
    if top <= 0:
        return ""
    bars = "".join(
        f"<i class='{'out' if i >= cut else ''}' title='{ordinal(i + 1)}: {p * 100:.0f}%' "
        f"style='height:{max(2.0, p / top * 100):.0f}%'></i>" for i, p in enumerate(dist))
    return f"<div class='da-seeds'>{bars}</div>"


def heat(value: str, diff: float, cls: str = "") -> str:
    """Table cell shaded on a diverging scale. diff in [-1, 1]: positive is
    favorable (blue), negative unfavorable (red), zero neutral. The value is
    always printed, so the shade is never the only way to read it."""
    alpha = min(0.34, abs(diff) * 0.34)   # a tint, never a saturated block
    pole = "var(--pos)" if diff > 0 else "var(--neg)"
    return (f"<td class='heat {cls}' style='background:color-mix(in srgb, {pole} "
            f"{alpha * 100:.0f}%, transparent)'>{value}</td>")


def delta(v, unit: str = "", digits: int = 1) -> str:
    """Signed change: a +/− sign plus an up/down arrow, so direction never
    depends on color."""
    if not isnum(v) or abs(v) < 10 ** -digits / 2:
        return f"<span class='da-dim'>no change</span>"
    cls, sign = ("da-up", "+") if v > 0 else ("da-down", "−")
    return f"<span class='{cls}'>{sign}{abs(v):.{digits}f}{unit}</span>"


# ------------------------------------------------------------------ blocks

def page_header(title: str, eyebrow: list[str] | None = None, sub: str | None = None,
                live: str | None = None):
    bits = [f"<span class='live'>{esc(live)}</span>"] if live else []
    bits += [f"<span>{esc(b)}</span>" for b in (eyebrow or []) if b]
    html(f"<div class='da-eyebrow'>{'<span class=sep>/</span>'.join(bits)}</div>"
         f"<div class='da-title'>{esc(title)}</div>"
         + (f"<div class='da-sub'>{sub}</div>" if sub else ""))


def section(title: str, note: str = ""):
    html(f"<div class='da-section'><div class='h'>{esc(title)}</div><span>{note}</span></div>")


def tiles(items: list[dict]):
    """KPI row. items: {k: label, v: value html, d: detail, lead: bool, bar: 0..1}"""
    out = []
    for t in items:
        bar = (f"<div style='margin-top:9px'>{meter(t['bar'], 'thin')}</div>"
               if t.get("bar") is not None else "")
        out.append(f"<div class='da-tile{' lead' if t.get('lead') else ''}'>"
                   f"<div class='k'>{esc(t['k'])}</div><div class='v'>{t['v']}</div>"
                   + (f"<div class='d'>{t['d']}</div>" if t.get("d") else "") + f"{bar}</div>")
    html(f"<div class='da-tiles'>{''.join(out)}</div>")


def alert(kind: str, label: str, body: str, extra: str = ""):
    """kind: good | bad | info | '' (warning)."""
    html(f"<div class='da-alert {kind}'><span class='k'>{esc(label)}</span>"
         f"<span>{body}</span>" + (f"<span class='x'>{extra}</span>" if extra else "") + "</div>")


def empty(body: str):
    html(f"<div class='da-empty'>{body}</div>")


def _th(label) -> str:
    """Header text; a (long, short) pair shows the short form on phones."""
    if isinstance(label, tuple):
        return (f"<span class='hide-m'>{esc(label[0])}</span>"
                f"<span class='show-m'>{esc(label[1])}</span>")
    return esc(label)


def table(head: list[tuple], rows: list[str], cls: str = "") -> str:
    """head: (label, th class), where label is text or a (long, short) pair.
    rows: complete <tr> strings."""
    ths = "".join(f"<th class='{c}'>{_th(label)}</th>" for label, c in head)
    return (f"<div class='da-wrap'><table class='da-table {cls}'><thead><tr>{ths}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def roster_table(df: pd.DataFrame, playoffs: bool = False) -> str:
    """Starters / bench / IR grouped roster table. With `playoffs`, adds each
    player's positional rank by projection over the fantasy-playoff weeks."""
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "RB/WR": 4, "FLEX": 5, "OP": 6,
             "DST": 7, "K": 8, "BN": 9, "IR": 10}
    df = df.copy()
    df["_o"] = df["slot"].map(order).fillna(99)
    df = df.sort_values(["_o", "ros_rank"], na_position="last")
    head = [("Slot", ""), ("Player", ""), ("Proj", "r"), ("This week", "r hide-m"),
            (("Rest of season", "Season"), "r")]
    if playoffs:
        head.append(("Playoffs", "r"))
    head.append(("Bye", "c hide-m"))
    groups = [("Starters", df[df["starter"] & (df["slot"] != "IR")]),
              ("Bench", df[df["slot"] == "BN"]), ("Injured reserve", df[df["slot"] == "IR"])]
    rows = []
    for label, g in groups:
        if g.empty:
            continue
        rows.append(f"<tr class='group'><td colspan='{len(head)}'>{label}</td></tr>")
        for _, p in g.iterrows():
            wk = f"{p['pos']}{int(p['weekly_rank'])}" if isnum(p.get("weekly_rank")) else "–"
            po = (f"<td class='r'>{p['pos']}{int(p['po_rank'])}</td>" if isnum(p.get("po_rank"))
                  else "<td class='r da-dim'>–</td>") if playoffs else ""
            bye = fmt_int(p.get("bye")) if isnum(p.get("bye")) and p.get("bye") else "–"
            rows.append(
                f"<tr><td><span class='da-slot'>{esc(p['slot'])}</span></td>"
                f"<td>{who(p, small=True)}</td>"
                f"<td class='r'><span class='da-strong'>{fmt_pts(p.get('week_proj'))}</span></td>"
                f"<td class='r hide-m'>{wk}</td>"
                f"<td class='r'>{esc(p.get('ros_pos_rank') or '–')}</td>{po}"
                f"<td class='c hide-m da-dim'>{bye}</td></tr>")
    return table(head, rows)


def news_list(items: list[dict]) -> str:
    out = []
    for n in items:
        extras = " · ".join(x for x in (n.get("source"), n.get("when")) if x)
        out.append(f"<div><a href='{esc(n['link'])}' target='_blank'>{esc(n['title'])}</a>"
                   f"<span>{esc(extras)}</span></div>")
    return f"<div class='da-news'>{''.join(out)}</div>"
