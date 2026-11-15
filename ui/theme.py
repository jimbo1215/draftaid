"""DraftAid design system: global CSS plus small HTML building blocks.

Every season page renders its data through these helpers so the look stays
consistent -- dark surfaces, one volt accent, condensed display type, and
tabular numbers. Components return HTML strings; `html()` puts them on the
page. Keep markup on one line per element: Markdown treats indented or
blank-line-separated HTML as code blocks.
"""

import html as _html
import math

import pandas as pd
import streamlit as st

POS_COLORS = {"QB": "#FF6B8B", "RB": "#2FD6A3", "WR": "#5AB0FF", "TE": "#FFA94D",
              "K": "#B69CFF", "DST": "#A3ADBA", "FLEX": "#C5F82A", "OP": "#C5F82A"}

CSS = """
<style>
:root {
  --bg: #0A0C0F; --s1: #111419; --s2: #171B21; --s3: #1E232B;
  --line: #222831; --line2: #2C333D;
  --text: #E6E9EE; --muted: #8B95A3; --faint: #5B6573;
  --volt: #C5F82A; --volt-dim: rgba(197,248,42,.12);
  --good: #2FD6A3; --bad: #FF5D5D; --warn: #FFC34D; --info: #5AB0FF;
  --display: 'Barlow Condensed', 'IBM Plex Sans', sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}
/* ---------- chrome */
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.2rem !important; max-width: 1180px; }
[data-testid="stSidebar"] { border-right: 1px solid #1C2129; }
[data-testid="stSidebarNav"] a { border-radius: 6px; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--volt-dim); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--volt) !important; }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line); }
[data-testid="stSidebarNav"] header, [data-testid="stNavSectionHeader"] {
  font-family: var(--mono); font-size: 10.5px !important; letter-spacing: .14em;
  text-transform: uppercase; color: var(--faint) !important; }
h1, h2, h3, h4 { letter-spacing: .01em; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }
div[data-testid="stColumn"] button { min-width: 40px; }
button[kind="primary"] { color: #0A0C0F !important; font-weight: 600; }
[data-testid="stExpander"] details { border-color: var(--line) !important; background: var(--s1); }
[data-testid="stExpander"] summary p { font-size: 13px; color: var(--muted); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---------- type + primitives */
.da-num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.da-eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); display: flex; gap: 10px;
  align-items: center; flex-wrap: wrap; }
.da-eyebrow .live { color: var(--volt); display: inline-flex; align-items: center; gap: 6px; }
.da-eyebrow .live::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--volt); box-shadow: 0 0 0 3px var(--volt-dim); }
.da-title { font-family: var(--display); font-weight: 700; font-size: 40px;
  line-height: 1.02; text-transform: uppercase; letter-spacing: .01em;
  color: var(--text); margin: 6px 0 2px; }
.da-sub { color: var(--muted); font-size: 14px; margin-bottom: 4px; }
.da-section { display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; margin: 30px 0 10px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.da-section .h { font-family: var(--display); font-size: 19px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .04em; margin: 0; padding: 0; color: var(--text);
  line-height: 1.2; }
.da-section span { font-size: 12px; color: var(--faint); }

.da-chip { display: inline-flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .04em;
  padding: 2px 6px; border-radius: 4px; min-width: 30px; line-height: 1.35;
  border: 1px solid currentColor; background: color-mix(in srgb, currentColor 12%, transparent); }
.da-tag { display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
  font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 3px 8px;
  border-radius: 999px; white-space: nowrap; }
.da-tag.volt { background: var(--volt); color: #0A0C0F; }
.da-tag.info { background: rgba(90,176,255,.14); color: var(--info); }
.da-tag.good { background: rgba(47,214,163,.14); color: var(--good); }
.da-tag.bad { background: rgba(255,93,93,.14); color: var(--bad); }
.da-tag.warn { background: rgba(255,195,77,.14); color: var(--warn); }
.da-tag.dim { background: var(--s3); color: var(--muted); }
.da-inj { font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--bad);
  background: rgba(255,93,93,.12); padding: 1px 5px; border-radius: 3px; letter-spacing: .04em; }
.da-logo { width: 16px; height: 16px; vertical-align: -3px; }
.da-shot { width: 38px; height: 38px; border-radius: 50%; object-fit: cover;
  background: var(--s3); flex: 0 0 38px; border: 1px solid var(--line2); }
.da-shot.sm { width: 30px; height: 30px; flex-basis: 30px; }
.da-shot.dst { object-fit: contain; padding: 5px; }

/* ---------- stat tiles */
.da-tiles { display: flex; flex-wrap: wrap;
  gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden; margin: 14px 0 6px; }
.da-tile { background: var(--s1); padding: 14px 16px 13px; flex: 1 1 170px; }
.da-tile .k { font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); }
.da-tile .v { font-family: var(--display); font-weight: 700; font-size: 30px;
  line-height: 1.1; margin-top: 4px; color: var(--text); }
.da-tile .v small { font-family: var(--mono); font-size: 13px; font-weight: 500;
  color: var(--faint); margin-left: 3px; }
.da-tile .d { font-size: 12px; color: var(--muted); margin-top: 2px; }
.da-tile.accent .v { color: var(--volt); }
.da-bar { height: 4px; background: var(--s3); border-radius: 2px; margin-top: 8px; overflow: hidden; }
.da-bar i { display: block; height: 100%; background: var(--volt); border-radius: 2px; }

/* ---------- tables (hand-built, not dataframes) */
.da-table { width: 100%; border-collapse: collapse; font-size: 13.5px; margin: 0; border: 0; }
.da-table th, .da-table td { border: 0; background: none; }
.da-table tr { background: none !important; border: 0; }
.da-table th { font-family: var(--mono); font-size: 10.5px; font-weight: 500;
  letter-spacing: .1em; text-transform: uppercase; color: var(--faint);
  text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.da-table td { padding: 9px 10px; border-bottom: 1px solid #191D24; vertical-align: middle; }
.da-table tr:last-child td { border-bottom: 0; }
.da-table tbody tr:hover td { background: #12161B; }
.da-table .r { text-align: right; }
.da-table .c { text-align: center; }
.da-table td.r, .da-table td.c { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.da-table tr.me td { background: rgba(197,248,42,.05); }
.da-table tr.cut td { border-bottom: 1px dashed #3A424D; }
.da-table tr.me td:first-child { box-shadow: inset 3px 0 0 var(--volt); }
.da-table tr.group td { font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--faint); padding: 16px 10px 6px; background: none !important;
  border-bottom: 1px solid var(--line); }
.da-wrap { background: var(--s1); border: 1px solid var(--line); border-radius: 10px;
  overflow-x: auto; }
.da-who { display: flex; align-items: center; gap: 10px; min-width: 0; }
.da-who .nm { font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.da-who .mt { font-size: 12px; color: var(--muted); display: flex; gap: 6px; align-items: center;
  white-space: nowrap; }
.da-slot { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.da-strong { color: var(--text); font-weight: 600; }
.da-dim { color: var(--faint); }

/* ---------- cards */
.da-card { background: var(--s1); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 18px; margin-bottom: 10px; }
.da-card.hot { border-color: rgba(197,248,42,.35);
  background: linear-gradient(180deg, rgba(197,248,42,.045), transparent 60%), var(--s1); }
.da-wv { display: grid; grid-template-columns: 28px 1fr auto; gap: 14px; align-items: start; }
.da-rank { font-family: var(--display); font-size: 26px; font-weight: 700; color: var(--faint);
  line-height: 1; padding-top: 6px; }
.da-card.hot .da-rank { color: var(--volt); }
.da-wv .top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.da-wv .nm { font-size: 16px; font-weight: 600; }
.da-bid { text-align: right; min-width: 92px; }
.da-bid .amt { font-family: var(--display); font-size: 34px; font-weight: 700; line-height: 1;
  color: var(--text); }
.da-card.hot .da-bid .amt { color: var(--volt); }
.da-bid .rng { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-top: 4px; }
.da-bid .lbl { font-family: var(--mono); font-size: 10px; letter-spacing: .12em; color: var(--faint);
  text-transform: uppercase; }
.da-stats { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 10px 0 0; font-size: 12px;
  color: var(--muted); }
.da-stats b { font-family: var(--mono); font-weight: 500; color: var(--text); margin-left: 4px; }
.da-swap { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px;
  padding: 9px 11px; background: var(--s2); border-radius: 8px; font-size: 13px; }
.da-swap .lab { font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: .12em;
  padding: 2px 6px; border-radius: 3px; }
.da-swap .add { background: rgba(47,214,163,.14); color: var(--good); }
.da-swap .drop { background: rgba(255,93,93,.12); color: var(--bad); }
.da-swap .arrow { color: var(--faint); }
.da-why { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-direction: column; gap: 4px; }
.da-why li { font-size: 12.5px; color: var(--muted); padding-left: 14px; position: relative; margin: 0; }
.da-why li::before { content: ""; position: absolute; left: 2px; top: 8px; width: 5px; height: 1px;
  background: var(--faint); }

/* ---------- alerts */
.da-alert { display: flex; gap: 12px; align-items: center; padding: 11px 14px; border-radius: 8px;
  background: var(--s1); border: 1px solid var(--line); border-left: 3px solid var(--warn);
  margin-bottom: 8px; font-size: 13.5px; }
.da-alert.bad { border-left-color: var(--bad); }
.da-alert.good { border-left-color: var(--good); }
.da-alert .k { font-family: var(--mono); font-size: 10px; letter-spacing: .12em; font-weight: 600;
  text-transform: uppercase; color: var(--warn); min-width: 64px; }
.da-alert.bad .k { color: var(--bad); }
.da-alert.good .k { color: var(--good); }
.da-alert .x { color: var(--muted); font-size: 12.5px; margin-left: auto; font-family: var(--mono); }

/* ---------- matchup score bug */
.da-mu { background: var(--s1); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.da-mu .rowm { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
  padding: 18px 20px 14px; gap: 12px; }
.da-mu .side .who { font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--faint); }
.da-mu .side .tm { font-weight: 600; font-size: 15px; margin-top: 2px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.da-mu .side .pts { font-family: var(--display); font-size: 46px; font-weight: 700; line-height: 1;
  margin-top: 6px; }
.da-mu .side .pj { font-family: var(--mono); font-size: 11.5px; color: var(--muted); margin-top: 4px; }
.da-mu .side.r { text-align: right; }
.da-mu .vs { font-family: var(--mono); font-size: 11px; color: var(--faint); letter-spacing: .1em; }
.da-mu .wp { display: flex; height: 6px; background: var(--s3); }
.da-mu .wp i { display: block; height: 100%; background: var(--volt); }
.da-mu .wpl { display: flex; justify-content: space-between; padding: 8px 20px 12px;
  font-family: var(--mono); font-size: 11px; color: var(--muted); }
.da-mu .lead { color: var(--volt); }

.da-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.da-sb { background: var(--s1); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
.da-sb.me { border-color: rgba(197,248,42,.4); }
.da-sb .ln { display: flex; justify-content: space-between; align-items: center; padding: 5px 0;
  font-size: 13.5px; gap: 10px; }
.da-sb .ln span:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.da-sb .ln .p { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.da-sb .ln.w span { color: var(--text); font-weight: 600; }
.da-sb .ln.l span { color: var(--muted); }
.da-sb .ft { font-family: var(--mono); font-size: 10px; letter-spacing: .12em; color: var(--faint);
  text-transform: uppercase; border-top: 1px solid var(--line); padding-top: 6px; margin-top: 4px; }

/* ---------- trade card */
.da-tr { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.da-tr .col { background: var(--s2); border-radius: 8px; padding: 10px 12px; }
.da-tr .col .h { font-family: var(--mono); font-size: 10px; letter-spacing: .14em; font-weight: 600;
  margin-bottom: 8px; }
.da-tr .get .h { color: var(--good); }
.da-tr .give .h { color: var(--bad); }
.da-tr .pl { display: flex; align-items: center; gap: 9px; padding: 4px 0; }
.da-meter { margin-top: 12px; }
.da-meter .track { position: relative; height: 6px; border-radius: 3px;
  background: linear-gradient(90deg, rgba(47,214,163,.5), var(--s3) 40%, var(--s3) 60%, rgba(255,93,93,.5)); }
.da-meter .track i { position: absolute; top: -4px; width: 3px; height: 14px; background: var(--text);
  border-radius: 2px; }
.da-meter .lg { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10px;
  color: var(--faint); margin-top: 6px; letter-spacing: .06em; }

/* ---------- heatmap */
.da-heat td.hc { text-align: center; font-family: var(--mono); font-size: 12.5px; }

/* ---------- misc */
.da-empty { border: 1px dashed var(--line2); border-radius: 12px; padding: 22px; color: var(--muted);
  font-size: 14px; background: var(--s1); }
.da-empty b { color: var(--text); }
.da-foot { font-size: 12px; color: var(--faint); margin-top: 18px; line-height: 1.6; }
.da-news a { color: var(--text) !important; text-decoration: none; font-weight: 500; }
.da-news a:hover { color: var(--volt) !important; }
.da-news div { padding: 10px 0; border-bottom: 1px solid var(--line); }
.da-news span { display: block; font-family: var(--mono); font-size: 11px; color: var(--faint); margin-top: 3px; }
.da-m { display: none; }

/* ---------- phones */
@media (max-width: 700px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 3.6rem 1rem 3rem !important; }
  div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.35rem !important; }
  div[data-testid="stColumn"] { min-width: 0 !important; }
  div[data-testid="stColumn"] button { padding: 0.3rem 0.45rem !important; }
  .da-title { font-size: 32px; }
  .da-tile { flex-basis: 140px; }
  .da-tile .v { font-size: 24px; }
  .da-wv { grid-template-columns: 1fr auto; }
  .da-rank { display: none; }
  .da-bid .amt { font-size: 28px; }
  .da-tr { grid-template-columns: 1fr; }
  .da-mu .side .pts { font-size: 36px; }
  .da-mu .rowm { padding: 14px 14px 10px; }
  .hide-m { display: none !important; }
  .da-table td, .da-table th { padding: 8px 6px; }
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


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


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


# ------------------------------------------------------------------ atoms

def pos_chip(pos: str) -> str:
    color = POS_COLORS.get(pos, "#8B95A3")
    return f"<span class='da-chip' style='color:{color}'>{esc(pos)}</span>"


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


def tag(text: str, tone: str = "dim") -> str:
    return f"<span class='da-tag {tone}'>{esc(text)}</span>"


def who(p, small: bool = False, meta: str | None = None) -> str:
    """Headshot + name + (pos chip, team logo, team, injury) block."""
    if meta is None:
        meta = (f"{pos_chip(p.get('pos', ''))}{team_logo(p.get('team'))}"
                f"<span>{esc(p.get('team', ''))}</span>{injury_tag(p.get('injury'))}")
    return (f"<div class='da-who'>{headshot(p, small)}<div style='min-width:0'>"
            f"<div class='nm'>{esc(p.get('player', ''))}</div>"
            f"<div class='mt'>{meta}</div></div></div>")


# ------------------------------------------------------------------ blocks

def page_header(title: str, eyebrow: list[str] | None = None, sub: str | None = None,
                live: str | None = None):
    bits = [f"<span class='live'>{esc(live)}</span>"] if live else []
    bits += [f"<span>{esc(b)}</span>" for b in (eyebrow or [])]
    sep = "<span style='color:#2C333D'>/</span>"
    html(f"<div class='da-eyebrow'>{sep.join(bits)}</div>"
         f"<div class='da-title'>{esc(title)}</div>"
         + (f"<div class='da-sub'>{sub}</div>" if sub else ""))


def section(title: str, note: str = ""):
    html(f"<div class='da-section'><div class='h'>{esc(title)}</div><span>{note}</span></div>")


def tiles(items: list[dict]):
    """items: {k: label, v: value html, d: detail, accent: bool, bar: 0..1}"""
    out = []
    for t in items:
        bar = (f"<div class='da-bar'><i style='width:{max(0, min(1, t['bar'])) * 100:.0f}%'></i></div>"
               if t.get("bar") is not None else "")
        out.append(f"<div class='da-tile{' accent' if t.get('accent') else ''}'>"
                   f"<div class='k'>{esc(t['k'])}</div><div class='v'>{t['v']}</div>"
                   + (f"<div class='d'>{t['d']}</div>" if t.get("d") else "") + f"{bar}</div>")
    html(f"<div class='da-tiles'>{''.join(out)}</div>")


def alert(kind: str, label: str, body: str, extra: str = ""):
    html(f"<div class='da-alert {kind}'><span class='k'>{esc(label)}</span>"
         f"<span>{body}</span>" + (f"<span class='x'>{extra}</span>" if extra else "") + "</div>")


def empty(body: str):
    html(f"<div class='da-empty'>{body}</div>")


def roster_table(df: pd.DataFrame, show_proj: bool = True) -> str:
    """Starters / bench / IR grouped roster table."""
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "RB/WR": 4, "FLEX": 5, "OP": 6,
             "DST": 7, "K": 8, "BN": 9, "IR": 10}
    df = df.copy()
    df["_o"] = df["slot"].map(order).fillna(99)
    df = df.sort_values(["_o", "ros_rank"], na_position="last")
    head = ("<tr><th>Slot</th><th>Player</th><th class='r'>Proj</th>"
            "<th class='r hide-m'>Wk rk</th><th class='r'>ROS</th><th class='r hide-m'>Ovr</th>"
            "<th class='c hide-m'>Bye</th></tr>")
    groups = [("Starters", df[df["starter"] & (df["slot"] != "IR")]),
              ("Bench", df[df["slot"] == "BN"]), ("Injured reserve", df[df["slot"] == "IR"])]
    rows = []
    for label, g in groups:
        if g.empty:
            continue
        rows.append(f"<tr class='group'><td colspan='7'>{label}</td></tr>")
        for _, p in g.iterrows():
            wk = (f"{p['pos']}{int(p['weekly_rank'])}" if isnum(p.get("weekly_rank")) else "–")
            proj = fmt_pts(p.get("week_proj"))
            rows.append(
                f"<tr><td><span class='da-slot'>{esc(p['slot'])}</span></td>"
                f"<td>{who(p, small=True)}</td>"
                f"<td class='r'><span class='da-strong'>{proj}</span></td>"
                f"<td class='r hide-m'>{wk}</td>"
                f"<td class='r'>{esc(p.get('ros_pos_rank') or '–')}</td>"
                f"<td class='r hide-m da-dim'>{fmt_int(p.get('ros_rank'))}</td>"
                f"<td class='c hide-m da-dim'>{fmt_int(p.get('bye')) if isnum(p.get('bye')) and p.get('bye') else '–'}</td></tr>")
    return (f"<div class='da-wrap'><table class='da-table'><thead>{head}</thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def news_list(items: list[dict]) -> str:
    out = []
    for n in items:
        extras = " · ".join(x for x in (n.get("source"), n.get("when")) if x)
        out.append(f"<div><a href='{esc(n['link'])}' target='_blank'>{esc(n['title'])}</a>"
                   f"<span>{esc(extras)}</span></div>")
    return f"<div class='da-news'>{''.join(out)}</div>"
