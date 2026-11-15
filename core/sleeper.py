"""Sleeper public API: player database, trending adds, and weekly projections.

The weekly projections are the app's forward-looking engine: one request per
NFL week returns matchup- and bye-aware PPR projections for every player,
including future weeks through the fantasy playoffs.
"""

import statistics

import pandas as pd
import streamlit as st

from core import http
from core.names import normalize_name, normalize_team

API = "https://api.sleeper.app/v1"
PROJ_API = "https://api.sleeper.com/projections/nfl"
_POS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST"}
# A single-game projection above these is a feed glitch (e.g. a season total
# landing in a weekly row), not a forecast.
_SANE_MAX = {"QB": 45.0, "RB": 42.0, "WR": 42.0, "TE": 34.0, "K": 20.0, "DST": 25.0}


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_players(players: dict) -> pd.DataFrame:
    """Raw Sleeper player map -> fantasy-relevant rows keyed for joining.

    Keys are built from the full name with our own normalization (Sleeper's
    search name keeps "III"/"Jr.", ours doesn't). Anyone on an NFL roster is
    kept whatever his status: "Inactive" is how Sleeper marks injured players,
    the ones whose status matters most."""
    rows = []
    for pid, p in players.items():
        pos = p.get("position")
        if pos == "DEF":
            rows.append({"sleeper_id": pid, "key": f"dst{normalize_team(pid).lower()}",
                         "espn_id": None, "nfl_team": normalize_team(pid), "injury": "",
                         "injury_note": "", "age": None, "years_exp": None, "college": "",
                         "depth_order": None})
            continue
        if pos not in ("QB", "RB", "WR", "TE", "K") or not (p.get("team") or p.get("active")):
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}"
        rows.append({
            "sleeper_id": pid,
            "key": f"{normalize_name(name)}|{pos}",
            "espn_id": _int_or_none(p.get("espn_id")),
            "nfl_team": normalize_team(p.get("team")),
            "injury": p.get("injury_status") or "",
            "injury_note": p.get("injury_body_part") or "",
            "age": p.get("age"),
            "years_exp": p.get("years_exp"),
            "college": p.get("college") or "",
            "depth_order": _int_or_none(p.get("depth_chart_order")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Two players can share a name and position; the one on a roster wins.
    df["_signed"] = df["nfl_team"] != ""
    df = df.sort_values("_signed", ascending=False, kind="stable")
    return df.drop_duplicates(subset="key", keep="first").drop(columns="_signed")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_players() -> pd.DataFrame:
    """Fantasy-relevant players with ids, injury status, and depth-chart slot."""
    return parse_players(http.get(f"{API}/players/nfl", timeout=120).json())


@st.cache_data(ttl=900, show_spinner=False)
def fetch_trending(limit: int = 100) -> dict:
    """sleeper_id -> add count over the last 24h."""
    data = http.get(f"{API}/players/nfl/trending/add",
                    params={"lookback_hours": 24, "limit": limit}).json()
    return {item["player_id"]: item["count"] for item in data}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_state() -> dict:
    """Current NFL season and week, per Sleeper."""
    data = http.get(f"{API}/state/nfl").json()
    return {"season": int(data.get("season") or 0), "week": int(data.get("week") or 0),
            "season_type": data.get("season_type", "")}


def clean_week(items: list) -> dict:
    """Raw projection rows for one week -> {sleeper_id: (pos, pts, opponent)}.
    Glitched values are replaced with that position's median for the week."""
    by_pos: dict[str, list] = {}
    for it in items:
        pos = _POS.get((it.get("player") or {}).get("position"))
        pts = (it.get("stats") or {}).get("pts_ppr")
        if pos is None or pts is None:
            continue
        by_pos.setdefault(pos, []).append((str(it.get("player_id")), float(pts),
                                           normalize_team(it.get("opponent"))))
    out = {}
    for pos, rows in by_pos.items():
        cap = _SANE_MAX[pos]
        sane = [pts for _, pts, _ in rows if 0 < pts <= cap]
        median = statistics.median(sane) if sane else 0.0
        for sid, pts, opp in rows:
            out[sid] = (pos, round(pts if pts <= cap else median, 2), opp)
    return out


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_week_projections(season: int, week: int) -> dict:
    """{sleeper_id: (pos, pts, opponent)} for one NFL week."""
    params = [("season_type", "regular"), ("order_by", "pts_ppr")]
    params += [("position[]", p) for p in _POS]
    items = http.get(f"{PROJ_API}/{season}/{week}", params=params, timeout=45).json()
    return clean_week(items)


def fetch_projections(season: int, weeks: tuple[int, ...]) -> dict:
    """Week-by-week projections for a span of weeks.
    Returns {"pts": {sleeper_id: {week: pts}}, "opp": {sleeper_id: {week: opp}},
             "pos": {sleeper_id: position}, "weeks": [weeks that loaded]};
    a week that fails to load is skipped."""
    pts: dict[str, dict] = {}
    opp: dict[str, dict] = {}
    pos: dict[str, str] = {}
    loaded = []
    for wk in weeks:
        try:
            data = fetch_week_projections(season, wk)
        except Exception:
            continue
        if not data:
            continue
        loaded.append(wk)
        for sid, (position, p, o) in data.items():
            pts.setdefault(sid, {})[wk] = p
            pos[sid] = position
            if o:
                opp.setdefault(sid, {})[wk] = o
    return {"pts": pts, "opp": opp, "pos": pos, "weeks": loaded}
