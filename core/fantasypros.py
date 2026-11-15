"""FantasyPros expert-consensus rankings (PPR): draft, rest-of-season, weekly.

Every rankings page embeds its data as `var ecrData = {...}`; that JSON is
the source here, so there is no HTML table scraping to break.
"""

import json
import time

import pandas as pd
import streamlit as st

from core import http
from core.names import POSITIONS, normalize_team, player_key

BASE = "https://www.fantasypros.com/nfl/rankings/"
_WEEKLY_PAGES = {
    "QB": "qb.php", "RB": "ppr-rb.php", "WR": "ppr-wr.php",
    "TE": "ppr-te.php", "K": "k.php", "DST": "dst.php", "FLEX": "ppr-flex.php",
}


def _fetch_ecr_page(page: str) -> tuple[list, dict]:
    """Return (players, page metadata) from a rankings page's embedded JSON."""
    text = http.get(BASE + page).text
    marker = "var ecrData = "
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"FantasyPros page layout changed: ecrData not found on {page}")
    data, _ = json.JSONDecoder().raw_decode(text[idx + len(marker):])
    return data.get("players", []), data


def _ecr_rows(players: list) -> list[dict]:
    rows = []
    for p in players:
        pos = p.get("player_position_id", "")
        if pos not in POSITIONS:
            continue
        bye = str(p.get("player_bye_week") or "")
        rows.append({
            "player": p["player_name"],
            "team": normalize_team(p.get("player_team_id", "")),
            "pos": pos,
            "ecr": int(p["rank_ecr"]),
            "ecr_avg": float(p.get("rank_ave") or p["rank_ecr"]),
            "ecr_std": float(p.get("rank_std") or 0),
            "ecr_best": int(p.get("rank_min") or p["rank_ecr"]),
            "ecr_worst": int(p.get("rank_max") or p["rank_ecr"]),
            "pos_rank": p.get("pos_rank", ""),
            "tier": int(p.get("tier") or 0),
            "bye": int(bye) if bye.isdigit() else 0,
        })
    return rows


def _with_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["key"] = [player_key(n, p, t) for n, p, t in zip(df["player"], df["pos"], df["team"])]
    return df


@st.cache_data(ttl=900, show_spinner="Fetching FantasyPros consensus rankings…")
def fetch_draft_ecr() -> pd.DataFrame:
    """Pre-draft PPR cheat sheet (ranks, tiers, byes, expert spread)."""
    players, data = _fetch_ecr_page("ppr-cheatsheets.php")
    df = _with_keys(pd.DataFrame(_ecr_rows(players)))
    df.attrs["updated"] = data.get("last_updated") or ""
    df.attrs["fetched"] = time.strftime("%I:%M %p").lstrip("0")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ros() -> pd.DataFrame:
    """Rest-of-season PPR consensus: the season-long value scale."""
    players, data = _fetch_ecr_page("ros-ppr-overall.php")
    df = _with_keys(pd.DataFrame(_ecr_rows(players))).rename(columns={
        "ecr": "ros_rank", "tier": "ros_tier", "pos_rank": "ros_pos_rank"})
    df.attrs["updated"] = data.get("last_updated") or ""
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekly() -> pd.DataFrame:
    """This week's positional PPR ranks (the start/sit scale).
    Columns: key, weekly_rank (within position), flex_rank (RB/WR/TE only)."""
    pos_frames, flex_frame, week = [], None, 0
    for pos, page in _WEEKLY_PAGES.items():
        try:
            players, data = _fetch_ecr_page(page)
        except Exception:
            continue
        week = int(data.get("week") or 0) or week
        col = "flex_rank" if pos == "FLEX" else "weekly_rank"
        rows = [{"key": player_key(p["player_name"], p.get("player_position_id", ""),
                                   p.get("player_team_id", "")),
                 col: int(p["rank_ecr"])}
                for p in players if p.get("player_position_id") in POSITIONS]
        if pos == "FLEX":
            flex_frame = pd.DataFrame(rows)
        else:
            pos_frames.append(pd.DataFrame(rows))
    if not pos_frames:
        return pd.DataFrame(columns=["key", "weekly_rank", "flex_rank"])
    # Each player appears on exactly one positional page, so rows just stack.
    out = pd.concat(pos_frames, ignore_index=True).drop_duplicates("key")
    if flex_frame is not None and not flex_frame.empty:
        out = out.merge(flex_frame.drop_duplicates("key"), on="key", how="outer")
    else:
        out["flex_rank"] = pd.NA
    out.attrs["week"] = week
    return out
