"""Shared context loader for the season pages."""

import pandas as pd
import streamlit as st

from data_sources import (fetch_fp_ros, fetch_fp_weekly, fetch_sleeper_players,
                          fetch_sleeper_trending)
from espn_league import get_free_agents, get_league, load_config
from season import enrich


def league_ctx():
    """Load config + league + rank-enriched rosters, or stop with a setup hint.
    Returns (cfg, league, teams: {team_id: DataFrame}, names: {team_id: str})."""
    cfg = load_config()
    if not cfg.get("league_id"):
        st.info("**Not connected to your ESPN league yet.** Head to League Setup, "
                "paste your league ID (plus espn_s2/SWID cookies if the league is "
                "private), and everything here lights up.")
        st.page_link("views/setup.py", label="→ Open League Setup", icon="⚙️")
        st.stop()
    try:
        league = get_league(cfg)
    except Exception as e:
        st.error(f"Couldn't load your ESPN league: {e}")
        st.page_link("views/setup.py", label="→ Check League Setup", icon="⚙️")
        st.stop()

    ros = fetch_fp_ros()
    weekly = fetch_fp_weekly()
    try:
        sleeper = fetch_sleeper_players()
        trending = fetch_sleeper_trending()
    except Exception:
        sleeper, trending = pd.DataFrame(), {}

    teams = {t["team_id"]: enrich(t["roster"], ros, weekly, sleeper, trending)
             for t in league["teams"]}
    names = {t["team_id"]: t["name"] for t in league["teams"]}
    return cfg, league, teams, names


def my_team_id(cfg: dict, league: dict) -> int | None:
    """Configured team id, else auto-detect from the SWID cookie."""
    tid = cfg.get("my_team_id")
    if tid is not None and any(t["team_id"] == tid for t in league["teams"]):
        return tid
    swid = (cfg.get("swid") or "").strip().upper()
    if swid:
        swid = "{" + swid.strip("{}") + "}"
        for t in league["teams"]:
            if swid in t.get("owners", []):
                return t["team_id"]
    return None


def need_my_team():
    st.warning("I don't know which team is yours yet — pick it in League Setup.")
    st.page_link("views/setup.py", label="→ Open League Setup", icon="⚙️")
    st.stop()


def refresh_row(league: dict, weekly: pd.DataFrame | None = None):
    c1, c2 = st.columns([0.8, 6], vertical_alignment="center", gap="small")
    if c1.button("🔄", key="season_refresh", width="stretch",
                 help="Re-pull ESPN league data, rankings, and trending now"):
        st.cache_data.clear()
        st.rerun()
    wk = league.get("week", "?")
    c2.caption(f"**{league.get('league_name', '')}** · NFL week {wk} · "
               "data refreshes every 5 min, 🔄 forces it")
