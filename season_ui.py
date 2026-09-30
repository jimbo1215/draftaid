"""Shared context loader and page chrome for the season pages."""

import time

import pandas as pd
import streamlit as st

import theme
from data_sources import (fetch_fp_ros, fetch_fp_weekly, fetch_market_values,
                          fetch_sleeper_players, fetch_sleeper_trending)
from espn_league import get_free_agents, get_league, load_config
from season import enrich


def _connect_prompt(title: str, body: str, link_label: str):
    theme.page_header(title, ["DraftAid"])
    theme.empty(body)
    st.page_link("views/setup.py", label=link_label, icon=":material/arrow_forward:")
    st.stop()


def _sources():
    ros = fetch_fp_ros()
    weekly = fetch_fp_weekly()
    try:
        sleeper = fetch_sleeper_players()
        trending = fetch_sleeper_trending()
    except Exception:
        sleeper, trending = pd.DataFrame(), {}
    try:
        market = fetch_market_values()
    except Exception:
        market = pd.DataFrame()
    return ros, weekly, sleeper, trending, market


def league_ctx():
    """Load config + league + rank-enriched rosters, or stop with a setup hint.
    Returns (cfg, league, teams: {team_id: DataFrame}, names: {team_id: str})."""
    cfg = load_config()
    if not cfg.get("league_id"):
        _connect_prompt("Connect your league",
                        "<b>DraftAid isn't connected to your ESPN league yet.</b> Add your "
                        "league ID (plus the espn_s2 / SWID cookies if it's private) and "
                        "every page here fills in.", "Open League Setup")
    try:
        with st.spinner("Syncing with ESPN…"):
            league = get_league(cfg)
    except Exception as e:
        _connect_prompt("League unavailable", f"<b>Couldn't load your ESPN league.</b> "
                        f"{theme.esc(e)}", "Check League Setup")

    with st.spinner("Pulling rankings and market data…"):
        ros, weekly, sleeper, trending, market = _sources()
    teams = {t["team_id"]: enrich(t["roster"], ros, weekly, sleeper, trending, market)
             for t in league["teams"]}
    names = {t["team_id"]: t["name"] for t in league["teams"]}
    return cfg, league, teams, names


def free_agents(cfg: dict, league: dict) -> pd.DataFrame:
    """The league's available players, rank-enriched like the rosters."""
    ros, weekly, sleeper, trending, market = _sources()
    return enrich(get_free_agents(cfg, league), ros, weekly, sleeper, trending, market)


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
    _connect_prompt("Which team is yours?",
                    "<b>I don't know which team is yours yet.</b> Pick it in League Setup "
                    "and your roster, waivers, and trades will be tailored to it.",
                    "Open League Setup")


def _synced(league: dict) -> str:
    ts = league.get("fetched")
    if not ts:
        return "Live"
    mins = int((time.time() - ts) // 60)
    return "Synced just now" if mins < 1 else f"Synced {mins} min ago"


def header(league: dict, title: str, sub: str | None = None):
    """Page title block with league/week context and a refresh button."""
    c1, c2 = st.columns([8, 1], vertical_alignment="top", gap="small")
    with c1:
        theme.page_header(title, [league.get("league_name", ""),
                                  f"Week {league.get('week', '?')}"],
                          sub=sub, live=_synced(league))
    with c2:
        if st.button("", icon=":material/refresh:", key="season_refresh",
                     help="Re-pull ESPN, rankings, and trending data now",
                     width="stretch"):
            st.cache_data.clear()
            st.rerun()
