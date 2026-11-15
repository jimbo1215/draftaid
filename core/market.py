"""FantasyCalc market values: what players actually trade for in real leagues."""

import pandas as pd
import streamlit as st

from core import http

URL = "https://api.fantasycalc.com/values/current"


def parse_values(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        player = item.get("player") or {}
        sid = player.get("sleeperId")
        if sid is None:
            continue
        try:
            espn_id = int(player.get("espnId"))
        except (TypeError, ValueError):
            espn_id = None
        rows.append({"sleeper_id": str(sid),
                     # The feed carries both ids, so it doubles as an ESPN<->Sleeper bridge.
                     "mkt_espn_id": espn_id,
                     "mkt_value": float(item.get("value") or 0),
                     "mkt_rank": item.get("overallRank"),
                     # 30-day change in market value: who the market is buying/selling.
                     "mkt_trend": float(item.get("trend30Day") or 0)})
    if not rows:
        return pd.DataFrame(columns=["sleeper_id", "mkt_espn_id", "mkt_value", "mkt_rank",
                                     "mkt_trend"])
    return pd.DataFrame(rows).drop_duplicates("sleeper_id")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_values(teams: int = 12) -> pd.DataFrame:
    """Redraft, 1-QB, PPR values derived from real trades, keyed by Sleeper id."""
    items = http.get(URL, params={"isDynasty": "false", "numQbs": 1,
                                  "numTeams": teams, "ppr": 1}).json()
    return parse_values(items)
