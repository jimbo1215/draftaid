"""FantasyFootballCalculator ADP from real drafts (used by the Draft Room)."""

import pandas as pd
import streamlit as st

from core import http
from core.names import POSITIONS, normalize_team, player_key

URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"


@st.cache_data(ttl=900, show_spinner="Fetching live ADP…")
def fetch_adp(teams: int = 12, year: int = 2026) -> pd.DataFrame:
    payload = http.get(URL, params={"teams": teams, "year": year}).json()
    rows = []
    for p in payload.get("players", []):
        pos = {"DEF": "DST", "PK": "K"}.get(p["position"], p["position"])
        if pos not in POSITIONS:
            continue
        team = normalize_team(p.get("team", ""))
        rows.append({
            "key": player_key(p["name"], pos, team),
            "adp": float(p["adp"]),
            "adp_std": max(float(p.get("stdev") or 6.0), 2.0),
            "adp_pick": p.get("adp_formatted", ""),
        })
    df = pd.DataFrame(rows)
    meta = payload.get("meta", {})
    df.attrs["drafts"] = meta.get("total_drafts", 0)
    df.attrs["end_date"] = meta.get("end_date", "")
    return df
