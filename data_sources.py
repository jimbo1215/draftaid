"""Data source fetchers and the merged draft board builder.

Sources (all free, no API key):
  - FantasyPros PPR expert-consensus rankings (embedded ecrData JSON on the
    cheat-sheet page) -- rankings, tiers, bye weeks, rank spread.
  - FantasyFootballCalculator ADP API -- real 12-team PPR draft ADP with stdev.
  - Sleeper API -- injury status and 24h trending adds.
  - Optional uploaded CSV (Underdog rankings export or any name+rank CSV).
"""

import json
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

FANTASYPROS_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/add"

# Team-code variants across sources, normalized to FantasyPros codes.
TEAM_ALIASES = {
    "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU", "LA": "LAR", "SD": "LAC", "OAK": "LV", "STL": "LAR",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase alpha-only key with generational suffixes stripped, so the
    same player joins across sources ("Marvin Harrison Jr." == "marvinharrison")."""
    if not isinstance(name, str):
        return ""
    tokens = re.sub(r"[^a-z\s]", "", name.lower()).split()
    tokens = [t for t in tokens if t not in _SUFFIXES]
    return "".join(tokens)


def normalize_team(team: str) -> str:
    team = (team or "").upper().strip()
    return TEAM_ALIASES.get(team, team)


def _player_key(row_name: str, pos: str, team: str) -> str:
    # Defenses have inconsistent names across sources; key them by team.
    if pos == "DST":
        return f"dst{normalize_team(team).lower()}"
    return f"{normalize_name(row_name)}|{pos}"


@st.cache_data(ttl=900, show_spinner="Fetching FantasyPros consensus rankings...")
def fetch_fantasypros() -> pd.DataFrame:
    resp = requests.get(FANTASYPROS_URL, headers=UA_HEADERS, timeout=30)
    resp.raise_for_status()
    marker = "var ecrData = "
    idx = resp.text.find(marker)
    if idx == -1:
        raise RuntimeError("FantasyPros page layout changed: ecrData not found")
    data, _ = json.JSONDecoder().raw_decode(resp.text[idx + len(marker):])
    rows = []
    for p in data.get("players", []):
        pos = p.get("player_position_id", "")
        if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
            continue
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
            "bye": int(p["player_bye_week"]) if str(p.get("player_bye_week") or "").isdigit() else 0,
        })
    df = pd.DataFrame(rows)
    df["key"] = [_player_key(n, p, t) for n, p, t in zip(df["player"], df["pos"], df["team"])]
    return df


@st.cache_data(ttl=900, show_spinner="Fetching live ADP...")
def fetch_ffc_adp(teams: int = 12, year: int = 2026) -> pd.DataFrame:
    resp = requests.get(FFC_ADP_URL, params={"teams": teams, "year": year}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = []
    for p in payload.get("players", []):
        pos = {"DEF": "DST", "PK": "K"}.get(p["position"], p["position"])
        if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
            continue
        rows.append({
            "adp_name": p["name"],
            "team": normalize_team(p.get("team", "")),
            "pos": pos,
            "adp": float(p["adp"]),
            "adp_std": max(float(p.get("stdev") or 6.0), 2.0),
            "adp_pick": p.get("adp_formatted", ""),
        })
    df = pd.DataFrame(rows)
    df["key"] = [_player_key(n, p, t) for n, p, t in zip(df["adp_name"], df["pos"], df["team"])]
    meta = payload.get("meta", {})
    df.attrs["drafts"] = meta.get("total_drafts", 0)
    df.attrs["end_date"] = meta.get("end_date", "")
    return df.drop(columns=["adp_name"])


@st.cache_data(ttl=6 * 3600, show_spinner="Loading Sleeper player database (one-time, ~15s)...")
def fetch_sleeper_players() -> pd.DataFrame:
    resp = requests.get(SLEEPER_PLAYERS_URL, timeout=120)
    resp.raise_for_status()
    players = resp.json()
    rows = []
    for pid, p in players.items():
        pos = p.get("position")
        if pos == "DEF":
            rows.append({"sleeper_id": pid, "key": f"dst{normalize_team(pid).lower()}",
                         "injury": "", "injury_note": ""})
            continue
        if pos not in {"QB", "RB", "WR", "TE", "K"} or p.get("status") not in {"Active", "Injured Reserve"}:
            continue
        name_key = p.get("search_full_name") or normalize_name(p.get("full_name", ""))
        rows.append({
            "sleeper_id": pid,
            "key": f"{name_key}|{pos}",
            "injury": p.get("injury_status") or "",
            "injury_note": p.get("injury_body_part") or "",
        })
    return pd.DataFrame(rows).drop_duplicates(subset="key", keep="first")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_sleeper_trending(limit: int = 60) -> dict:
    """Map of sleeper_id -> add count over the last 24h."""
    resp = requests.get(SLEEPER_TRENDING_URL,
                        params={"lookback_hours": 24, "limit": limit}, timeout=30)
    resp.raise_for_status()
    return {item["player_id"]: item["count"] for item in resp.json()}


def parse_rankings_csv(file) -> pd.DataFrame:
    """Flexible parser for an uploaded rankings CSV (Underdog export or similar).
    Finds a player-name column (or firstName+lastName) and a rank/ADP column,
    and returns key + csv_rank. Raises ValueError with a readable message."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "firstname" in df.columns and "lastname" in df.columns:
        names = df["firstname"].fillna("") + " " + df["lastname"].fillna("")
    else:
        name_col = next((c for c in ("player", "name", "player_name", "full_name", "player name")
                         if c in df.columns), None)
        if name_col is None:
            raise ValueError(f"No player-name column found. Columns: {list(df.columns)}")
        names = df[name_col].astype(str)

    rank_col = next((c for c in ("rank", "overall", "overall rank", "adp", "ecr", "rk", "projectedpoints")
                     if c in df.columns), None)
    if rank_col is None:
        raise ValueError(f"No rank/ADP column found. Columns: {list(df.columns)}")

    ranks = pd.to_numeric(df[rank_col], errors="coerce")
    if rank_col == "projectedpoints":  # higher points = better -> convert to rank
        ranks = ranks.rank(ascending=False)

    pos_col = next((c for c in ("pos", "position", "slotname") if c in df.columns), None)
    out = pd.DataFrame({
        "name_key": names.map(normalize_name),
        "csv_rank": ranks,
        "csv_pos": df[pos_col].astype(str).str.upper().str.extract(r"([A-Z]+)")[0] if pos_col else None,
    }).dropna(subset=["csv_rank"])
    out = out.sort_values("csv_rank").drop_duplicates(subset="name_key", keep="first")
    return out


def build_board(csv_ranks: pd.DataFrame | None = None,
                weights: dict | None = None) -> pd.DataFrame:
    """Merge all sources into the master board, sorted by blended consensus rank."""
    fp = fetch_fantasypros()
    board = fp.copy()

    adp_drafts, adp_date = 0, ""
    try:
        adp = fetch_ffc_adp()
        board = board.merge(adp[["key", "adp", "adp_std", "adp_pick"]], on="key", how="left")
        adp_drafts = adp.attrs.get("drafts", 0)
        adp_date = adp.attrs.get("end_date", "")
    except Exception:
        board["adp"], board["adp_std"], board["adp_pick"] = np.nan, np.nan, ""

    try:
        sleeper = fetch_sleeper_players()
        board = board.merge(sleeper, on="key", how="left")
        trending = fetch_sleeper_trending()
        board["trending"] = board["sleeper_id"].map(trending).fillna(0).astype(int)
    except Exception:
        board["injury"], board["injury_note"], board["trending"] = "", "", 0
    board["injury"] = board["injury"].fillna("")

    if csv_ranks is not None and not csv_ranks.empty:
        name_keys = board["player"].map(normalize_name)
        board["csv_rank"] = name_keys.map(csv_ranks.set_index("name_key")["csv_rank"])
    else:
        board["csv_rank"] = np.nan

    # Blended consensus: weighted average of the source ranks each player has.
    w = weights or {"ecr": 0.5, "adp": 0.3, "csv": 0.2}
    adp_rank = board["adp"].rank(method="first")
    csv_rank = board["csv_rank"].rank(method="first")
    parts = pd.DataFrame({
        "ecr": board["ecr"] * w.get("ecr", 0),
        "adp": adp_rank * w.get("adp", 0),
        "csv": csv_rank * w.get("csv", 0),
    })
    weight_sum = (
        w.get("ecr", 0) * board["ecr"].notna()
        + w.get("adp", 0) * adp_rank.notna()
        + w.get("csv", 0) * csv_rank.notna()
    )
    board["consensus"] = parts.sum(axis=1) / weight_sum.replace(0, np.nan)
    board = board.sort_values("consensus").reset_index(drop=True)
    board["rank"] = board.index + 1

    # Value vs the market: positive = the market drafts him later than experts rank him.
    board["value"] = (board["adp"] - board["consensus"]).round(1)
    board.attrs["adp_drafts"] = adp_drafts
    board.attrs["adp_date"] = adp_date
    return board
