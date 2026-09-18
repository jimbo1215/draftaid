"""ESPN Fantasy Football league client (unofficial v3 API).

Public leagues need only the league ID. Private leagues also need the viewer's
`espn_s2` and `SWID` cookies (read-only access as that user). Credentials are
stored locally in league_config.json (gitignored) or Streamlit secrets under
[espn] -- they never leave the app except in requests to ESPN itself.
"""

import json
from pathlib import Path

import requests
import streamlit as st

from data_sources import normalize_name

CONFIG_FILE = Path(__file__).parent / "league_config.json"
BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
PRO_TEAMS = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI",
    23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}
SLOT_MAP = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "OP", 16: "DST", 17: "K",
            20: "BN", 21: "IR", 23: "FLEX"}
BENCH_SLOTS = {20, 21}


# ------------------------------------------------------------------ config

def load_config() -> dict:
    """League connection settings: file first, Streamlit secrets as fallback."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        if "espn" in st.secrets:
            return dict(st.secrets["espn"])
    except Exception:
        pass
    return {}


def save_config(cfg: dict) -> bool:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg))
        return True
    except OSError:
        return False


def _cookies(cfg: dict) -> dict:
    out = {}
    if cfg.get("espn_s2"):
        out["espn_s2"] = cfg["espn_s2"].strip()
    if cfg.get("swid"):
        swid = cfg["swid"].strip()
        if not swid.startswith("{"):
            swid = "{" + swid.strip("{}") + "}"
        out["SWID"] = swid
    return out


# ------------------------------------------------------------------ fetchers

@st.cache_data(ttl=300, show_spinner="Loading your ESPN league...")
def fetch_league_raw(league_id: str, year: int, espn_s2: str = "", swid: str = "") -> dict:
    resp = requests.get(
        BASE.format(year=year, league_id=league_id),
        params=[("view", v) for v in
                ("mTeam", "mRoster", "mSettings", "mMatchupScore")],
        cookies=_cookies({"espn_s2": espn_s2, "swid": swid}),
        headers=UA, timeout=30)
    if resp.status_code == 401:
        raise PermissionError(
            "ESPN says this league is private (401). Add your espn_s2 and SWID "
            "cookies in League Setup.")
    if resp.status_code == 404:
        raise LookupError(f"ESPN can't find league {league_id} for {year} (404). "
                          "Double-check the league ID and season.")
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


@st.cache_data(ttl=300, show_spinner="Scanning the waiver wire...")
def fetch_free_agents(league_id: str, year: int, week: int,
                      espn_s2: str = "", swid: str = "", limit: int = 300) -> list:
    fltr = {"players": {
        "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
        "limit": limit,
        "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
    }}
    resp = requests.get(
        BASE.format(year=year, league_id=league_id),
        params={"view": "kona_player_info", "scoringPeriodId": week},
        cookies=_cookies({"espn_s2": espn_s2, "swid": swid}),
        headers={**UA, "X-Fantasy-Filter": json.dumps(fltr)}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        data = data[0] if data else {}
    return data.get("players", [])


# ------------------------------------------------------------------ parsing

def _week_proj(player: dict, week: int) -> float | None:
    """ESPN projected points for a given week (statSourceId 1 = projection)."""
    for s in player.get("stats", []) or []:
        if s.get("scoringPeriodId") == week and s.get("statSourceId") == 1:
            v = s.get("appliedTotal")
            if v is not None:
                return round(float(v), 1)
    return None


def _season_actual(player: dict) -> float | None:
    for s in player.get("stats", []) or []:
        if s.get("scoringPeriodId") == 0 and s.get("statSourceId") == 0:
            v = s.get("appliedTotal")
            if v is not None:
                return round(float(v), 1)
    return None


def parse_player(player: dict, week: int) -> dict:
    pos = POS_MAP.get(player.get("defaultPositionId"), "?")
    team = PRO_TEAMS.get(player.get("proTeamId"), "FA")
    name = player.get("fullName") or ""
    own = (player.get("ownership") or {}).get("percentOwned")
    return {
        "espn_id": player.get("id"),
        "player": name,
        "pos": pos,
        "team": team,
        "key": (f"dst{team.lower()}" if pos == "DST"
                else f"{normalize_name(name)}|{pos}"),
        "espn_injury": player.get("injuryStatus") or "",
        "pct_owned": round(float(own), 1) if own is not None else None,
        "week_proj": _week_proj(player, week),
        "season_pts": _season_actual(player),
    }


def parse_league(raw: dict) -> dict:
    """Normalize the raw ESPN payload into teams/rosters/schedule/settings."""
    settings = raw.get("settings", {}) or {}
    acq = settings.get("acquisitionSettings", {}) or {}
    week = int(raw.get("scoringPeriodId") or 1)
    out = {
        "league_name": settings.get("name", "ESPN League"),
        "week": week,
        "faab_budget": int(acq.get("acquisitionBudget") or 0),
        "uses_faab": bool(acq.get("isUsingAcquisitionBudget",
                                  acq.get("acquisitionBudget"))),
        "teams": [],
        "schedule": [],
    }
    for t in raw.get("teams", []) or []:
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip() \
            or f"Team {t.get('id')}"
        rec = ((t.get("record") or {}).get("overall") or {})
        counter = t.get("transactionCounter") or {}
        roster = []
        for entry in ((t.get("roster") or {}).get("entries") or []):
            pool = entry.get("playerPoolEntry") or {}
            player = pool.get("player") or {}
            if not player:
                continue
            p = parse_player(player, week)
            slot_id = entry.get("lineupSlotId")
            p["slot"] = SLOT_MAP.get(slot_id, str(slot_id))
            p["starter"] = slot_id not in BENCH_SLOTS
            roster.append(p)
        out["teams"].append({
            "team_id": t.get("id"),
            "owners": [str(o).upper() for o in (t.get("owners") or [])],
            "name": name,
            "abbrev": t.get("abbrev", ""),
            "wins": rec.get("wins", 0),
            "losses": rec.get("losses", 0),
            "ties": rec.get("ties", 0),
            "points_for": round(float(rec.get("pointsFor") or 0), 1),
            "points_against": round(float(rec.get("pointsAgainst") or 0), 1),
            "faab_spent": int(counter.get("acquisitionBudgetSpent") or 0),
            "moves": int(counter.get("acquisitions") or 0),
            "roster": roster,
        })
    for m in raw.get("schedule", []) or []:
        home, away = m.get("home") or {}, m.get("away") or {}
        out["schedule"].append({
            "week": m.get("matchupPeriodId"),
            "home_id": home.get("teamId"),
            "home_pts": round(float(home.get("totalPoints") or 0), 1),
            "away_id": away.get("teamId"),
            "away_pts": round(float(away.get("totalPoints") or 0), 1),
            "winner": m.get("winner", ""),
        })
    return out


def get_league(cfg: dict) -> dict:
    raw = fetch_league_raw(str(cfg["league_id"]), int(cfg.get("year") or 2026),
                           cfg.get("espn_s2", ""), cfg.get("swid", ""))
    return parse_league(raw)


def get_free_agents(cfg: dict, week: int) -> list[dict]:
    players = fetch_free_agents(str(cfg["league_id"]), int(cfg.get("year") or 2026),
                                week, cfg.get("espn_s2", ""), cfg.get("swid", ""))
    out = []
    for entry in players:
        player = entry.get("player") or entry.get("playerPoolEntry", {}).get("player") or {}
        if player:
            out.append(parse_player(player, week))
    return out
