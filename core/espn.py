"""ESPN Fantasy Football league client (unofficial v3 API).

Public leagues need only the league ID. Private leagues also need the viewer's
`espn_s2` and `SWID` cookies (read-only access as that user). Credentials are
stored locally in league_config.json (gitignored) or Streamlit secrets under
[espn]; they never leave the app except in requests to ESPN itself.
"""

import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

from core.http import UA
from core.names import player_key

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "league_config.json"
BASE = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}"
        "/segments/0/leagues/{league_id}")
VIEWS = ("mTeam", "mRoster", "mSettings", "mMatchupScore", "mScoreboard")

POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
PRO_TEAMS = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI",
    23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}
SLOT_MAP = {0: "QB", 2: "RB", 3: "RB/WR", 4: "WR", 6: "TE", 7: "OP", 16: "DST",
            17: "K", 20: "BN", 21: "IR", 23: "FLEX"}
BENCH_SLOTS = {20, 21}
# lineupSlotCounts ids -> the starter slots the lineup model understands.
LINEUP_SLOT_IDS = {0: "QB", 2: "RB", 3: "RB/WR", 4: "WR", 6: "TE", 7: "OP",
                   16: "DST", 17: "K", 23: "FLEX"}
DECIDED = ("HOME", "AWAY", "TIE")

# Dev-only: point DRAFTAID_FIXTURE at a folder holding league.json + fas.json
# (raw ESPN payloads, see tools/make_fixture.py) to run without a live league.
_FIXTURE = os.environ.get("DRAFTAID_FIXTURE")


def _fixture(name: str):
    folder = Path(_FIXTURE)
    if not folder.is_absolute():
        folder = ROOT / folder
    return json.loads((folder / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ config

def load_config() -> dict:
    """League connection settings: file first, Streamlit secrets as fallback."""
    if _FIXTURE:
        return {"league_id": "fixture", "year": 2026, "my_team_id": 8,
                **_saved_prefs()}
    saved = _read_config_file()
    if saved.get("league_id"):
        return saved
    try:
        if "espn" in st.secrets:
            return {**dict(st.secrets["espn"]), **_saved_prefs()}
    except Exception:
        pass
    return saved


def _read_config_file() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _saved_prefs() -> dict:
    """Non-credential preferences (e.g. theme) saved alongside the config."""
    return {k: v for k, v in _read_config_file().items() if k in ("theme",)}


def save_config(cfg: dict) -> bool:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg))
        return True
    except OSError:
        return False


def save_pref(key: str, value) -> bool:
    """Persist one preference without disturbing saved credentials."""
    return save_config({**_read_config_file(), key: value})


def _cookies(espn_s2: str, swid: str) -> dict:
    out = {}
    if espn_s2:
        out["espn_s2"] = espn_s2.strip()
    if swid:
        out["SWID"] = "{" + swid.strip().strip("{}") + "}"
    return out


# ------------------------------------------------------------------ fetchers

@st.cache_data(ttl=180, show_spinner=False)
def fetch_league_raw(league_id: str, year: int, espn_s2: str = "", swid: str = "") -> dict:
    if _FIXTURE:
        return {**_fixture("league.json"), "_fetched": time.time()}
    resp = requests.get(
        BASE.format(year=year, league_id=league_id),
        params=[("view", v) for v in VIEWS],
        cookies=_cookies(espn_s2, swid), headers=UA, timeout=30)
    if resp.status_code == 401:
        raise PermissionError(
            "ESPN says this league is private (401). Add your espn_s2 and SWID "
            "cookies in Settings.")
    if resp.status_code == 404:
        raise LookupError(f"ESPN can't find league {league_id} for {year} (404). "
                          "Double-check the league ID and season.")
    resp.raise_for_status()
    data = resp.json()
    data = data[0] if isinstance(data, list) else data
    data["_fetched"] = time.time()
    return data


@st.cache_data(ttl=180, show_spinner=False)
def fetch_free_agents(league_id: str, year: int, week: int,
                      espn_s2: str = "", swid: str = "", limit: int = 300) -> list:
    if _FIXTURE:
        return _fixture("fas.json")
    fltr = {"players": {
        "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
        "limit": limit,
        "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
    }}
    resp = requests.get(
        BASE.format(year=year, league_id=league_id),
        params={"view": "kona_player_info", "scoringPeriodId": week},
        cookies=_cookies(espn_s2, swid),
        headers={**UA, "X-Fantasy-Filter": json.dumps(fltr)}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        data = data[0] if data else {}
    return data.get("players", [])


# ------------------------------------------------------------------ parsing

def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _stat(player: dict, week: int, source: int) -> float | None:
    """A player's applied (league-scored) total for a scoring period.
    source 0 = actual, 1 = projected; week 0 = season total."""
    for s in player.get("stats", []) or []:
        if s.get("scoringPeriodId") == week and s.get("statSourceId") == source:
            v = s.get("appliedTotal")
            if v is not None:
                return round(float(v), 1)
    return None


def parse_player(player: dict, week: int) -> dict:
    pos = POS_MAP.get(player.get("defaultPositionId"), "?")
    team = PRO_TEAMS.get(player.get("proTeamId"), "FA")
    name = player.get("fullName") or ""
    ownership = player.get("ownership") or {}
    own, own_chg = ownership.get("percentOwned"), ownership.get("percentChange")
    return {
        "espn_id": player.get("id"),
        "player": name,
        "pos": pos,
        "team": team,
        "key": player_key(name, pos, team),
        "espn_injury": player.get("injuryStatus") or "",
        "pct_owned": round(float(own), 1) if own is not None else None,
        "own_change": round(float(own_chg), 1) if own_chg is not None else None,
        "week_proj": _stat(player, week, 1),
        "week_pts": _stat(player, week, 0),
        "season_pts": _stat(player, 0, 0),
    }


def _side(side: dict, matchup_week) -> tuple[float, float | None]:
    """(points, live projected final) for one side of a matchup. ESPN fills
    totalPoints only once games are final; live scores arrive in
    totalPointsLive / pointsByScoringPeriod."""
    candidates = [side.get("totalPoints"), side.get("totalPointsLive"),
                  (side.get("pointsByScoringPeriod") or {}).get(str(matchup_week))]
    vals = [float(v) for v in candidates if v is not None]
    live = side.get("totalProjectedPointsLive")
    return (round(max(vals), 1) if vals else 0.0,
            round(float(live), 1) if live is not None else None)


def parse_league(raw: dict) -> dict:
    """Normalize the raw ESPN payload into settings, teams, and schedule."""
    settings = raw.get("settings") or {}
    acq = settings.get("acquisitionSettings") or {}
    sched = settings.get("scheduleSettings") or {}
    trade = settings.get("tradeSettings") or {}
    status = raw.get("status") or {}
    week = _int(raw.get("scoringPeriodId"), 1) or 1
    counts = (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    lineup = {LINEUP_SLOT_IDS[int(k)]: int(v) for k, v in counts.items()
              if int(k) in LINEUP_SLOT_IDS and int(v) > 0}
    reg_weeks = _int(sched.get("matchupPeriodCount"), 14) or 14
    deadline = trade.get("deadlineDate")
    out = {
        "league_name": settings.get("name", "ESPN League"),
        "week": week,
        "matchup_period": _int(status.get("currentMatchupPeriod"), week) or week,
        "final_week": _int(status.get("finalScoringPeriod"), 17) or 17,
        "fetched": raw.get("_fetched"),
        "lineup": lineup,
        "bench_slots": _int(counts.get("20") or counts.get(20)),
        "regular_season_weeks": reg_weeks,
        "playoff_teams": _int(sched.get("playoffTeamCount")),
        "playoff_round_weeks": max(1, _int(sched.get("playoffMatchupPeriodLength"), 1)),
        "seeding_rule": sched.get("playoffSeedingRule") or "",
        # matchup period -> the NFL weeks it spans (multi-week playoff rounds)
        "matchup_weeks": {_int(k): [_int(x) for x in v]
                          for k, v in (sched.get("matchupPeriods") or {}).items()},
        "trade_deadline": float(deadline) / 1000 if deadline else None,
        "faab_budget": _int(acq.get("acquisitionBudget")),
        "min_bid": _int(acq.get("minimumBid")),
        "uses_faab": bool(acq.get("isUsingAcquisitionBudget",
                                  acq.get("acquisitionBudget"))),
        "teams": [],
        "schedule": [],
    }
    for t in raw.get("teams") or []:
        name = (t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip()
                or f"Team {t.get('id')}")
        rec = (t.get("record") or {}).get("overall") or {}
        counter = t.get("transactionCounter") or {}
        roster = []
        for entry in (t.get("roster") or {}).get("entries") or []:
            player = (entry.get("playerPoolEntry") or {}).get("player") or {}
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
            "division_id": t.get("divisionId"),
            "playoff_seed": _int(t.get("playoffSeed")) or None,
            "waiver_rank": _int(t.get("waiverRank")) or None,
            "wins": _int(rec.get("wins")),
            "losses": _int(rec.get("losses")),
            "ties": _int(rec.get("ties")),
            "points_for": round(_num(rec.get("pointsFor")), 1),
            "points_against": round(_num(rec.get("pointsAgainst")), 1),
            "faab_spent": _int(counter.get("acquisitionBudgetSpent")),
            "moves": _int(counter.get("acquisitions")),
            "roster": roster,
        })

    for m in raw.get("schedule") or []:
        home, away = m.get("home") or {}, m.get("away") or {}
        if home.get("teamId") is None or away.get("teamId") is None:
            continue  # playoff byes and unseeded bracket slots
        mweek = m.get("matchupPeriodId")
        home_pts, home_live = _side(home, mweek)
        away_pts, away_live = _side(away, mweek)
        out["schedule"].append({
            "week": mweek,
            "home_id": home.get("teamId"), "home_pts": home_pts, "home_live": home_live,
            "away_id": away.get("teamId"), "away_pts": away_pts, "away_live": away_live,
            "winner": m.get("winner") or "",
            "tier": m.get("playoffTierType") or "NONE",
        })
    return out


def get_league(cfg: dict) -> dict:
    raw = fetch_league_raw(str(cfg["league_id"]), int(cfg.get("year") or 2026),
                           cfg.get("espn_s2", ""), cfg.get("swid", ""))
    return parse_league(raw)


def get_free_agents(cfg: dict, league: dict) -> list[dict]:
    """Free agents and waiver-wire players, minus anyone on a roster.

    ESPN's status filter alone isn't trustworthy: players picked up since the
    last league pull can still come back as available, and the two requests
    are cached separately. So every entry is checked three ways (its own
    status, its onTeamId, and the league's actual rosters) before it counts
    as available."""
    week = league["week"]
    players = fetch_free_agents(str(cfg["league_id"]), int(cfg.get("year") or 2026),
                                week, cfg.get("espn_s2", ""), cfg.get("swid", ""))
    rostered = {p["espn_id"] for t in league["teams"] for p in t["roster"]}
    out = []
    for entry in players:
        player = entry.get("player") or (entry.get("playerPoolEntry") or {}).get("player") or {}
        if not player or player.get("id") in rostered:
            continue
        if entry.get("onTeamId") not in (None, 0):
            continue
        status = entry.get("status")
        if status and status not in ("FREEAGENT", "WAIVERS"):
            continue
        p = parse_player(player, week)
        p["on_waivers"] = status == "WAIVERS"
        out.append(p)
    return out
