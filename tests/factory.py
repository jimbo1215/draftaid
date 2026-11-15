"""Small builders for test data: enriched roster rows and parsed leagues."""

import itertools

import pandas as pd

from core.names import player_key

_ids = itertools.count(1000)


def player(name: str, pos: str, team: str = "DAL", *, ros=None, proj=None, slot="BN",
           wk=None, **extra) -> dict:
    """One enriched roster row. `ros` is overall rest-of-season rank, `proj`
    this week's ESPN projection, `wk` a {week: points} projection dict."""
    row = {
        "espn_id": next(_ids), "player": name, "pos": pos, "team": team,
        "key": player_key(name, pos, team), "slot": slot,
        "starter": slot not in ("BN", "IR"),
        "ros_rank": ros, "ros_pos_rank": f"{pos}{ros}" if ros else None,
        "weekly_rank": None, "flex_rank": None, "week_proj": proj, "week_pts": None,
        "bye": 0, "injury": "", "sleeper_id": None, "depth_order": None,
        "trend_rank": None, "own_change": None, "pct_owned": None, "on_waivers": False,
        "mkt_value": None, "mkt_rank": None, "mkt_trend": None,
        "wk": wk or {}, "wk_opp": {}, "po_pts": None, "po_rank": None,
    }
    row.update(extra)
    return row


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def team(team_id: int, name: str = "", *, wins=0, losses=0, ties=0, pf=0.0, pa=0.0,
         faab_spent=0, division_id=0, playoff_seed=None) -> dict:
    return {"team_id": team_id, "name": name or f"Team {team_id}", "owners": [],
            "abbrev": "", "division_id": division_id, "playoff_seed": playoff_seed,
            "waiver_rank": None, "wins": wins, "losses": losses, "ties": ties,
            "points_for": pf, "points_against": pa, "faab_spent": faab_spent,
            "moves": 0, "roster": []}


def game(week: int, home: int, away: int, home_pts=0.0, away_pts=0.0, winner="UNDECIDED",
         tier="NONE", home_live=None, away_live=None) -> dict:
    return {"week": week, "home_id": home, "away_id": away, "home_pts": home_pts,
            "away_pts": away_pts, "winner": winner, "tier": tier,
            "home_live": home_live, "away_live": away_live}


def league(teams: list[dict], schedule: list[dict], *, week=10, reg_weeks=14,
           playoff_teams=6, **extra) -> dict:
    out = {
        "league_name": "Test League", "week": week, "matchup_period": week,
        "final_week": 17, "fetched": None,
        "lineup": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
        "bench_slots": 5, "regular_season_weeks": reg_weeks,
        "playoff_teams": playoff_teams, "playoff_round_weeks": 1, "seeding_rule": "",
        "matchup_weeks": {}, "trade_deadline": None, "faab_budget": 100, "min_bid": 0,
        "uses_faab": True, "teams": teams, "schedule": schedule,
    }
    out.update(extra)
    return out


def round_robin(team_ids: list[int], weeks: range) -> list[tuple[int, int, int]]:
    """(week, home, away) pairings: a rotating round robin over `weeks`."""
    ids = list(team_ids)
    n = len(ids)
    out = []
    for i, wk in enumerate(weeks):
        k = i % (n - 1)
        rot = [ids[0]] + ids[1:][k:] + ids[1:][:k]
        for j in range(n // 2):
            out.append((wk, rot[j], rot[n - 1 - j]))
    return out
