"""End-to-end: every page renders offline against a small synthetic league,
and degrades instead of crashing when a source fails."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import board, espn, fantasypros, market, news, sleeper
from core import outlook as sim

APP = str(Path(__file__).resolve().parent.parent / "draftaid.py")
SEASON_PAGES = ["views/playoffs.py", "views/waivers.py", "views/trades.py", "views/league.py"]
POS_ID = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DST": 16}
SLOT_ID = {"QB": 0, "RB": 2, "WR": 4, "BN": 20}
WEEK = 10


def _player(pid: int, pos: str, slot: str) -> dict:
    return {"lineupSlotId": SLOT_ID[slot], "playerPoolEntry": {"player": {
        "id": pid, "fullName": f"{pos} Player{pid}", "defaultPositionId": POS_ID[pos],
        "proTeamId": 1 + pid % 30, "injuryStatus": "ACTIVE",
        "ownership": {"percentOwned": 60.0, "percentChange": 1.0},
        "stats": [{"scoringPeriodId": WEEK, "statSourceId": 1, "appliedTotal": 8.0 + pid % 9}]}}}


def _raw_league() -> dict:
    """Four teams, QB/RB/WR lineups with two bench spots, nine weeks played."""
    teams, pid = [], 0
    for tid in range(1, 5):
        entries = []
        for pos, slot in (("QB", "QB"), ("RB", "RB"), ("WR", "WR"), ("RB", "BN"), ("WR", "BN")):
            pid += 1
            entries.append(_player(pid, pos, slot))
        teams.append({"id": tid, "name": f"Team {tid}", "owners": [f"{{O{tid}}}"],
                      "record": {"overall": {"wins": 6 - tid, "losses": 3 + tid - 1,
                                             "pointsFor": 900.0 - tid * 20,
                                             "pointsAgainst": 850.0}},
                      "transactionCounter": {"acquisitionBudgetSpent": 10 * tid,
                                             "acquisitions": tid},
                      "roster": {"entries": entries}})
    sched = []
    for wk in range(1, 15):
        for home, away in (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))[wk % 3]:
            done = wk < WEEK
            sched.append({"matchupPeriodId": wk, "playoffTierType": "NONE",
                          "winner": ("HOME" if (wk + home) % 2 else "AWAY") if done
                          else "UNDECIDED",
                          "home": {"teamId": home, "totalPoints": 100.0 + wk if done else 0.0},
                          "away": {"teamId": away, "totalPoints": 95.0 + home if done else 0.0}})
    return {"scoringPeriodId": WEEK, "status": {"currentMatchupPeriod": WEEK},
            "settings": {"name": "Test League",
                         "acquisitionSettings": {"acquisitionBudget": 100,
                                                 "isUsingAcquisitionBudget": True},
                         "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 1, "4": 1, "20": 2}},
                         "scheduleSettings": {"matchupPeriodCount": 14, "playoffTeamCount": 2},
                         "tradeSettings": {"deadlineDate": 4102444800000}},
            "teams": teams, "schedule": sched, "_fetched": 1.0}


def _free_agents() -> list:
    out = []
    for pid, pos in ((101, "RB"), (102, "WR"), (103, "QB"), (104, "RB")):
        out.append({"status": "FREEAGENT", "onTeamId": 0,
                    "player": _player(pid, pos, "BN")["playerPoolEntry"]["player"]})
    return out


def _sources() -> dict:
    """Rankings, ids, market values, and weekly projections for every player."""
    rows = []
    for tid_players in [t["roster"]["entries"] for t in _raw_league()["teams"]]:
        rows += [e["playerPoolEntry"]["player"] for e in tid_players]
    rows += [e["player"] for e in _free_agents()]
    keys, pos, ids = [], [], []
    for p in rows:
        position = {v: k for k, v in POS_ID.items()}[p["defaultPositionId"]]
        keys.append(espn.parse_player(p, WEEK)["key"])
        pos.append(position)
        ids.append(p["id"])
    n = len(rows)
    ros = pd.DataFrame({"key": keys, "ros_rank": [5 + i * 6 for i in range(n)],
                        "ros_tier": [1 + i // 4 for i in range(n)],
                        "ros_pos_rank": [f"{pos[i]}{1 + i // 3}" for i in range(n)],
                        "bye": [11 if i == 1 else 7 for i in range(n)]})
    weekly = pd.DataFrame({"key": keys, "weekly_rank": [1 + i // 3 for i in range(n)],
                           "flex_rank": [1 + i for i in range(n)]})
    players = pd.DataFrame({"sleeper_id": [f"s{i}" for i in ids], "key": keys,
                            "espn_id": ids, "nfl_team": ["DAL"] * n, "injury": [""] * n,
                            "depth_order": [1 + i % 3 for i in range(n)]})
    values = pd.DataFrame({"sleeper_id": [f"s{i}" for i in ids], "mkt_espn_id": ids,
                           "mkt_value": [9000.0 - i * 300 for i in range(n)],
                           "mkt_rank": list(range(1, n + 1)),
                           "mkt_pos_rank": [1 + i // 2 for i in range(n)],
                           "mkt_trend": [(-1) ** i * 120.0 for i in range(n)]})
    weeks = list(range(WEEK, 17))
    proj = {"pts": {f"s{i}": {w: 9.0 + (i + w) % 8 for w in weeks} for i in ids},
            "opp": {f"s{i}": {w: "NYG" for w in weeks} for i in ids},
            "pos": {f"s{ids[i]}": pos[i] for i in range(n)}, "weeks": weeks}
    return {"ros": ros, "weekly": weekly, "players": players, "values": values, "proj": proj}


@pytest.fixture
def offline(monkeypatch):
    """Point every network fetch at synthetic data."""
    src = _sources()
    cfg = {"league_id": "1", "year": 2026, "my_team_id": 2}
    monkeypatch.setattr(espn, "load_config", lambda: dict(cfg))
    monkeypatch.setattr(espn, "save_pref", lambda *a, **k: True)
    monkeypatch.setattr(espn, "fetch_league_raw", lambda *a, **k: _raw_league())
    monkeypatch.setattr(espn, "fetch_free_agents", lambda *a, **k: _free_agents())
    monkeypatch.setattr(fantasypros, "fetch_ros", lambda: src["ros"])
    monkeypatch.setattr(fantasypros, "fetch_weekly", lambda: src["weekly"])
    monkeypatch.setattr(sleeper, "fetch_players", lambda: src["players"])
    monkeypatch.setattr(sleeper, "fetch_trending", lambda *a, **k: {"s101": 500, "s3": 100})
    monkeypatch.setattr(sleeper, "fetch_projections", lambda *a, **k: src["proj"])
    monkeypatch.setattr(market, "fetch_values", lambda *a, **k: src["values"])
    monkeypatch.setattr(news, "fetch_player_news", lambda *a, **k: [])
    return cfg


def _run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    return at


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_my_team_shows_matchup_odds_and_roster(offline):
    at = _run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Team 2" in text and "Playoff odds" in text and "Title odds" in text
    assert "win probability" in text and "Lineup check" in text
    assert "QB Player6" in text                       # my roster made it to the table


@pytest.mark.parametrize("page", SEASON_PAGES)
def test_season_pages_render(offline, page):
    at = _run(page)
    assert not at.exception, at.exception
    assert "da-title" in _text(at)


def test_playoff_picture_has_every_section(offline):
    text = _text(_run("views/playoffs.py"))
    for heading in ("Make the playoffs", "What this week is worth", "Your remaining games",
                    "s odds", "Power rankings", "Your playoff lineup"):
        assert heading in text, heading


def test_trade_finder_prices_deals_and_shows_the_deadline(offline):
    text = _text(_run("views/trades.py"))
    assert "Trade deadline" in text and "Positional strength" in text
    assert "Your title odds" in text


def test_pages_survive_a_failed_simulation(offline, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no odds today")
    monkeypatch.setattr(sim, "simulate", boom)
    at = _run()
    assert not at.exception and "Playoff odds" not in _text(at)       # tiles just drop out
    at.switch_page("views/playoffs.py")
    at.run()
    assert not at.exception and "couldn't run" in _text(at)
    for page in ("views/trades.py", "views/league.py", "views/waivers.py"):
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, at.exception)


def test_missing_projections_and_market_still_render(offline, monkeypatch):
    def down(*a, **k):
        raise RuntimeError("feed down")
    monkeypatch.setattr(sleeper, "fetch_projections", down)
    monkeypatch.setattr(sleeper, "fetch_players", down)
    monkeypatch.setattr(market, "fetch_values", down)
    at = _run()
    assert not at.exception, at.exception
    for page in SEASON_PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, at.exception)


def test_a_dead_rankings_feed_explains_itself(offline, monkeypatch):
    def down():
        raise RuntimeError("FantasyPros 503")
    monkeypatch.setattr(fantasypros, "fetch_ros", down)
    at = _run()
    assert not at.exception
    assert "Rankings unavailable" in _text(at) and "FantasyPros 503" in _text(at)


def test_unconnected_and_unreachable_league_prompt_for_settings(offline, monkeypatch):
    monkeypatch.setattr(espn, "load_config", lambda: {})
    at = _run()
    assert not at.exception and "isn't connected" in _text(at)

    def private(*a, **k):
        raise PermissionError("ESPN says this league is private (401).")
    monkeypatch.setattr(espn, "load_config", lambda: {"league_id": "1"})
    monkeypatch.setattr(espn, "fetch_league_raw", private)
    at = _run()
    assert not at.exception and "private (401)" in _text(at)

    monkeypatch.setattr(espn, "fetch_league_raw", lambda *a, **k: _raw_league())
    at = _run()                                        # connected, but no team chosen
    assert not at.exception and "which team is yours" in _text(at).lower()


def test_settings_switches_theme(offline):
    at = _run("views/setup.py")
    assert not at.exception, at.exception
    assert at.session_state["theme"] == "midnight" and "--bg: #0E1217" in _text(at)
    at.button(key="theme_daylight").click()
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["theme"] == "daylight" and "--bg: #F5F4F0" in _text(at)


def test_saved_theme_is_applied_on_load(offline, monkeypatch):
    monkeypatch.setattr(espn, "load_config", lambda: {**offline, "theme": "dusk"})
    at = _run()
    assert at.session_state["theme"] == "dusk" and "--mark: #9085E9" in _text(at)


def test_draft_room_logs_and_undoes_a_pick(offline, monkeypatch):
    names = ["Alpha Back", "Bravo Wide", "Charlie Quarter", "Delta End"]
    pos = ["RB", "WR", "QB", "TE"]
    frame = pd.DataFrame({
        "player": names, "team": ["DAL", "PHI", "KC", "SF"], "pos": pos,
        "key": [f"{n.lower().replace(' ', '')}|{p}" for n, p in zip(names, pos)],
        "ecr": [1, 2, 3, 4], "ecr_best": [1, 1, 2, 3], "ecr_worst": [3, 4, 6, 8],
        "pos_rank": ["RB1", "WR1", "QB1", "TE1"], "tier": [1, 1, 2, 2], "bye": [7, 9, 10, 14],
        "adp": [1.5, 2.5, 30.0, 40.0], "adp_std": [2.0, 2.0, 6.0, 6.0],
        "adp_pick": ["1.02", "1.03", "3.06", "4.04"], "sleeper_id": ["1", "2", "3", "4"],
        "injury": ["", "", "", ""], "injury_note": ["", "", "", ""], "trending": [0, 0, 0, 0],
        "csv_rank": [None] * 4})
    monkeypatch.setattr(board, "build_board", lambda *a, **k: board.blend(frame))
    at = _run("views/draft_room.py")
    assert not at.exception, at.exception
    taken = [b for b in at.button if (b.key or "").startswith("tk_")]
    assert len(taken) == 4
    taken[0].click()
    at.run()
    assert [p["player"] for p in at.session_state.picks] == ["Alpha Back"]
    at.button(key="btn_undo").click()
    at.run()
    assert at.session_state.picks == []


# ------------------------------------------------------------------ late-season states

def _playoff_league() -> dict:
    """The same league in week 16: semifinals final, title game under way,
    trade deadline long gone."""
    raw = _raw_league()
    raw["scoringPeriodId"] = 16
    raw["status"] = {"currentMatchupPeriod": 16}
    raw["settings"]["scheduleSettings"]["playoffTeamCount"] = 4
    raw["settings"]["tradeSettings"]["deadlineDate"] = 1000
    for m in raw["schedule"]:
        if m["winner"] == "UNDECIDED":
            m["winner"] = "HOME"
            m["home"]["totalPoints"], m["away"]["totalPoints"] = 110.0, 90.0
    for t in raw["teams"]:
        t["playoffSeed"] = t["id"]
    bracket = {"playoffTierType": "WINNERS_BRACKET"}
    raw["schedule"] += [
        {**bracket, "matchupPeriodId": 15, "winner": "HOME",
         "home": {"teamId": 1, "totalPoints": 120.0}, "away": {"teamId": 4, "totalPoints": 80.0}},
        {**bracket, "matchupPeriodId": 15, "winner": "AWAY",
         "home": {"teamId": 2, "totalPoints": 95.0}, "away": {"teamId": 3, "totalPoints": 99.0}},
        {**bracket, "matchupPeriodId": 16, "winner": "UNDECIDED",
         "home": {"teamId": 1, "totalPoints": 0.0, "totalPointsLive": 40.0,
                  "totalProjectedPointsLive": 105.0},
         "away": {"teamId": 3, "totalPoints": 0.0, "totalPointsLive": 52.0,
                  "totalProjectedPointsLive": 101.0}},
        {"playoffTierType": "WINNERS_BRACKET", "matchupPeriodId": 17, "home": {"teamId": 1}},
    ]
    return raw


def test_every_page_works_during_the_playoffs(offline, monkeypatch):
    monkeypatch.setattr(espn, "fetch_league_raw", lambda *a, **k: _playoff_league())
    at = _run()                                   # my team (2) lost its semifinal
    assert not at.exception, at.exception
    for page in SEASON_PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, at.exception)
        if page == "views/trades.py":
            assert "deadline has passed" in _text(at)

    monkeypatch.setattr(espn, "load_config", lambda: {**offline, "my_team_id": 1})
    at = _run()                                   # a finalist, title game in progress
    assert not at.exception, at.exception
    assert "Projected final" in _text(at)
    at.switch_page("views/playoffs.py")
    at.run()
    assert not at.exception, at.exception


def test_playoff_odds_only_go_to_teams_still_alive(offline):
    league = espn.parse_league(_playoff_league())
    model = sim.Model(tids=[1, 2, 3, 4], weeks=[16, 17],
                      mu=__import__("numpy").full((4, 2), 100.0))
    out = sim.simulate(league, model, n=2000, my_id=1)
    assert out.title[2] == 0.0 and out.title[4] == 0.0
    assert out.title[1] + out.title[3] == pytest.approx(1)
    assert out.title[1] > out.title[3]             # ahead on the live projection
    assert out.playoff == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


def test_waiver_priority_league_has_no_dollar_bids(offline, monkeypatch):
    raw = _raw_league()
    raw["settings"]["acquisitionSettings"] = {"isUsingAcquisitionBudget": False}
    monkeypatch.setattr(espn, "fetch_league_raw", lambda *a, **k: raw)
    at = _run()
    assert not at.exception and "FAAB left" not in _text(at)
    at.switch_page("views/waivers.py")
    at.run()
    assert not at.exception, at.exception
    assert "Priority" in _text(at) and "Your FAAB" not in _text(at)
    at.switch_page("views/league.py")
    at.run()
    assert not at.exception, at.exception
