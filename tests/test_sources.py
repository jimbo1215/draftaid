"""Parsing and joining: names, Sleeper projections, market values, ESPN payloads."""

import pandas as pd

from core import espn, market, sleeper
from core.names import normalize_name, player_key
from core.players import Sources, enrich


def test_names_join_across_spellings():
    assert normalize_name("Marvin Harrison Jr.") == "marvinharrison"
    assert normalize_name("Ja'Marr Chase") == normalize_name("JaMarr Chase")
    assert normalize_name(None) == ""
    assert player_key("Kenneth Walker III", "RB", "SEA") == "kennethwalker|RB"
    # Defenses key by team, whatever the source calls them.
    assert player_key("Jaguars D/ST", "DST", "JAC") == player_key("Jacksonville", "DST", "JAX")


def _proj(pid, pos, pts, opp="NYG"):
    return {"player_id": pid, "opponent": opp, "stats": {"pts_ppr": pts},
            "player": {"position": pos}}


def test_clean_week_replaces_glitched_projection_with_median():
    items = [_proj("k1", "K", 75.2), _proj("k2", "K", 9.0), _proj("k3", "K", 8.0),
             _proj("k4", "K", 10.0), _proj("GB", "DEF", 7.9, "MIA"),
             _proj("p1", "P", 3.0), _proj("r1", "RB", None)]
    out = sleeper.clean_week(items)
    assert out["k1"] == ("K", 9.0, "NYG")          # season total in a weekly row
    assert out["k2"][1] == 9.0
    assert out["GB"] == ("DST", 7.9, "MIA")        # DEF maps to DST
    assert "p1" not in out and "r1" not in out     # non-fantasy / no projection


def test_fetch_projections_skips_weeks_that_fail(monkeypatch):
    def fake(season, week):
        if week == 12:
            raise RuntimeError("boom")
        return {"a": ("RB", 10.0 + week, "DAL"), "b": ("WR", 5.0, "")}
    monkeypatch.setattr(sleeper, "fetch_week_projections", fake)
    out = sleeper.fetch_projections(2026, (11, 12, 13))
    assert out["weeks"] == [11, 13]
    assert out["pts"]["a"] == {11: 21.0, 13: 23.0}
    assert out["opp"]["a"] == {11: "DAL", 13: "DAL"} and "b" not in out["opp"]
    assert out["pos"] == {"a": "RB", "b": "WR"}


def test_market_parse_keeps_trend_and_skips_unmapped():
    df = market.parse_values([
        {"player": {"sleeperId": 9221, "espnId": "4429795"}, "value": 10681,
         "overallRank": 1, "trend30Day": 74},
        {"player": {"sleeperId": None}, "value": 500, "overallRank": 90},
    ])
    assert list(df["sleeper_id"]) == ["9221"]
    assert df.iloc[0]["mkt_value"] == 10681 and df.iloc[0]["mkt_trend"] == 74
    assert df.iloc[0]["mkt_espn_id"] == 4429795
    assert market.parse_values([]).empty


def test_parse_players_keeps_suffixed_and_injured_players():
    raw = {
        "8151": {"full_name": "Kenneth Walker III", "search_full_name": "kennethwalkeriii",
                 "position": "RB", "status": "Active", "team": "KC", "active": True,
                 "depth_chart_order": 1},
        "4634": {"full_name": "Kenneth Walker", "position": "RB", "status": "Active",
                 "team": None, "active": True},
        "9493": {"full_name": "Puka Nacua", "position": "WR", "status": "Inactive",
                 "team": "LAR", "active": True, "injury_status": "Out", "espn_id": "4426515"},
        "111": {"full_name": "Long Retired", "position": "WR", "status": "Inactive",
                "team": None, "active": False},
        "222": {"full_name": "Some Lineman", "position": "OT", "team": "DAL", "active": True},
        "JAC": {"position": "DEF"},
    }
    df = sleeper.parse_players(raw).set_index("key")
    # "III" is stripped so he joins FantasyPros/ESPN; the rostered namesake wins.
    assert df.loc["kennethwalker|RB", "sleeper_id"] == "8151"
    assert df.loc["kennethwalker|RB", "depth_order"] == 1
    # "Inactive" means hurt, not gone.
    assert df.loc["pukanacua|WR", "injury"] == "Out"
    assert df.loc["pukanacua|WR", "espn_id"] == 4426515
    assert "longretired|WR" not in df.index and "somelineman|OT" not in df.index
    assert df.loc["dstjax", "sleeper_id"] == "JAC"
    assert sleeper.parse_players({}).empty


# ------------------------------------------------------------------ ESPN

def _espn_player(pid, name, pos_id, team_id, slot, proj=12.0, pts=None, inj="ACTIVE"):
    stats = [{"scoringPeriodId": 10, "statSourceId": 1, "appliedTotal": proj},
             {"scoringPeriodId": 0, "statSourceId": 0, "appliedTotal": 100.0}]
    if pts is not None:
        stats.append({"scoringPeriodId": 10, "statSourceId": 0, "appliedTotal": pts})
    return {"lineupSlotId": slot, "playerPoolEntry": {"player": {
        "id": pid, "fullName": name, "defaultPositionId": pos_id, "proTeamId": team_id,
        "injuryStatus": inj, "ownership": {"percentOwned": 88.0, "percentChange": 2.5},
        "stats": stats}}}


RAW = {
    "scoringPeriodId": 10,
    "status": {"currentMatchupPeriod": 10, "finalScoringPeriod": 17},
    "settings": {
        "name": "The League",
        "acquisitionSettings": {"acquisitionBudget": 100, "isUsingAcquisitionBudget": True,
                                "minimumBid": 1},
        "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1,
                                                "16": 1, "17": 1, "20": 5, "21": 1, "1": 0}},
        "scheduleSettings": {"matchupPeriodCount": 14, "playoffTeamCount": 6,
                             "playoffMatchupPeriodLength": 1,
                             "matchupPeriods": {"15": [15], "16": [16, 17]}},
        "tradeSettings": {"deadlineDate": 1795600000000},
    },
    "teams": [
        {"id": 8, "name": "Jimmy", "owners": ["{abc}"], "divisionId": 0, "playoffSeed": 3,
         "record": {"overall": {"wins": 6, "losses": 3, "pointsFor": 1100.5,
                                "pointsAgainst": 1000.25}},
         "transactionCounter": {"acquisitionBudgetSpent": 30, "acquisitions": 7},
         "roster": {"entries": [
             _espn_player(1, "Josh Allen", 1, 2, 0, proj=24.0, pts=31.2),
             _espn_player(2, "Chris Olave", 3, 18, 20, inj="QUESTIONABLE"),
             _espn_player(-16002, "Bills D/ST", 16, 2, 16),
             {"lineupSlotId": 20, "playerPoolEntry": {}},
         ]}},
        {"id": 3, "location": "Team", "nickname": "Scab",
         "record": {"overall": {"wins": 2, "losses": 7}}},
    ],
    "schedule": [
        {"matchupPeriodId": 9, "winner": "HOME",
         "home": {"teamId": 8, "totalPoints": 120.4}, "away": {"teamId": 3, "totalPoints": 99.1}},
        {"matchupPeriodId": 10, "winner": "UNDECIDED",
         "home": {"teamId": 3, "totalPoints": 0, "totalPointsLive": 41.5,
                  "totalProjectedPointsLive": 104.2},
         "away": {"teamId": 8, "totalPoints": 0, "pointsByScoringPeriod": {"10": 55.0}}},
        {"matchupPeriodId": 15, "playoffTierType": "WINNERS_BRACKET", "home": {"teamId": 8}},
    ],
}


def test_parse_league_reads_settings_and_playoff_structure():
    lg = espn.parse_league(RAW)
    assert lg["league_name"] == "The League" and lg["week"] == 10
    assert lg["lineup"] == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
    assert lg["bench_slots"] == 5 and lg["regular_season_weeks"] == 14
    assert lg["playoff_teams"] == 6 and lg["matchup_weeks"][16] == [16, 17]
    assert lg["trade_deadline"] == 1795600000.0
    assert lg["uses_faab"] and lg["faab_budget"] == 100 and lg["min_bid"] == 1


def test_parse_league_teams_rosters_and_schedule():
    lg = espn.parse_league(RAW)
    me, other = lg["teams"]
    assert (me["wins"], me["losses"], me["faab_spent"], me["playoff_seed"]) == (6, 3, 30, 3)
    assert other["name"] == "Team Scab" and other["roster"] == []
    allen, olave, dst = me["roster"]
    assert allen["starter"] and allen["slot"] == "QB"
    assert allen["week_proj"] == 24.0 and allen["week_pts"] == 31.2
    assert not olave["starter"] and olave["week_pts"] is None
    assert dst["key"] == "dstbuf" and dst["pos"] == "DST"
    # The bye-only playoff entry is dropped; live scores use the best field present.
    assert [m["week"] for m in lg["schedule"]] == [9, 10]
    live = lg["schedule"][1]
    assert (live["home_pts"], live["home_live"]) == (41.5, 104.2)
    assert live["away_pts"] == 55.0 and live["away_live"] is None
    assert lg["schedule"][0]["tier"] == "NONE"


def test_parse_league_survives_an_empty_payload():
    lg = espn.parse_league({})
    assert lg["teams"] == [] and lg["schedule"] == [] and lg["week"] == 1
    assert lg["regular_season_weeks"] == 14 and lg["trade_deadline"] is None


def test_free_agents_exclude_anyone_rostered(monkeypatch):
    lg = espn.parse_league(RAW)
    entries = [
        {"status": "FREEAGENT", "onTeamId": 0, "player": {"id": 50, "fullName": "A B",
                                                          "defaultPositionId": 2}},
        {"status": "WAIVERS", "onTeamId": 0, "player": {"id": 51, "fullName": "C D",
                                                        "defaultPositionId": 3}},
        {"status": "ONTEAM", "onTeamId": 0, "player": {"id": 52, "fullName": "E F"}},
        {"status": "FREEAGENT", "onTeamId": 4, "player": {"id": 53, "fullName": "G H"}},
        {"status": "FREEAGENT", "onTeamId": 0, "player": {"id": 2, "fullName": "Chris Olave"}},
    ]
    monkeypatch.setattr(espn, "fetch_free_agents", lambda *a, **k: entries)
    out = espn.get_free_agents({"league_id": "1"}, lg)
    assert [p["espn_id"] for p in out] == [50, 51]
    assert [p["on_waivers"] for p in out] == [False, True]


# ------------------------------------------------------------------ enrich

def _sources(**over):
    ros = pd.DataFrame({"key": ["joshallen|QB", "chrisolave|WR"], "ros_rank": [12, 60],
                        "ros_tier": [2, 6], "ros_pos_rank": ["QB2", "WR25"], "bye": [7, 11]})
    weekly = pd.DataFrame({"key": ["joshallen|QB"], "weekly_rank": [3], "flex_rank": [None]})
    sl = pd.DataFrame({"sleeper_id": ["4984", "8144", "BUF"],
                       "key": ["joshallen|QB", "christopherolave|WR", "dstbuf"],
                       "espn_id": [1, 2, None], "injury": ["", "Questionable", ""],
                       "depth_order": [1, 2, None]})
    mkt = pd.DataFrame({"sleeper_id": ["4984"], "mkt_espn_id": [1], "mkt_value": [7000.0],
                        "mkt_rank": [9], "mkt_trend": [-120.0]})
    base = dict(ros=ros, weekly=weekly, sleeper=sl, trending={"8144": 900, "4984": 50},
                market=mkt,
                proj={"4984": {15: 22.0, 16: 20.0, 17: 25.0}, "8144": {15: 11.0}, "BUF": {15: 8.0}},
                opp={"4984": {15: "CHI"}}, proj_pos={"4984": "QB", "8144": "WR", "BUF": "DST"},
                weeks=[15, 16, 17], playoff_weeks=[15, 16, 17])
    base.update(over)
    return Sources(**base)


def test_enrich_joins_by_espn_id_when_names_differ():
    players = espn.parse_league(RAW)["teams"][0]["roster"]
    df = enrich(players, _sources()).set_index("player")
    # Sleeper spells him "Christopher Olave": only the ESPN id links them.
    assert df.loc["Chris Olave", "sleeper_id"] == "8144"
    assert df.loc["Chris Olave", "injury"] == "Questionable"
    assert df.loc["Chris Olave", "trend_rank"] == 1
    assert df.loc["Bills D/ST", "sleeper_id"] == "BUF"          # by team key
    assert df.loc["Josh Allen", "mkt_value"] == 7000.0
    assert df.loc["Josh Allen", "ros_rank"] == 12 and df.loc["Josh Allen", "weekly_rank"] == 3


def test_enrich_bridges_ids_through_the_market_feed():
    players = espn.parse_league(RAW)["teams"][0]["roster"]
    src = _sources()
    src.sleeper = src.sleeper.assign(espn_id=None)          # Sleeper has no ESPN ids...
    src.market = pd.DataFrame({"sleeper_id": ["8144"], "mkt_espn_id": [2],
                               "mkt_value": [4100.0], "mkt_rank": [40], "mkt_trend": [15.0]})
    df = enrich(players, src).set_index("player")
    assert df.loc["Chris Olave", "sleeper_id"] == "8144"    # ...but the market feed does
    assert df.loc["Chris Olave", "mkt_value"] == 4100.0
    assert "mkt_espn_id" not in df.columns


def test_enrich_attaches_playoff_window_projections():
    players = espn.parse_league(RAW)["teams"][0]["roster"]
    df = enrich(players, _sources()).set_index("player")
    assert df.loc["Josh Allen", "wk"] == {15: 22.0, 16: 20.0, 17: 25.0}
    assert df.loc["Josh Allen", "po_pts"] == 67.0 and df.loc["Josh Allen", "po_rank"] == 1
    assert df.loc["Chris Olave", "po_pts"] == 11.0
    assert df.loc["Josh Allen", "wk_opp"] == {15: "CHI"}


def test_enrich_works_with_only_rankings():
    players = espn.parse_league(RAW)["teams"][0]["roster"]
    src = Sources(ros=_sources().ros, weekly=pd.DataFrame())
    df = enrich(players, src).set_index("player")
    assert df.loc["Josh Allen", "ros_rank"] == 12
    assert df.loc["Chris Olave", "injury"] == "Questionable"      # ESPN's status
    assert df.loc["Josh Allen", "wk"] == {} and pd.isna(df.loc["Josh Allen", "po_rank"])
    assert enrich([], src).empty
