"""Lineup model, start/sit, and waiver recommendations."""

import pandas as pd
import pytest

from core import lineup as lu
from core.startsit import bench_points, bye_watch, start_sit
from core.waivers import handcuffs, playoff_stashes, waiver_targets
from tests.factory import frame, league, player, team

LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}


def my_roster():
    return frame([
        player("Quinn Back", "QB", ros=30, proj=20.0, slot="QB", weekly_rank=5),
        player("Ron One", "RB", "DET", ros=5, proj=18.0, slot="RB", flex_rank=4, weekly_rank=3),
        player("Ron Two", "RB", ros=40, proj=12.0, slot="RB", flex_rank=40, weekly_rank=20),
        player("Wes One", "WR", ros=8, proj=17.0, slot="WR", flex_rank=6, weekly_rank=4),
        player("Wes Two", "WR", ros=35, proj=13.0, slot="WR", flex_rank=30, weekly_rank=18),
        player("Ty End", "TE", ros=70, proj=9.0, slot="TE", weekly_rank=8, flex_rank=80),
        player("Flex Guy", "WR", ros=75, proj=10.0, slot="FLEX", flex_rank=60, weekly_rank=38),
        player("Kick Er", "K", ros=200, proj=8.0, slot="K", weekly_rank=10),
        player("Bears D/ST", "DST", "CHI", ros=210, proj=6.0, slot="DST", weekly_rank=14),
        player("Bench Back", "RB", ros=90, proj=9.0, flex_rank=70, weekly_rank=40),
        player("Bench Wide", "WR", ros=150, proj=6.0, flex_rank=95, weekly_rank=60),
        player("Hurt Guy", "RB", ros=20, proj=0.0, slot="IR"),
    ])


# ------------------------------------------------------------------ lineup

def test_fill_lineup_uses_dedicated_slots_then_flex():
    players = [{"id": i, "pos": pos, "v": v} for i, (pos, v) in enumerate(
        [("RB", 20), ("RB", 15), ("RB", 14), ("WR", 12), ("WR", 9), ("WR", 8), ("TE", 5)])]
    slots = lu.fill_lineup(players, {"RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, "v")
    assert [(s, p["id"]) for s, p in slots] == [
        ("RB", 0), ("RB", 1), ("WR", 3), ("WR", 4), ("TE", 6), ("FLEX", 2)]


def test_best_lineup_counts_bench_at_a_discount():
    players = [{"id": 1, "pos": "RB", "v": 50.0}, {"id": 2, "pos": "RB", "v": 30.0},
               {"id": 3, "pos": "QB", "v": 40.0}, {"id": 4, "pos": "QB", "v": 20.0}]
    total, used = lu.best_lineup(players, {"QB": 1, "RB": 1}, "v", bench=False)
    assert (total, used) == (90.0, {1, 3})
    with_bench, _ = lu.best_lineup(players, {"QB": 1, "RB": 1}, "v")
    assert with_bench == pytest.approx(90 + 0.30 * 30 + 0.08 * 20)


def test_week_points_handles_byes_ir_and_missing_feeds():
    roster = frame([
        player("A", "RB", slot="RB", proj=15.0, wk={11: 14.0, 12: 0.0}),
        player("B", "RB", slot="RB", proj=11.0, wk={11: 10.0, 12: 12.0}),
        player("C", "RB", proj=9.0, wk={11: 8.0, 12: 9.0}),
        player("D", "RB", slot="IR", proj=0.0, wk={11: 30.0, 12: 30.0}),
        player("E", "RB", proj=7.0, bye=12),                     # no weekly feed
    ])
    two_rb = {"RB": 2}
    assert lu.week_points(roster, two_rb, 11) == 24.0             # IR never starts
    assert lu.week_points(roster, two_rb, 12) == 21.0             # A on bye, C steps in
    # E has no feed: ESPN's number stands in, but not on his bye week.
    assert lu.week_value(roster.iloc[4], 11) == 7.0
    assert lu.week_value(roster.iloc[4], 12) == 0.0
    # A covered player falls back to ESPN only for the current week.
    assert lu.week_value(roster.iloc[0], 13, current_week=13) == 15.0
    assert lu.week_value(roster.iloc[0], 13, current_week=10) == 0.0


def test_drop_costs_puts_the_free_cut_first():
    costs = lu.drop_costs(my_roster(), LINEUP)
    assert "Hurt Guy" not in set(costs["player"])                 # IR isn't a roster spot
    assert costs.iloc[0]["player"] == "Bench Wide"
    assert costs.iloc[-1]["player"] in ("Ron One", "Wes One")
    assert (costs["drop_cost"] >= -1e-9).all()


def test_positional_needs_counts_startable_depth():
    needs = lu.positional_needs(my_roster())
    assert needs["RB"]["starter_avg"] == (5 + 20) / 2
    assert needs["RB"]["depth"] == 2                               # ROS 40 and 90
    assert needs["TE"] == {"starter_avg": 70, "depth": 0, "count": 1}


# ------------------------------------------------------------------ start / sit

def test_start_sit_flags_only_legal_swaps():
    roster = my_roster()
    assert start_sit(roster, LINEUP) == []
    # A bench WR the experts love should replace the weakest flex-eligible starter...
    roster.loc[roster["player"] == "Bench Wide", ["flex_rank", "weekly_rank"]] = [20, 12]
    flags = start_sit(roster, LINEUP)
    assert len(flags) == 1
    assert (flags[0]["in"]["player"], flags[0]["out"]["player"]) == ("Bench Wide", "Flex Guy")
    assert flags[0]["note"] == "WR12 vs WR38"
    # ...but never the tight end, whatever their ranks say.
    assert all(f["out"]["pos"] != "TE" for f in flags)


def test_start_sit_flags_unranked_starters():
    roster = my_roster()
    roster.loc[roster["player"] == "Ron Two", ["flex_rank", "weekly_rank"]] = [None, None]
    flags = start_sit(roster, LINEUP)
    empties = [f for f in flags if f["in"] is None]
    assert [f["out"]["player"] for f in empties] == ["Ron Two"]
    assert empties[0]["note"] == "not ranked this week (ruled out?)"
    roster.loc[roster["player"] == "Ron Two", "bye"] = 10
    assert start_sit(roster, LINEUP, week=10)[-1]["note"] == "on bye this week"


def test_bench_points_measures_what_is_left_on_the_bench():
    roster = my_roster()
    assert bench_points(roster, LINEUP)["gap"] == 0.0
    roster.loc[roster["player"] == "Bench Back", "week_proj"] = 16.0
    out = bench_points(roster, LINEUP)
    assert out["gap"] == 6.0 and out["best"] == out["set"] + 6.0   # over Flex Guy's 10


def test_bye_watch_lists_core_starters_on_bye():
    roster = my_roster()
    roster.loc[roster["player"].isin(["Ron One", "Wes One"]), "bye"] = 12
    roster.loc[roster["player"] == "Bench Wide", "bye"] = 12       # not a starter: ignored
    out = bye_watch(roster, LINEUP, [11, 12, 13])
    assert [w["week"] for w in out] == [12]
    assert {p["player"] for p in out[0]["players"]} == {"Ron One", "Wes One"}


# ------------------------------------------------------------------ waivers

def _league(my_spent=20, rival_spent=60):
    # Two bench slots: my_roster() is exactly full, so every add needs a drop.
    return league([team(8, "Me", faab_spent=my_spent), team(3, "Rival", faab_spent=rival_spent)],
                  [], week=10, bench_slots=2)


def test_waiver_add_needs_no_drop_with_an_open_roster_spot():
    fas = frame([player("Wire Star", "RB", ros=28, proj=14.0)])
    roomy = {**_league(), "bench_slots": 5}
    out = waiver_targets(fas, my_roster(), roomy, 8, {8: my_roster(), 3: rival_roster()})
    assert out.iloc[0]["player"] == "Wire Star" and out.iloc[0]["drop"] is None


def rival_roster():
    return frame([
        player("R QB", "QB", ros=25, slot="QB"), player("R RB1", "RB", ros=10, slot="RB"),
        player("R RB2", "RB", ros=95, slot="RB"), player("R WR1", "WR", ros=15, slot="WR"),
        player("R WR2", "WR", ros=45, slot="WR"), player("R TE", "TE", ros=80, slot="TE"),
        player("R FLEX", "WR", ros=85, slot="FLEX"), player("R K", "K", ros=205, slot="K"),
        player("R DST", "DST", "SF", ros=215, slot="DST"),
    ])


def test_waiver_target_pairs_the_add_with_the_right_drop():
    fas = frame([
        player("Wire Star", "RB", ros=28, proj=14.0, trend_rank=3),
        player("Wire Scrub", "WR", ros=260, proj=2.0),
        player("Wire Kicker", "K", ros=198, proj=9.5),
    ])
    lg = _league()
    out = waiver_targets(fas, my_roster(), lg, 8, {8: my_roster(), 3: rival_roster()})
    names = list(out["player"])
    assert names[0] == "Wire Star" and "Wire Scrub" not in names
    star = out.iloc[0]
    assert star["drop"]["player"] == "Bench Wide"                  # cheapest cut
    assert star["tag"] in ("Solid add", "Priority add")
    assert any("most-added" in w for w in star["why"])
    assert any("would start him" in w for w in star["why"])
    # Never more than the richest interested rival + $1, or than I have left.
    assert 0 < star["bid"] <= 41 and star["bid_lo"] <= star["bid"] <= star["bid_hi"] <= 80
    kicker = out[out["player"] == "Wire Kicker"].iloc[0]
    assert kicker["drop"]["pos"] == "K" and kicker["bid"] <= 1     # straight swap, no money


def test_waiver_bids_respect_my_budget_and_priority_leagues():
    fas = frame([player("Wire Star", "RB", ros=6, proj=19.0)])
    broke = _league(my_spent=97)
    out = waiver_targets(fas, my_roster(), broke, 8, {8: my_roster(), 3: rival_roster()})
    assert out.iloc[0]["bid"] <= 3 and out.iloc[0]["tag"] == "Priority add"
    priority = {**_league(), "uses_faab": False, "faab_budget": 0}
    out = waiver_targets(fas, my_roster(), priority, 8, {8: my_roster(), 3: rival_roster()})
    assert out.iloc[0]["claim"] == "Worth your top waiver priority"
    assert waiver_targets(pd.DataFrame(), my_roster(), broke, 8, {}).empty


def test_handcuffs_find_the_direct_backup_on_the_wire():
    fas = frame([
        player("Det Backup", "RB", "DET", ros=140, depth_order=2),
        player("Det Third", "RB", "DET", ros=260, depth_order=3),
        player("Other Back", "RB", "NYG", ros=100, depth_order=2),
        player("Det Wideout", "WR", "DET", ros=90, depth_order=2),
    ])
    out = handcuffs(my_roster(), fas)
    assert [(h["starter"]["player"], h["backup"]["player"]) for h in out] == [
        ("Ron One", "Det Backup")]
    assert handcuffs(my_roster(), frame([player("X", "RB", "DET", ros=99)])) == []


def test_playoff_stashes_rank_by_points_added_to_playoff_lineups():
    weeks = [15, 16, 17]
    roster = frame([
        player("My RB1", "RB", slot="RB", wk={15: 15.0, 16: 15.0, 17: 15.0}),
        player("My RB2", "RB", slot="RB", wk={15: 8.0, 16: 8.0, 17: 0.0}),
    ])
    fas = frame([
        player("Playoff Riser", "RB", wk={15: 13.0, 16: 12.0, 17: 14.0}, po_pts=39.0),
        player("No Help", "RB", wk={15: 5.0, 16: 5.0, 17: 0.5}, po_pts=10.5),
        player("No Feed", "RB", po_pts=None),
    ])
    out = playoff_stashes(fas, roster, {"RB": 2}, weeks)
    assert list(out["player"]) == ["Playoff Riser"]
    assert out.iloc[0]["po_gain"] == (13 - 8) + (12 - 8) + 14
    assert playoff_stashes(fas, roster, {"RB": 2}, []).empty
