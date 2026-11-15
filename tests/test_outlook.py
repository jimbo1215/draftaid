"""Season simulator: bracket math, the strength model, and odds invariants."""

import numpy as np
import pytest

from core import outlook as ol
from tests.factory import frame, game, league, player, round_robin, team

IDS = list(range(1, 13))


def flat_model(tids, weeks, mu=100.0, sigma=24.0, **over):
    arr = np.full((len(tids), len(weeks)), mu, dtype=float)
    for tid, value in over.get("teams", {}).items():
        arr[tids.index(tid)] = value
    return ol.Model(tids=list(tids), weeks=list(weeks), mu=arr, sigma=sigma)


def midseason(wins=None, playoff_teams=6, **extra):
    """12 teams after nine weeks, with weeks 10-14 still to play."""
    wins = wins or {tid: 4 + (tid % 3) for tid in IDS}
    teams = [team(tid, wins=wins[tid], losses=9 - wins[tid], pf=900.0 + tid) for tid in IDS]
    sched = [game(w, h, a) for w, h, a in round_robin(IDS, range(10, 15))]
    return league(teams, sched, week=10, playoff_teams=playoff_teams, **extra)


# ------------------------------------------------------------------ structure

def test_bracket_order_and_field():
    assert ol.bracket_order(4) == [1, 4, 2, 3]
    assert ol.bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert ol.playoff_field(midseason()) == (6, 8, 3)
    assert ol.playoff_field(midseason(playoff_teams=4)) == (4, 4, 2)
    assert ol.playoff_field(midseason(playoff_teams=0)) == (6, 8, 3)       # sensible default
    assert ol.playoff_field(league([team(1), team(2)], [], playoff_teams=0))[0] == 2


def test_period_and_playoff_weeks():
    lg = midseason()
    assert ol.period_weeks(lg, 12) == [12] and ol.period_weeks(lg, 16) == [16]
    assert ol.playoff_weeks(lg) == [15, 16, 17]
    two_week = midseason(playoff_teams=4, playoff_round_weeks=2, regular_season_weeks=13)
    assert ol.playoff_weeks(two_week) == [14, 15, 16, 17]
    mapped = midseason(matchup_weeks={15: [15], 16: [16, 17]})
    assert ol.period_weeks(mapped, 16) == [16, 17]


def test_all_play_strips_schedule_luck():
    teams = [team(1), team(2), team(3)]
    sched = [game(1, 1, 2, 120, 110, "HOME"), game(1, 3, 3, 90, 90, "TIE"),
             game(2, 2, 3, 100, 100, "TIE"), game(2, 1, 1, 80, 80, "TIE"),
             game(3, 1, 2)]                                            # undecided: ignored
    rec = ol.all_play(league(teams, sched))
    assert (rec[1]["w"], rec[1]["l"]) == (2.0, 2.0)                    # 2-0 then 0-2
    assert (rec[2]["w"], rec[2]["l"]) == (2.5, 1.5)                    # 1-1, then a win and a tie
    assert (rec[3]["w"], rec[3]["l"]) == (1.5, 2.5)
    assert rec[1]["pct"] == 0.5


# ------------------------------------------------------------------ model

def test_project_teams_uses_each_weeks_best_lineup():
    roster = frame([player("A", "RB", slot="RB", wk={11: 14.0, 12: 0.0}),
                    player("B", "RB", wk={11: 9.0, 12: 11.0})])
    assert ol.project_teams({7: roster}, {"RB": 1}, [11, 12]) == {7: {11: 14.0, 12: 11.0}}


def test_model_without_history_is_the_raw_projection():
    lg = league([team(1), team(2)], [])
    m = ol.build_model(lg, {1: {10: 110.0}, 2: {10: 90.0}}, [10])
    assert m.mu.tolist() == [[110.0], [90.0]] and m.sigma == ol.DEFAULT_SIGMA
    assert m.col(10).tolist() == [110.0, 90.0]
    assert m.col(99).tolist() == [110.0, 90.0]                         # outside the window


def test_model_rescales_to_the_league_and_leans_on_history():
    teams = [team(1), team(2)]
    sched = []
    for wk in range(1, 9):                       # team 1 really scores 130, team 2 scores 90
        sched.append(game(wk, 1, 2, 130 + (wk % 2) * 10, 90 - (wk % 2) * 10, "HOME"))
    proj = {1: {9: 100.0, 10: 100.0}, 2: {9: 100.0, 10: 100.0}}       # feed says they're equal
    m = ol.build_model(league(teams, sched, week=9), proj, [9, 10])
    assert m.ppg[1] == 135.0 and m.ppg[2] == 85.0
    # League scoring (110) is above the feed (100): rescaled, then pulled toward history.
    assert m.mu[0, 0] == pytest.approx(110 + 0.32 * (135 - 110))
    assert m.mu[1, 0] == pytest.approx(110 + 0.32 * (85 - 110))
    assert 16.0 <= m.sigma <= 34.0
    assert m.strength()[1] > m.strength()[2]


# ------------------------------------------------------------------ simulation

def test_odds_are_consistent_probabilities():
    lg = midseason()
    out = ol.simulate(lg, flat_model(IDS, range(10, 18)), n=4000, my_id=8)
    assert sum(out.playoff.values()) == pytest.approx(6)
    assert sum(out.bye.values()) == pytest.approx(2)
    assert sum(out.finals.values()) == pytest.approx(2)
    assert sum(out.title.values()) == pytest.approx(1)
    assert all(sum(dist) == pytest.approx(1) for dist in out.seed_dist.values())
    base_wins = sum(t["wins"] for t in lg["teams"])
    assert sum(out.exp_wins.values()) == pytest.approx(base_wins + 5 * 6)
    assert out.games_left == 5 and (out.playoff_teams, out.byes) == (6, 2)
    # My remaining games, and what this week's result is worth.
    assert sorted(out.week_win) == [10, 11, 12, 13, 14]
    assert out.leverage["period"] == 10
    assert out.leverage["playoff_win"] > out.leverage["playoff_loss"]
    assert 0.4 < out.leverage["p_win"] < 0.6


def test_a_dominant_team_and_an_eliminated_team():
    wins = {tid: 8 for tid in IDS}
    wins[12] = 0                                    # 0-9: five wins won't catch seven teams
    lg = midseason(wins=wins)
    model = flat_model(IDS, range(10, 18), teams={1: 200.0})
    out = ol.simulate(lg, model, n=3000)
    assert out.playoff[1] == 1.0 and out.title[1] > 0.9
    assert out.playoff[12] == 0.0 and out.title[12] == 0.0
    assert out.leverage is None                     # no team of mine was given


def test_same_seed_is_reproducible_and_prices_a_what_if():
    lg = midseason()
    base = flat_model(IDS, range(10, 18))
    a = ol.simulate(lg, base, n=3000, my_id=8)
    b = ol.simulate(lg, base, n=3000, my_id=8)
    assert a.playoff == b.playoff and a.title == b.title
    better = flat_model(IDS, range(10, 18), teams={8: 115.0})
    c = ol.simulate(lg, better, n=3000, my_id=8)
    assert c.playoff[8] > a.playoff[8] and c.title[8] > a.title[8]
    assert c.exp_wins[8] > a.exp_wins[8]


def test_four_team_field_has_no_byes():
    out = ol.simulate(midseason(playoff_teams=4), flat_model(IDS, range(10, 18)), n=2000)
    assert out.byes == 0 and sum(out.playoff.values()) == pytest.approx(4)
    assert all(v == 0.0 for v in out.bye.values())
    assert sum(out.finals.values()) == pytest.approx(2)


def test_live_scores_all_but_settle_the_current_week():
    lg = midseason()
    for m in lg["schedule"]:
        if m["week"] == 10 and 8 in (m["home_id"], m["away_id"]):
            mine = "home" if m["home_id"] == 8 else "away"
            theirs = "away" if mine == "home" else "home"
            m[f"{mine}_pts"], m[f"{mine}_live"] = 148.0, 150.0      # nearly done, way ahead
            m[f"{theirs}_pts"], m[f"{theirs}_live"] = 88.0, 90.0
    out = ol.simulate(lg, flat_model(IDS, range(10, 18)), n=2000, my_id=8)
    assert out.week_win[10][1] > 0.99


def test_division_winners_are_seeded_first():
    teams = [team(1, wins=10, division_id=0), team(2, wins=9, division_id=0),
             team(3, wins=3, division_id=1), team(4, wins=2, division_id=1)]
    lg = league(teams, [], week=15, playoff_teams=2)
    out = ol.simulate(lg, flat_model([1, 2, 3, 4], [15, 16]), n=500)
    assert out.playoff == {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}


def test_playoffs_in_progress_lock_finished_rounds():
    teams = [team(tid, wins=13 - tid, losses=tid, pf=2000.0 - tid, playoff_seed=tid)
             for tid in IDS]
    sched = [game(15, 4, 5, 90, 120, "AWAY", tier="WINNERS_BRACKET"),      # 5 upsets 4
             game(15, 3, 6, 130, 100, "HOME", tier="WINNERS_BRACKET"),
             game(15, 7, 8, 99, 98, "HOME", tier="LOSERS_CONSOLATION_LADDER"),
             game(16, 1, 5, tier="WINNERS_BRACKET"), game(16, 2, 3, tier="WINNERS_BRACKET")]
    lg = league(teams, sched, week=16)
    out = ol.simulate(lg, flat_model(IDS, [16, 17]), n=4000)
    assert all(out.playoff[tid] == 1.0 for tid in range(1, 7))
    assert all(out.playoff[tid] == 0.0 for tid in range(7, 13))
    alive = {1, 2, 3, 5}
    assert all(out.title[tid] == 0.0 for tid in IDS if tid not in alive)
    assert sum(out.title[tid] for tid in alive) == pytest.approx(1)
    assert all(0.15 < out.title[tid] < 0.35 for tid in alive)             # even teams
    assert out.games_left == 0 and out.week_win == {}


def test_power_rankings_and_schedule_strength():
    teams = [team(1, wins=2, losses=0), team(2, wins=0, losses=2), team(3, wins=1, losses=1),
             team(4, wins=1, losses=1)]
    sched = [game(1, 1, 2, 100, 99, "HOME"), game(1, 3, 4, 140, 60, "HOME"),
             game(2, 1, 4, 90, 60, "HOME"), game(2, 2, 3, 130, 140, "AWAY"),
             game(3, 1, 3), game(3, 2, 4)]
    lg = league(teams, sched, week=3)
    # Team 3's record is wrong for the scenario above, but the point is luck:
    model = flat_model([1, 2, 3, 4], [3, 4], teams={3: 140.0, 1: 100.0, 2: 115.0, 4: 60.0})
    table = ol.power_rankings(lg, model)
    assert list(table["team_id"]) == [3, 2, 1, 4] and list(table["rank"]) == [1, 2, 3, 4]
    row = table.set_index("team_id")
    assert row.loc[1, "luck"] > 0 > row.loc[2, "luck"]       # 2-0 on middling scores
    sos = ol.remaining_sos(lg, model)
    assert sos == {1: 140.0, 3: 100.0, 2: 60.0, 4: 115.0}
