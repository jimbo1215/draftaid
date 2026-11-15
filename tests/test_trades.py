"""Trade finder: fairness windows, the two-scale guard, steals, market movers."""

from core.trades import (FAIR_1FOR1, FAIR_2FOR1, STEAL, league_weakness, market_movers,
                         trade_ideas)
from core.values import player_value, rank_value
from tests.factory import frame, player

NAMES = {8: "Me", 3: "Scab", 5: "Avg"}


def teams():
    """Me: thin at RB, deep at WR. Scab: the mirror image. Avg: neutral."""
    me = frame([
        player("My QB", "QB", ros=20), player("My RB1", "RB", ros=5),
        player("My RB2", "RB", ros=170), player("My WR1", "WR", ros=8),
        player("My WR2", "WR", ros=12), player("My WR3", "WR", ros=30),
        player("My WR4", "WR", ros=45), player("My WR5", "WR", ros=55),
        player("My TE", "TE", ros=60),
    ])
    scab = frame([
        player("S QB", "QB", ros=25), player("S RB1", "RB", ros=4),
        player("S RB2", "RB", ros=15), player("S RB3", "RB", ros=28),
        player("S RB4", "RB", ros=60), player("S WR1", "WR", ros=120),
        player("S WR2", "WR", ros=140), player("S TE", "TE", ros=65),
    ])
    avg = frame([
        player("A QB", "QB", ros=22), player("A RB1", "RB", ros=30),
        player("A RB2", "RB", ros=50), player("A WR1", "WR", ros=31),
        player("A WR2", "WR", ros=51), player("A TE", "TE", ros=62),
    ])
    return {8: me, 3: scab, 5: avg}


def test_value_curve_is_steep_at_the_top():
    assert rank_value(1) > 115 and 80 < rank_value(25) < 90 and rank_value(None) == 0.0
    assert rank_value(10) - rank_value(20) > 4 * (rank_value(110) - rank_value(120))
    assert player_value({"mkt_value": 9000.0, "ros_rank": 200}) == 100.0   # market first
    assert player_value({"mkt_value": None, "ros_rank": 1}) == rank_value(1)


def test_weakness_is_relative_to_the_league():
    _, weak = league_weakness(teams())
    assert weak[8]["RB"] > 30 and weak[8]["WR"] < 0
    assert weak[3]["WR"] > 30 and weak[3]["RB"] < 0


def test_fair_ideas_target_my_need_and_stay_inside_the_windows():
    ideas = trade_ideas(teams(), 8, NAMES)
    assert ideas and all(i["team"] == "Scab" and i["team_id"] == 3 for i in ideas)
    assert len(ideas) <= 2                                   # two per opposing team
    for i in ideas:
        assert i["get"]["pos"] == "RB" and all(g["pos"] == "WR" for g in i["give"])
        lo, hi = FAIR_1FOR1 if i["kind"] == "1-for-1" else FAIR_2FOR1
        assert lo <= i["ratio"] <= hi
        assert i["why_me"] and i["why_them"]
    assert {i["kind"] for i in ideas} <= {"1-for-1", "2-for-1"}


def test_expert_scale_blocks_a_deal_the_market_calls_even():
    t = teams()
    # The market prices both players at 80, but the experts have my receiver
    # far ahead of their back: giving him up would be an overpay.
    t[3].loc[t[3]["player"] == "S RB4", "mkt_value"] = 7200.0
    t[8].loc[t[8]["player"] == "My WR3", "mkt_value"] = 7200.0
    pairs = {(i["get"]["player"], tuple(g["player"] for g in i["give"]))
             for i in trade_ideas(t, 8, NAMES, top_n=50)}
    assert ("S RB4", ("My WR3",)) not in pairs


def test_steals_are_opt_in_and_get_their_own_slot():
    assert all(i["kind"] != "steal" for i in trade_ideas(teams(), 8, NAMES))
    ideas = trade_ideas(teams(), 8, NAMES, include_steals=True)
    steals = [i for i in ideas if i["kind"] == "steal"]
    assert len(steals) == 1 and len(ideas) == 3               # two fair + one steal
    assert STEAL[0] <= steals[0]["ratio"] < STEAL[1]
    assert steals[0]["get"]["player"] == "S RB3"
    assert [g["player"] for g in steals[0]["give"]] == ["My WR5"]   # cheapest piece that works


def test_no_ideas_when_nothing_would_improve_the_lineup():
    t = teams()
    t[8].loc[t[8]["player"] == "My RB2", "ros_rank"] = 9      # no hole left to fill
    assert trade_ideas(t, 8, NAMES) == []
    assert trade_ideas(t, 8, NAMES, include_steals=True) == []
    assert trade_ideas({8: t[8]}, 8, NAMES) == []             # nobody to trade with


def test_market_movers_split_into_sell_high_and_buy_low():
    t = teams()
    t[8].loc[t[8]["player"] == "My WR4", ["mkt_rank", "mkt_trend"]] = [20, 300.0]
    t[8].loc[t[8]["player"] == "My WR5", ["mkt_rank", "mkt_trend"]] = [40, -50.0]   # falling
    t[3].loc[t[3]["player"] == "S RB3", ["mkt_rank", "mkt_trend"]] = [55, -200.0]
    t[5].loc[t[5]["player"] == "A RB1", ["mkt_rank", "mkt_trend"]] = [31, -400.0]   # no gap
    movers = market_movers(t, 8, NAMES)
    assert [m["player"]["player"] for m in movers["sell"]] == ["My WR4"]
    assert [(m["player"]["player"], m["team"]) for m in movers["buy"]] == [("S RB3", "Scab")]
    assert movers["sell"][0]["gap"] == 25 and movers["buy"][0]["gap"] == 27
