"""Trade finder: deals that fit both rosters, priced on two scales, and
measured by what they would do to your playoff and title odds."""

import math
import time

import pandas as pd
import streamlit as st

from core.trades import POSITIONS, league_weakness, market_movers, trade_ideas
from core.values import player_value
from ui import theme as t
from ui.context import days_to_deadline, header, load

ctx = load()
league, names, mid, roster = ctx.league, ctx.names, ctx.my_id, ctx.roster
out = ctx.outlook()

header(ctx, "Trade Finder",
       "Deals that fill a real need on both rosters and look fair on two "
       "independent scales: FantasyCalc's real-trade market and expert "
       "rest-of-season consensus.")

# --- the clock, and where you stand before any deal
days = days_to_deadline(league)
tiles = []
if days is not None:
    when = time.strftime("%a, %b %d", time.localtime(league["trade_deadline"]))
    if days < 0:
        tiles.append({"k": "Trade deadline", "v": "Passed", "d": when})
    else:
        left = "Today" if days < 1 else f"{math.ceil(days)}<small>day{'s' if days >= 1.5 else ''}</small>"
        tiles.append({"k": "Trade deadline", "v": left, "d": when, "lead": days <= 7})
if out:
    tiles += [{"k": "Your playoff odds", "v": t.fmt_pct(out.playoff[mid]), "bar": out.playoff[mid]},
              {"k": "Your title odds", "v": t.fmt_pct(out.title[mid]), "bar": out.title[mid]}]
if tiles:
    t.tiles(tiles)
closed = days is not None and days < 0
if closed:
    t.alert("", "Closed", "Your league's trade deadline has passed. The waiver wire is "
            "the only way left to improve your roster.")


def _after(idea: dict) -> dict:
    """Both rosters as they'd look after the deal."""
    theirs = ctx.teams[idea["team_id"]]
    give_ids = {g["espn_id"] for g in idea["give"]}
    get_id = idea["get"]["espn_id"]
    bench = {"slot": "BN", "starter": False}
    incoming = theirs[theirs["espn_id"] == get_id].assign(**bench)
    outgoing = roster[roster["espn_id"].isin(give_ids)].assign(**bench)
    return {mid: pd.concat([roster[~roster["espn_id"].isin(give_ids)], incoming],
                           ignore_index=True),
            idea["team_id"]: pd.concat([theirs[theirs["espn_id"] != get_id], outgoing],
                                       ignore_index=True)}


def _odds_shift(idea: dict) -> dict | None:
    """How the deal moves playoff/title odds, on the same simulated seasons."""
    if out is None:
        return None
    memo = ctx._memo.setdefault("trade_odds", {})
    sig = (idea["team_id"], idea["get"]["espn_id"], tuple(g["espn_id"] for g in idea["give"]))
    if sig not in memo:
        alt = ctx.what_if(_after(idea))
        memo[sig] = None if alt is None else {
            "playoff": (out.playoff[mid], alt.playoff[mid]),
            "title": (out.title[mid], alt.title[mid]),
            "their_playoff": (out.playoff[idea["team_id"]], alt.playoff[idea["team_id"]]),
        }
    return memo[sig]


ideas = []
if not closed:
    steals_on = st.toggle("Include steals: offers tilted your way (they'll often "
                          "decline, but asking is free)", value=True)
    with st.spinner("Matching rosters and pricing each deal…"):
        for idea in trade_ideas(ctx.teams, mid, names, top_n=14 if steals_on else 10,
                                include_steals=steals_on):
            idea["odds"] = _odds_shift(idea)
            o = idea["odds"]
            # A deal the simulation says lowers both of your odds isn't a recommendation.
            if o and o["playoff"][1] - o["playoff"][0] < -0.005 and \
                    o["title"][1] - o["title"][0] < -0.003:
                continue
            ideas.append(idea)
    ideas = ideas[:12 if steals_on else 8]

needs_all, weak_all = league_weakness(ctx.teams)
avg = {p: sum(n[p]["starter_avg"] for n in needs_all.values()) / len(needs_all)
       for p in POSITIONS}

if not closed:
    t.section("Trade ideas", f"{len(ideas)} found")
    if not ideas:
        weak = [p for p in POSITIONS if weak_all[mid][p] > 5]
        if not weak:
            t.empty("<b>No fair-trade fits, and here's why:</b> your starters are at or "
                    "above league average at every position, so there's no hole worth "
                    "paying market price to fix. That changes when injuries or busts drop "
                    "a position group below average, or when an opponent gets desperate.")
        else:
            t.empty(f"<b>No fits right now.</b> You're thin at {', '.join(weak)}, but no "
                    "opponent is both deep there <i>and</i> thin where you have spare "
                    "depth, at a price both value sources call fair and that helps your "
                    "odds. Check back after injuries shake up rosters, or hunt manually "
                    "with the strength table below.")


def _pl(p) -> str:
    ranks = t.esc(p.get("ros_pos_rank") or "")
    if t.isnum(p.get("po_rank")):
        ranks += f" <span class='da-dim'>· playoffs {p['pos']}{int(p['po_rank'])}</span>"
    return (f"<div class='pl'>{t.who(p, small=True)}"
            f"<span class='da-num' style='margin-left:auto;font-size:12px;white-space:nowrap'>"
            f"{ranks}</span></div>")


def _shift(label: str, pair: tuple) -> str:
    before, after = pair
    return (f"<span>{label} <b>{t.fmt_pct(before)} → {t.fmt_pct(after)}</b> "
            f"{t.delta((after - before) * 100, ' pts')}</span>")


cards = []
for i in ideas:
    ratio = i["ratio"]
    if i["kind"] == "steal":
        verdict = t.tag("Tilted your way", "accent")
    elif i["kind"] == "1-for-1":
        verdict = (t.tag("Dead even", "info") if 0.97 <= ratio <= 1.07
                   else t.tag("Slight win for you", "good") if ratio < 0.97
                   else t.tag("You pay a little extra", "warn"))
    else:
        verdict = t.tag("Standard 2-for-1 premium", "warn")
    give_v = sum(player_value(g) for g in i["give"])
    get_v = player_value(i["get"]) or 1
    pos = min(100, max(0, (give_v / get_v - 0.5) * 100))
    o = i.get("odds")
    shift = ""
    if o:
        shift = (f"<div class='da-delta'>{_shift('Your playoff odds', o['playoff'])}"
                 f"{_shift('Your title odds', o['title'])}"
                 f"{_shift('Their playoff odds', o['their_playoff'])}</div>")
    helps = o and o["title"][1] > o["title"][0] and o["playoff"][1] >= o["playoff"][0]
    cards.append(
        f"<div class='da-card{' lead' if helps else ''}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;"
        f"flex-wrap:wrap'><div><span class='da-eyebrow' style='display:inline'>"
        f"{t.esc(i['kind'])} with</span> <span class='da-strong' style='font-size:16px'>"
        f"{t.esc(i['team'])}</span></div>{verdict}</div>"
        f"<div class='da-tr'><div class='col get'><div class='h'>YOU GET</div>{_pl(i['get'])}</div>"
        f"<div class='col give'><div class='h'>YOU GIVE</div>"
        f"{''.join(_pl(g) for g in i['give'])}</div></div>"
        f"<div class='da-fair'><div class='track'><i style='left:calc({pos:.0f}% - 2px)'></i></div>"
        f"<div class='lg'><span>Win for you</span><span>Even value</span><span>Overpay</span></div></div>"
        f"{shift}"
        f"<ul class='da-why'><li><b>For you:</b> {t.esc(i['why_me'])}</li>"
        f"<li><b>For them:</b> {t.esc(i['why_them'])}</li></ul></div>")
if cards:
    t.html("".join(cards))

# --- market movers: where the trade market and the experts disagree
movers = market_movers(ctx.teams, mid, names)


def _mover_rows(items: list, owner: bool) -> list[str]:
    rows = []
    for m in items:
        p = m["player"]
        rows.append(
            f"<tr><td>{t.who(p, small=True)}</td>"
            + (f"<td class='hide-m da-dim'>{t.esc(m['team'])}</td>" if owner else "")
            + f"<td class='r'>{t.esc(p['pos'])}{t.fmt_int(p.get('mkt_pos_rank'))}</td>"
              f"<td class='r'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
              f"<td class='r hide-m'>{t.delta(m['trend'], '', 0)}</td></tr>")
    return rows


if movers["sell"] and not closed:
    t.section("Sell high", "Your players the trade market ranks above the experts at "
              "their position, while their value is still climbing")
    t.html(t.table([("Player", ""), ("Market", "r"), ("Experts", "r"),
                    ("30-day value", "r hide-m")], _mover_rows(movers["sell"], False)))
if movers["buy"] and not closed:
    t.section("Buy low", "Rivals' starters the experts rank above a falling market price")
    t.html(t.table([("Player", ""), ("Team", "hide-m"), ("Market", "r"),
                    ("Experts", "r"), ("30-day value", "r hide-m")],
                   _mover_rows(movers["buy"], True)))

# --- league positional strength
t.section("Positional strength", "Average rest-of-season rank of each team's starters "
          "(lower is better) · blue = stronger than the league, red = weaker")
rows = []
for tid in sorted(ctx.teams, key=lambda k: k != mid):
    n = needs_all[tid]
    cells = "".join(
        t.heat(f"{n[p]['starter_avg']:.0f}",
               max(-1.0, min(1.0, (avg[p] - n[p]["starter_avg"]) / max(avg[p], 1))))
        for p in POSITIONS)
    rows.append(f"<tr class='{'me' if tid == mid else ''}'>"
                f"<td><span class='da-strong'>{t.esc(names[tid])}</span></td>{cells}"
                f"<td class='c da-dim'>{sum(x['depth'] for x in n.values())}</td></tr>")
t.html(t.table([("Team", "")] + [(p, "c") for p in POSITIONS] + [("Spare starters", "c")],
               rows))
