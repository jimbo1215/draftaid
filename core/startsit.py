"""Start/sit: lineup swaps the experts would make, and points left on the bench."""

import pandas as pd

from core.lineup import (DEFAULT_LINEUP, as_players, best_lineup, fill_lineup, fits,
                         holding)
from core.names import FLEX_POSITIONS


def _week_rank(r):
    """This week's expert rank on the scale that compares across his slot:
    FLEX ranks for RB/WR/TE, positional ranks for QB/K/DST."""
    return r.get("flex_rank") if r["pos"] in FLEX_POSITIONS else r.get("weekly_rank")


def start_sit(my_roster: pd.DataFrame, lineup: dict | None = None,
              week: int | None = None) -> list[dict]:
    """Bench players the experts rank above a starter they could replace.

    Builds the best lineup from this week's FantasyPros ranks and flags only
    swaps the slot rules allow: a WR never "replaces" a TE sitting in the
    TE slot. Each flag is {"in": row | None, "out": row, "note": str}."""
    if my_roster.empty or "starter" not in my_roster.columns:
        return []
    lineup = lineup or DEFAULT_LINEUP
    active = holding(my_roster)
    players = []
    for i, r in active.iterrows():
        rank = _week_rank(r)
        if rank is None or pd.isna(rank):
            continue  # unranked this week: bye, out, or irrelevant
        pid = r.get("espn_id")
        players.append({"id": pid if pid is not None and pd.notna(pid) else ("row", i),
                        "pos": r["pos"], "row": r, "rank": float(rank),
                        "ss_v": 400.0 - float(rank)})
    _, best = best_lineup(players, lineup, "ss_v", bench=False)
    ins = sorted((p for p in players if p["id"] in best and not p["row"]["starter"]),
                 key=lambda p: p["rank"])
    outs = [p for p in players if p["row"]["starter"] and p["id"] not in best]

    flags = []
    for b in ins:
        cand = [o for o in outs if fits(b["pos"], o["row"]["slot"])]
        if not cand:
            continue
        o = max(cand, key=lambda o: o["rank"])  # replace the weakest option
        if o["pos"] == b["pos"]:  # same position: its own ranks are sharper
            scale, br, orank = b["pos"], b["row"].get("weekly_rank"), o["row"].get("weekly_rank")
        else:
            scale, br, orank = "FLEX", b["rank"], o["rank"]
        if pd.isna(br) or pd.isna(orank) or orank - br <= 2:
            continue
        outs.remove(o)
        flags.append({"in": b["row"], "out": o["row"],
                      "note": f"{scale}{int(br)} vs {scale}{int(orank)}"})
    # Starters with no rank at all this week (bye / ruled out) are their own flag.
    for _, r in active[active["starter"]].iterrows():
        rank = _week_rank(r)
        if (rank is None or pd.isna(rank)) and r["pos"] in ("QB", "RB", "WR", "TE"):
            on_bye = week is not None and r.get("bye") == week
            flags.append({"in": None, "out": r,
                          "note": "on bye this week" if on_bye
                          else "not ranked this week (ruled out?)"})
    return flags


def bench_points(my_roster: pd.DataFrame, lineup: dict | None = None) -> dict:
    """Projected points of the lineup you've set versus the best one available
    (ESPN projections, your league's scoring).
    Returns {"set": float, "best": float, "gap": float}."""
    if my_roster.empty or "starter" not in my_roster.columns:
        return {"set": 0.0, "best": 0.0, "gap": 0.0}
    active = holding(my_roster)
    players = as_players(active)
    current = sum(p["wk_v"] for p in players if p["row"]["starter"])
    best = sum(p["wk_v"] for _, p in fill_lineup(players, lineup or DEFAULT_LINEUP, "wk_v"))
    return {"set": round(current, 1), "best": round(best, 1),
            "gap": round(max(0.0, best - current), 1)}


def bye_watch(my_roster: pd.DataFrame, lineup: dict | None, weeks: list[int]) -> list[dict]:
    """Upcoming weeks where players in your best lineup are on bye.
    Returns [{"week": int, "players": [row, ...]}] for weeks with any."""
    if my_roster.empty:
        return []
    active = holding(my_roster)
    players = as_players(active)
    _, core = best_lineup(players, lineup or DEFAULT_LINEUP, "ros_v", bench=False)
    regulars = [p["row"] for p in players if p["id"] in core]
    out = []
    for wk in weeks:
        off = [r for r in regulars if r.get("bye") == wk]
        if off:
            out.append({"week": wk, "players": off})
    return out
