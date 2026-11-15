"""The lineup model: who starts, what a roster is worth, what a cut costs.

Everything downstream (waivers, start/sit, the season simulation) asks the
same question in different units, "what is the best lineup this roster can
field?", so the slot-filling logic lives here once.
"""

import pandas as pd

from core.values import rank_value

DEFAULT_LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
# Which positions can fill each flexible slot.
FLEX_ELIGIBLE = {"FLEX": ("RB", "WR", "TE"), "RB/WR": ("RB", "WR"),
                 "OP": ("QB", "RB", "WR", "TE")}
# ROS overall rank a player must beat to be a plausible starter in 12-team PPR.
STARTABLE_ROS = {"QB": 130, "RB": 110, "WR": 110, "TE": 140, "K": 999, "DST": 999}
# Bench depth still matters (byes, injuries), just far less than starters:
# weights for the best leftover RB/WR/TE, plus one backup QB and TE.
BENCH_FLEX_WEIGHTS = (0.30, 0.22, 0.15, 0.10)
BENCH_EXTRA_WEIGHTS = {"QB": 0.08, "TE": 0.05}


def fits(pos: str, slot: str) -> bool:
    return slot == pos or pos in FLEX_ELIGIBLE.get(slot, ())


def fill_lineup(players: list[dict], lineup: dict, col: str) -> list[tuple[str, dict]]:
    """Best (slot, player) assignment by `col`: dedicated slots first, then
    flexible slots from whoever is left."""
    pool = sorted((p for p in players if p[col] > 0), key=lambda p: -p[col])
    used, out = set(), []
    for flexible in (False, True):
        for slot, n in lineup.items():
            if (slot in FLEX_ELIGIBLE) != flexible:
                continue
            picks = [p for p in pool if fits(p["pos"], slot) and p["id"] not in used][:n]
            for p in picks:
                used.add(p["id"])
                out.append((slot, p))
    return out


def best_lineup(players: list[dict], lineup: dict, col: str,
                bench: bool = True) -> tuple[float, set]:
    """(value, starter ids) of the best lineup. With `bench`, leftover depth
    adds a discounted share of its value."""
    starters = fill_lineup(players, lineup, col)
    used = {p["id"] for _, p in starters}
    total = sum(p[col] for _, p in starters)
    if bench:
        rest = sorted((p for p in players if p["id"] not in used and p[col] > 0),
                      key=lambda p: -p[col])
        flex = [p for p in rest if p["pos"] in ("RB", "WR", "TE")]
        total += sum(w * p[col] for w, p in zip(BENCH_FLEX_WEIGHTS, flex))
        for pos, w in BENCH_EXTRA_WEIGHTS.items():
            backup = next((p for p in rest if p["pos"] == pos), None)
            if backup:
                total += w * backup[col]
    return total, used


def holding(roster: pd.DataFrame) -> pd.DataFrame:
    """Players occupying a roster spot (IR slots don't count)."""
    if roster.empty or "slot" not in roster.columns:
        return roster
    return roster[roster["slot"] != "IR"]


def as_players(df: pd.DataFrame) -> list[dict]:
    """Roster rows -> light dicts: ROS value plus this week's ESPN projection."""
    out = []
    for i, r in df.iterrows():
        proj, pid = r.get("week_proj"), r.get("espn_id")
        out.append({"id": pid if pid is not None and pd.notna(pid) else ("row", i),
                    "pos": r["pos"], "row": r,
                    "ros_v": rank_value(r.get("ros_rank")),
                    "wk_v": float(proj) if proj is not None and pd.notna(proj) else 0.0})
    return out


def week_value(row, week: int, current_week: int | None = None) -> float:
    """A player's projected points for one NFL week: the week-by-week feed
    when it covers him, else ESPN's projection (current week only)."""
    wk = row.get("wk")
    if isinstance(wk, dict) and week in wk:
        return float(wk[week])
    if week == current_week or not isinstance(wk, dict) or not wk:
        if row.get("bye") == week:
            return 0.0
        proj = row.get("week_proj")
        if proj is not None and pd.notna(proj):
            return float(proj)
    return 0.0


def week_lineup(roster: pd.DataFrame, lineup: dict, week: int,
                current_week: int | None = None) -> list[tuple[str, dict]]:
    """Best lineup for one NFL week as (slot, player) with `pts` on each player."""
    players = []
    for i, r in holding(roster).iterrows():
        pid = r.get("espn_id")
        players.append({"id": pid if pid is not None and pd.notna(pid) else ("row", i),
                        "pos": r["pos"], "row": r,
                        "pts": week_value(r, week, current_week)})
    return fill_lineup(players, lineup or DEFAULT_LINEUP, "pts")


def week_points(roster: pd.DataFrame, lineup: dict, week: int,
                current_week: int | None = None) -> float:
    """Projected points of the best lineup for one NFL week."""
    return sum(p["pts"] for _, p in week_lineup(roster, lineup, week, current_week))


def positional_needs(roster: pd.DataFrame) -> dict:
    """Per position: average ROS rank of the starters (higher = weaker) and
    how many startable players sit behind them."""
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        ranks = roster[roster["pos"] == pos]["ros_rank"].dropna().sort_values().tolist()
        n_start = DEFAULT_LINEUP.get(pos, 1)
        starters = ranks[:n_start] or [250]
        out[pos] = {"starter_avg": sum(starters) / len(starters),
                    "depth": sum(1 for r in ranks[n_start:] if r <= STARTABLE_ROS[pos]),
                    "count": int((roster["pos"] == pos).sum())}
    return out


def drop_costs(roster: pd.DataFrame, lineup: dict | None = None) -> pd.DataFrame:
    """Every droppable player with the lineup value lost by cutting him
    (cheapest cut first)."""
    if roster.empty:
        return roster
    lineup = lineup or DEFAULT_LINEUP
    players = as_players(holding(roster))
    base, _ = best_lineup(players, lineup, "ros_v")
    idx, vals = [], []
    for p in players:
        without = [q for q in players if q is not p]
        idx.append(p["row"].name)
        vals.append(base - best_lineup(without, lineup, "ros_v")[0])
    out = roster.loc[idx].copy()
    out["drop_cost"] = vals
    return out.sort_values("drop_cost")
