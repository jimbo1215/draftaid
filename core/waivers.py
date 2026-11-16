"""Waiver wire: roster-aware add/drop recommendations and FAAB bids."""

import pandas as pd

from core.lineup import DEFAULT_LINEUP, as_players, best_lineup, holding, week_points
from core.names import FLEX_POSITIONS


# Playoff positional rank worth calling out: a real starter at the position.
PLAYOFF_STARTER = {"QB": 10, "TE": 10, "K": 6, "DST": 6}


def _bid_pct(ros_gain: float) -> float:
    """Share of the interested rivals' budgets a ROS lineup gain is worth.
    Convex: a lineup-changing starter (gain ~40) ~55%, a solid upgrade
    (~20) ~25%, depth (~10) ~12%, a marginal bump (~3) ~3%."""
    if ros_gain <= 0:
        return 0.0
    return min(0.6, 0.55 * (ros_gain / 40.0) ** 1.1)


def _rival_floors(teams: dict, my_id, lineup: dict) -> dict:
    """Each rival's weakest ROS starter by position: who would start a pickup."""
    out = {}
    for tid, df in teams.items():
        if tid == my_id or df.empty:
            continue
        players = as_players(holding(df))
        _, starters = best_lineup(players, lineup, "ros_v", bench=False)
        floor = {}
        for p in players:
            if p["id"] in starters:
                floor[p["pos"]] = min(floor.get(p["pos"], 1e9), p["ros_v"])
        out[tid] = floor
    return out


def waiver_targets(fas: pd.DataFrame, my_roster: pd.DataFrame, league: dict,
                   my_id: int, teams: dict[int, pd.DataFrame],
                   top_n: int = 20) -> pd.DataFrame:
    """Rank free agents by what they do for YOUR lineup, pair each with the
    right drop, and size a FAAB bid to what it takes to beat the rivals who
    would actually want him."""
    if fas.empty:
        return fas
    lineup = league.get("lineup") or DEFAULT_LINEUP
    budget = league.get("faab_budget") or 0
    uses_faab = league.get("uses_faab", False)
    min_bid = league.get("min_bid", 0)
    faab_left = {t["team_id"]: max(0, budget - t["faab_spent"]) for t in league["teams"]}
    my_left = faab_left.get(my_id, 0)

    # Season clock: unspent FAAB is worth less every week, so the same pickup
    # justifies a bigger share of the budget later on.
    weeks = max(1, league.get("regular_season_weeks") or 14)
    progress = min(1.0, max(0.0, (league.get("week", 1) - 1) / weeks))
    season_mult = 1.0 + 0.6 * progress

    mine = as_players(holding(my_roster))
    bench_slots = league.get("bench_slots") or 0
    open_spot = bench_slots > 0 and len(mine) < sum(lineup.values()) + bench_slots
    base_ros, _ = best_lineup(mine, lineup, "ros_v")
    base_wk, _ = best_lineup(mine, lineup, "wk_v", bench=False)
    rival_floor = _rival_floors(teams, my_id, lineup)

    pool = fas[fas["pos"].isin({"QB", "RB", "WR", "TE", "K", "DST"})].copy()
    pool = pool[pool["ros_rank"].notna() | pool["weekly_rank"].notna()
                | (pool["trend_rank"].fillna(999) <= 50)]
    # Keep the math small: best 120 by ROS, plus anyone projected to start.
    pool = pool.sort_values("ros_rank", na_position="last")
    keep = pool.head(120).index.union(pool[pool["week_proj"].fillna(0) >= 8].index)
    pool = pool.loc[keep]

    rows = []
    for idx, fa in pool.iterrows():
        cand = as_players(pd.DataFrame([fa]))[0]
        cand["id"] = ("fa", idx)
        best = None
        for d in ([None] if open_spot else []) + mine:
            if d is not None and ("K" in (d["pos"], cand["pos"]) or "DST" in (
                    d["pos"], cand["pos"])) and cand["pos"] != d["pos"]:
                continue  # kickers and defenses are straight swaps
            after = [q for q in mine if q is not d] + [cand]
            r_after, r_start = best_lineup(after, lineup, "ros_v")
            w_after, w_start = best_lineup(after, lineup, "wk_v", bench=False)
            ros_gain, wk_gain = r_after - base_ros, w_after - base_wk
            # This week's points count, but one week is a small slice of
            # what's left of the season.
            score = ros_gain + 0.5 * wk_gain
            if best is None or score > best["score"]:
                best = {"drop": d, "ros_gain": ros_gain, "wk_gain": wk_gain,
                        "score": score, "starts_ros": cand["id"] in r_start,
                        "starts_wk": cand["id"] in w_start}
        if best is None or best["score"] <= 0.5:
            continue

        # --- competition: rivals who'd put him in their starting lineup
        interested = []
        for tid, floor in rival_floor.items():
            slots = [fl for pos, fl in floor.items()
                     if pos == cand["pos"] or (cand["pos"] in FLEX_POSITIONS
                                               and pos in FLEX_POSITIONS)]
            if cand["ros_v"] > (min(slots) if slots else 0) + 2:
                interested.append(tid)
        crowd = len(interested)
        rich = max((faab_left.get(t, 0) for t in interested), default=0)
        ref = sum(faab_left.get(t, 0) for t in interested) / crowd if crowd else 0

        # --- FAAB bid: a share of what the interested rivals can spend
        pct = _bid_pct(best["ros_gain"])
        if best["wk_gain"] > 0:  # streamers: a little extra for a real bump
            pct += min(0.03, best["wk_gain"] * 0.003)
        heat = 1.0
        trend = fa.get("trend_rank")
        if pd.notna(trend) and trend <= 10:
            heat += 0.25
        elif pd.notna(trend) and trend <= 25:
            heat += 0.12
        chg = fa.get("own_change")
        if chg is not None and pd.notna(chg) and chg >= 10:
            heat += 0.1
        heat *= 0.6 if crowd == 0 else 1.0 if crowd <= 2 else 1.15 if crowd <= 5 else 1.3
        pool_ref = ref if crowd else budget * 0.25
        bid = pct * season_mult * heat * pool_ref
        if crowd:
            bid = min(bid, rich + 1)  # never need more than the richest rival has
        if cand["pos"] in ("K", "DST"):
            bid = min(bid, 1)  # streamers: never pay real money
        bid = int(round(min(max(bid, min_bid), my_left)))
        lo = min(bid, max(min_bid, int(round(bid * 0.8))))
        hi = min(my_left, max(bid, int(round(bid * 1.25))))

        # --- tag + reasons
        if best["ros_gain"] >= 25:
            tag = "Priority add"
        elif best["ros_gain"] >= 8:
            tag = "Solid add"
        elif best["wk_gain"] >= 2 and best["ros_gain"] < 4:
            tag = "Stream"
        else:
            tag = "Depth"
        why = []
        if best["starts_ros"]:
            why.append("cracks your rest-of-season starting lineup")
        if best["starts_wk"] and best["wk_gain"] >= 1:
            why.append(f"+{best['wk_gain']:.1f} projected points in your lineup this week")
        po = fa.get("po_rank")
        if po is not None and pd.notna(po) and po <= PLAYOFF_STARTER.get(cand["pos"], 24):
            why.append(f"{cand['pos']}{int(po)} by projection for the fantasy playoffs")
        if pd.notna(trend) and trend <= 25:
            why.append(f"#{int(trend)} most-added on Sleeper today")
        if crowd:
            who = f"{crowd} rival{'s' if crowd > 1 else ''} would start him"
            why.append(f"{who} (most FAAB among them: ${rich})" if uses_faab else who)
        elif best["ros_gain"] > 0:
            why.append("no rival needs him, so bid low")

        if uses_faab:
            claim = None
        elif best["ros_gain"] >= 25:
            claim = "Worth your top waiver priority"
        elif fa.get("on_waivers"):
            claim = "Claim him, but don't burn a high priority"
        else:
            claim = "Free agent, just add him"

        rows.append({**fa.to_dict(),
                     "drop": best["drop"]["row"] if best["drop"] is not None else None,
                     "ros_gain": best["ros_gain"], "wk_gain": best["wk_gain"],
                     "score": best["score"], "bid": bid, "bid_lo": lo, "bid_hi": hi,
                     "interested": crowd, "rich_rival": rich, "tag": tag,
                     "why": why, "claim": claim})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def handcuffs(my_roster: pd.DataFrame, fas: pd.DataFrame, max_rank: int = 60) -> list[dict]:
    """Available backups to the running backs you rely on. If your starter
    goes down in December, his direct backup is the most valuable player on
    the wire, and by then everyone bids on him.

    Returns [{"starter": row, "backup": row}], best starter first."""
    if my_roster.empty or fas.empty or "depth_order" not in fas.columns:
        return []
    studs = my_roster[(my_roster["pos"] == "RB") & (my_roster["ros_rank"] <= max_rank)
                      & (my_roster["team"] != "FA")].sort_values("ros_rank")
    out = []
    for _, stud in studs.iterrows():
        backups = fas[(fas["pos"] == "RB") & (fas["team"] == stud["team"])
                      & fas["depth_order"].notna()]
        if backups.empty:
            continue
        nxt = backups.sort_values(["depth_order", "ros_rank"], na_position="last").iloc[0]
        if nxt["depth_order"] <= 3:
            out.append({"starter": stud, "backup": nxt})
    return out


def playoff_stashes(fas: pd.DataFrame, my_roster: pd.DataFrame, lineup: dict,
                    playoff_weeks: list[int], top_n: int = 6) -> pd.DataFrame:
    """Free agents who would raise your projected lineup total over the
    fantasy-playoff weeks. `po_gain` is points added across those weeks versus
    your current roster (0 = he wouldn't crack a playoff lineup)."""
    if fas.empty or not playoff_weeks or "wk" not in fas.columns:
        return pd.DataFrame()
    lineup = lineup or DEFAULT_LINEUP
    base = sum(week_points(my_roster, lineup, w) for w in playoff_weeks)
    cands = fas[fas["po_pts"].fillna(0) > 0].sort_values("po_pts", ascending=False)
    # The best few at each position cover every lineup slot he could fill.
    cands = cands.groupby("pos", group_keys=False).head(6)
    rows = []
    for _, fa in cands.iterrows():
        row = fa.copy()
        row["slot"], row["starter"] = "BN", False
        trial = pd.concat([my_roster, pd.DataFrame([row])], ignore_index=True)
        gain = sum(week_points(trial, lineup, w) for w in playoff_weeks) - base
        if gain >= 1.0:
            rows.append({**fa.to_dict(), "po_gain": round(gain, 1)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("po_gain", ascending=False)
            .head(top_n).reset_index(drop=True))
