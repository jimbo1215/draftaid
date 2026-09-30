"""Season-long logic: waiver targets with FAAB bids, trade ideas, start/sit."""

import math

import pandas as pd

STARTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
FLEX_POSITIONS = {"RB", "WR", "TE"}
# ROS overall rank a player must beat to be a plausible starter in 12-team PPR.
STARTABLE_ROS = {"QB": 130, "RB": 110, "WR": 110, "TE": 140, "K": 999, "DST": 999}


def enrich(players: list[dict], ros: pd.DataFrame, weekly: pd.DataFrame,
           sleeper: pd.DataFrame | None = None, trending: dict | None = None,
           market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join ESPN players with FantasyPros ROS/weekly ranks, Sleeper data, and
    FantasyCalc market values."""
    df = pd.DataFrame(players)
    if df.empty:
        return df
    df = df.merge(ros[["key", "ros_rank", "ros_tier", "ros_pos_rank", "bye"]],
                  on="key", how="left")
    if not weekly.empty:
        df = df.merge(weekly, on="key", how="left")
    else:
        df["weekly_rank"], df["flex_rank"] = pd.NA, pd.NA
    if sleeper is not None and not sleeper.empty:
        df = df.merge(sleeper[["key", "sleeper_id", "injury"]], on="key", how="left")
        df["injury"] = df["injury"].fillna("")
        if trending:
            df["trending"] = df["sleeper_id"].map(trending).fillna(0).astype(int)
            # Raw add counts swing by orders of magnitude (millions on waiver
            # day), so rank within today's list is the stable signal.
            order = {sid: i + 1 for i, sid in enumerate(
                sorted(trending, key=lambda k: -trending[k]))}
            df["trend_rank"] = df["sleeper_id"].map(order)
        else:
            df["trending"], df["trend_rank"] = 0, pd.NA
    else:
        df["sleeper_id"], df["injury"], df["trending"] = None, "", 0
        df["trend_rank"] = pd.NA
    if (market is not None and not market.empty
            and df["sleeper_id"].notna().any()):
        df = df.merge(market, on="sleeper_id", how="left")
    else:
        df["mkt_value"], df["mkt_rank"] = pd.NA, pd.NA
    df["injury"] = df.apply(
        lambda r: r["injury"] or (r.get("espn_injury") or "").replace("_", " ").title()
        if (r.get("espn_injury") or "") not in ("", "ACTIVE") else r["injury"], axis=1)
    return df


def positional_needs(roster: pd.DataFrame) -> dict:
    """How weak each position group is: avg ROS rank of the startable pieces
    (higher = weaker), plus depth count beyond starters."""
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        grp = roster[roster["pos"] == pos].sort_values("ros_rank")
        ranks = grp["ros_rank"].dropna().tolist()
        n_start = STARTER_SLOTS.get(pos, 1)
        starters = ranks[:n_start] or [250]
        depth = sum(1 for r in ranks[n_start:] if r <= STARTABLE_ROS[pos])
        out[pos] = {"starter_avg": sum(starters) / len(starters),
                    "depth": depth, "count": len(grp)}
    return out


# ------------------------------------------------------------------ lineup model
# Which positions can fill each flexible slot.
FLEX_ELIGIBLE = {"FLEX": ("RB", "WR", "TE"), "RB/WR": ("RB", "WR"),
                 "OP": ("QB", "RB", "WR", "TE")}
# Bench depth still matters (byes, injuries), just far less than starters:
# weights for the best leftover RB/WR/TE, plus one backup QB and TE.
BENCH_FLEX_WEIGHTS = (0.30, 0.22, 0.15, 0.10)
BENCH_EXTRA_WEIGHTS = {"QB": 0.08, "TE": 0.05}


def _lineup(players: list[dict], lineup: dict, col: str,
            bench: bool = True) -> tuple[float, set]:
    """Best lineup value for a set of players under the league's slot counts.
    Returns (value, ids of the starters)."""
    pool = sorted((p for p in players if p[col] > 0), key=lambda p: -p[col])
    used, total = set(), 0.0
    for slot, n in lineup.items():  # dedicated slots first
        if slot in FLEX_ELIGIBLE:
            continue
        for p in [q for q in pool if q["pos"] == slot][:n]:
            used.add(p["id"])
            total += p[col]
    for slot, n in lineup.items():
        if slot not in FLEX_ELIGIBLE:
            continue
        for p in [q for q in pool if q["pos"] in FLEX_ELIGIBLE[slot]
                  and q["id"] not in used][:n]:
            used.add(p["id"])
            total += p[col]
    if bench:
        rest = [q for q in pool if q["id"] not in used]
        flex = [q for q in rest if q["pos"] in ("RB", "WR", "TE")]
        total += sum(w * q[col] for w, q in zip(BENCH_FLEX_WEIGHTS, flex))
        for pos, w in BENCH_EXTRA_WEIGHTS.items():
            backup = next((q for q in rest if q["pos"] == pos), None)
            if backup:
                total += w * backup[col]
    return total, used


def _as_players(df: pd.DataFrame) -> list[dict]:
    """Roster rows -> light dicts with a ROS value and this week's projection."""
    out = []
    for i, r in df.iterrows():
        proj = r.get("week_proj")
        pid = r.get("espn_id")
        out.append({"id": pid if pid is not None and pd.notna(pid) else ("row", i),
                    "pos": r["pos"], "row": r,
                    "ros_v": trade_value(r.get("ros_rank")),
                    "wk_v": float(proj) if proj is not None and pd.notna(proj) else 0.0})
    return out


def _holding(roster: pd.DataFrame) -> pd.DataFrame:
    """Players occupying a roster spot (IR slots don't count)."""
    if roster.empty or "slot" not in roster.columns:
        return roster
    return roster[roster["slot"] != "IR"]


def drop_costs(roster: pd.DataFrame, lineup: dict | None = None) -> pd.DataFrame:
    """Every droppable player with how much lineup value you'd lose by
    cutting him (lowest first)."""
    if roster.empty:
        return roster
    lineup = lineup or STARTER_SLOTS
    players = _as_players(_holding(roster))
    base, _ = _lineup(players, lineup, "ros_v")
    idx, vals = [], []
    for p in players:
        without = [q for q in players if q is not p]
        idx.append(p["row"].name)
        vals.append(base - _lineup(without, lineup, "ros_v")[0])
    out = roster.loc[idx].copy()
    out["drop_cost"] = vals
    return out.sort_values("drop_cost")


def droppables(roster: pd.DataFrame, n: int = 5,
               lineup: dict | None = None) -> pd.DataFrame:
    """The players whose loss would hurt your lineup least."""
    return drop_costs(roster, lineup).head(n)


def _bid_pct(ros_gain: float) -> float:
    """Share of the interested rivals' budgets a ROS lineup gain is worth.
    Convex: a lineup-changing starter (gain ~40) ~55%, a solid upgrade
    (~20) ~25%, depth (~10) ~12%, a marginal bump (~3) ~3%."""
    if ros_gain <= 0:
        return 0.0
    return min(0.6, 0.55 * (ros_gain / 40.0) ** 1.1)


def waiver_targets(fas: pd.DataFrame, my_roster: pd.DataFrame, league: dict,
                   my_id: int, teams: dict[int, pd.DataFrame],
                   top_n: int = 20) -> pd.DataFrame:
    """Rank free agents by what they do for YOUR lineup, pair each with the
    right drop, and size a FAAB bid to what it takes to beat the rivals who
    would actually want him."""
    if fas.empty:
        return fas
    lineup = league.get("lineup") or STARTER_SLOTS
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

    mine = _as_players(_holding(my_roster))
    bench_slots = league.get("bench_slots") or 0
    open_spot = bench_slots > 0 and len(mine) < sum(lineup.values()) + bench_slots
    base_ros, _ = _lineup(mine, lineup, "ros_v")
    base_wk, _ = _lineup(mine, lineup, "wk_v", bench=False)

    # Each rival's weakest ROS starter by position, to see who'd start a player.
    rival_floor = {}
    for tid, df in teams.items():
        if tid == my_id or df.empty:
            continue
        players = _as_players(_holding(df))
        _, starters = _lineup(players, lineup, "ros_v", bench=False)
        floor = {}
        for p in players:
            if p["id"] in starters:
                floor[p["pos"]] = min(floor.get(p["pos"], 1e9), p["ros_v"])
        rival_floor[tid] = floor

    pool = fas[fas["pos"].isin({"QB", "RB", "WR", "TE", "K", "DST"})].copy()
    pool = pool[pool["ros_rank"].notna() | pool["weekly_rank"].notna()
                | (pool["trend_rank"].fillna(999) <= 50)]
    # Keep the math small: best 120 by ROS, plus anyone projected to start.
    pool = pool.sort_values("ros_rank", na_position="last")
    keep = pool.head(120).index.union(pool[pool["week_proj"].fillna(0) >= 8].index)
    pool = pool.loc[keep]

    rows = []
    for idx, fa in pool.iterrows():
        cand = _as_players(pd.DataFrame([fa]))[0]
        cand["id"] = ("fa", idx)
        best = None
        for d in ([None] if open_spot else []) + mine:
            if d is not None and ("K" in (d["pos"], cand["pos"]) or "DST" in (
                    d["pos"], cand["pos"])) and cand["pos"] != d["pos"]:
                continue  # kickers and defenses are straight swaps
            after = [q for q in mine if q is not d] + [cand]
            r_after, r_start = _lineup(after, lineup, "ros_v")
            w_after, w_start = _lineup(after, lineup, "wk_v", bench=False)
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


def start_sit(my_roster: pd.DataFrame, lineup: dict | None = None) -> list[dict]:
    """Bench players the experts rank above a starter they could replace.

    Builds the best lineup from this week's FantasyPros ranks (FLEX ranks for
    RB/WR/TE so they compare across positions, positional ranks for QB/K/DST)
    and flags only swaps the slot rules allow -- a WR never "replaces" a TE
    sitting in the TE slot."""
    if my_roster.empty or "starter" not in my_roster.columns:
        return []
    lineup = lineup or STARTER_SLOTS
    active = _holding(my_roster)
    players = []
    for i, r in active.iterrows():
        rank = r.get("flex_rank") if r["pos"] in FLEX_POSITIONS else r.get("weekly_rank")
        if rank is None or pd.isna(rank):
            continue  # unranked this week: bye, out, or irrelevant
        pid = r.get("espn_id")
        players.append({"id": pid if pid is not None and pd.notna(pid) else ("row", i),
                        "pos": r["pos"], "row": r, "rank": float(rank),
                        "ss_v": 400.0 - float(rank)})
    _, best = _lineup(players, lineup, "ss_v", bench=False)
    starting = [p for p in players if p["row"]["starter"]]
    ins = sorted((p for p in players if p["id"] in best and not p["row"]["starter"]),
                 key=lambda p: p["rank"])
    outs = [p for p in starting if p["id"] not in best]

    def fits(p, slot):
        return slot == p["pos"] or p["pos"] in FLEX_ELIGIBLE.get(slot, ())

    flags = []
    for b in ins:
        cand = [o for o in outs if fits(b, o["row"]["slot"])]
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
        rank = r.get("flex_rank") if r["pos"] in FLEX_POSITIONS else r.get("weekly_rank")
        if (rank is None or pd.isna(rank)) and r["pos"] in ("QB", "RB", "WR", "TE"):
            flags.append({"in": None, "out": r, "note": "not ranked this week (bye or out?)"})
    return flags


def trade_value(rank) -> float:
    """Trade-value curve. Rank differences aren't linear: ROS 10 vs 20 is a
    chasm, ROS 110 vs 120 is noise. v(1)≈118, v(24)≈85, v(50)≈59, v(100)≈29."""
    if rank is None or pd.isna(rank):
        return 0.0
    return 120.0 * math.exp(-float(rank) / 70.0)


def player_value(p) -> float:
    """A player's trade value: FantasyCalc market value when available (what
    real leagues actually trade at, scaled onto the curve's range), else the
    expert-consensus curve."""
    mv = p.get("mkt_value")
    if mv is not None and pd.notna(mv):
        return float(mv) / 90.0  # top of market (~10,700) ≈ curve top (~118)
    return trade_value(p.get("ros_rank"))


def _depth_pieces(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    grp = df[(df["pos"] == pos) & df["ros_rank"].notna()].sort_values("ros_rank")
    return grp.iloc[STARTER_SLOTS.get(pos, 1):]


def trade_ideas(teams: dict[int, pd.DataFrame], my_id: int,
                team_names: dict[int, str], top_n: int = 8,
                include_steals: bool = False) -> list[dict]:
    """Propose trades the OTHER side could plausibly say yes to.

    Only deals inside a fairness window on the value curve are kept, every
    idea must address a real need on both sides, and 2-for-1 packages let you
    consolidate depth into one better starter (paying the usual premium)."""
    needs = {tid: positional_needs(df) for tid, df in teams.items()}
    if my_id not in needs or len(needs) < 2:
        return []
    positions = ("QB", "RB", "WR", "TE")
    league_avg = {pos: sum(n[pos]["starter_avg"] for n in needs.values()) / len(needs)
                  for pos in positions}

    def weakness(tid):  # starter_avg above league average = weak (positive)
        return {pos: needs[tid][pos]["starter_avg"] - league_avg[pos]
                for pos in positions}

    my_roster = teams[my_id]
    my_weak = weakness(my_id)
    weak_targets = sorted((p for p in positions if my_weak[p] > 5),
                          key=lambda p: -my_weak[p])[:2]
    if not weak_targets and not include_steals:
        return []

    # My tradable depth: bench-quality-or-better pieces beyond my starters,
    # best six by value so package loops stay small.
    my_depth = []
    for pos in positions:
        for _, p in _depth_pieces(my_roster, pos).iterrows():
            if p["ros_rank"] <= STARTABLE_ROS[pos] * 1.3:
                my_depth.append(p)
    my_depth.sort(key=lambda p: p["ros_rank"])
    my_depth = my_depth[:6]

    ideas = []
    for tid, their in teams.items():
        if tid == my_id:
            continue
        th_weak = weakness(tid)
        th_needs = needs[tid]
        for w_pos in weak_targets:
            grp = their[(their["pos"] == w_pos)
                        & their["ros_rank"].notna()].sort_values("ros_rank")
            n_start = STARTER_SLOTS.get(w_pos, 1)
            gettable = [r for _, r in grp.iloc[n_start:n_start + 2].iterrows()]
            # If they're 2+ deep beyond starters, even their last starter is in
            # play for the right return.
            if len(grp) - n_start >= 2 and n_start >= 1:
                gettable.insert(0, grp.iloc[n_start - 1])

            my_starters = my_roster[(my_roster["pos"] == w_pos)
                                    & my_roster["ros_rank"].notna()].sort_values("ros_rank")
            worst_starter = my_starters.iloc[:n_start].tail(1)
            worst_v = player_value(worst_starter.iloc[0]) if len(worst_starter) else 0.0

            for tgt in gettable:
                tv = player_value(tgt)
                if tv <= worst_v + 3:
                    continue  # wouldn't move my lineup
                slot_note = (f"slots in over {worst_starter.iloc[0]['player']}"
                             if len(worst_starter) else f"becomes your {w_pos}1")

                # ---- 1-for-1: my depth piece, near-even value, at a spot
                # where THEY are actually thin.
                # Second opinion: the pure expert-consensus curve. A deal must
                # look fair on BOTH scales -- when sources disagree hard about
                # a player, we propose nothing rather than risk a fleece.
                ecr_tv = trade_value(tgt.get("ros_rank"))

                for off in my_depth:
                    if off["pos"] == w_pos:
                        continue
                    ratio = player_value(off) / tv
                    their_gap = th_weak.get(off["pos"], 0.0)
                    if not (0.88 <= ratio <= 1.18) or their_gap < 3:
                        continue
                    if ecr_tv > 0:
                        # one-directional guard: never overpay on the expert
                        # scale, even when the market calls it even
                        if trade_value(off.get("ros_rank")) / ecr_tv > 1.25:
                            continue
                    score = (30 * (1 - abs(ratio - 1.02) / 0.16)
                             + min(their_gap, 30) + min(my_weak[w_pos], 30))
                    ideas.append({
                        "team": team_names.get(tid, f"Team {tid}"), "kind": "1-for-1",
                        "get": tgt, "give": [off], "ratio": ratio, "score": score,
                        "why_me": f"{tgt['player']} {slot_note}",
                        "why_them": (f"{off['player']} upgrades their {off['pos']} "
                                     f"(their {off['pos']} starters average ROS "
                                     f"{th_needs[off['pos']]['starter_avg']:.0f})"),
                    })

                # ---- 2-for-1: two depth pieces for their better player. The
                # side getting two pays a consolidation premium (~10-30% by
                # combined value), which is why these get accepted.
                for i in range(len(my_depth)):
                    for j in range(i + 1, len(my_depth)):
                        a, b = my_depth[i], my_depth[j]
                        if w_pos in (a["pos"], b["pos"]):
                            continue
                        pkg = player_value(a) + 0.7 * player_value(b)
                        ratio = pkg / tv
                        their_gap = max(th_weak.get(a["pos"], 0), th_weak.get(b["pos"], 0))
                        if not (1.02 <= ratio <= 1.40) or their_gap < 3:
                            continue
                        if ecr_tv > 0:
                            pkg_ecr = (trade_value(a.get("ros_rank"))
                                       + 0.7 * trade_value(b.get("ros_rank")))
                            if pkg_ecr / ecr_tv > 1.45:
                                continue
                        score = (25 * (1 - abs(ratio - 1.18) / 0.25)
                                 + min(their_gap, 30) + min(my_weak[w_pos], 30) + 6)
                        ideas.append({
                            "team": team_names.get(tid, f"Team {tid}"), "kind": "2-for-1",
                            "get": tgt, "give": [a, b], "ratio": ratio, "score": score,
                            "why_me": (f"consolidate two bench pieces into one "
                                       f"starter — {tgt['player']} {slot_note}"),
                            "why_them": (f"they turn one player into two rotation "
                                         f"pieces where they're thin "
                                         f"({a['pos']}/{b['pos']})"),
                        })

    # ---- steals: offers tilted MY way (below the fair window). The other
    # side often declines, but asking costs nothing. Any position upgrade
    # qualifies, not just my weak spots.
    if include_steals and my_depth:
        cheap_first = sorted(my_depth, key=player_value)
        for tid, their in teams.items():
            if tid == my_id:
                continue
            for pos in positions:
                grp = their[(their["pos"] == pos)
                            & their["ros_rank"].notna()].sort_values("ros_rank")
                n_start = STARTER_SLOTS.get(pos, 1)
                my_st = my_roster[(my_roster["pos"] == pos)
                                  & my_roster["ros_rank"].notna()].sort_values("ros_rank")
                worst_st = my_st.iloc[:n_start].tail(1)
                worst_v = player_value(worst_st.iloc[0]) if len(worst_st) else 0.0
                for _, tgt in grp.iloc[n_start:n_start + 2].iterrows():
                    tv = player_value(tgt)
                    if tv <= worst_v + 4:
                        continue
                    for off in cheap_first:
                        if off["pos"] == pos:
                            continue
                        ratio = player_value(off) / tv
                        if 0.45 <= ratio < 0.88:
                            gain = tv - player_value(off)
                            ideas.append({
                                "team": team_names.get(tid, f"Team {tid}"),
                                "kind": "steal", "get": tgt, "give": [off],
                                "ratio": ratio, "score": 15 + gain * 0.4,
                                "why_me": (f"{tgt['player']} upgrades your {pos} "
                                           f"at a discount"),
                                "why_them": ("honestly, not much — this one's "
                                             "tilted your way. Worst case they "
                                             "say no."),
                            })
                            break

    ideas.sort(key=lambda i: -i["score"])
    out, per_team = [], {}
    for i in ideas:
        if per_team.get(i["team"], 0) >= 2:
            continue
        per_team[i["team"]] = per_team.get(i["team"], 0) + 1
        out.append(i)
    return out[:top_n]
