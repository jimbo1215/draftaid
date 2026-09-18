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
        else:
            df["trending"] = 0
    else:
        df["sleeper_id"], df["injury"], df["trending"] = None, "", 0
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


def droppables(roster: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Worst rostered players by ROS rank (bench only, unranked first)."""
    bench = roster[~roster["starter"]] if "starter" in roster.columns else roster
    if bench.empty:
        bench = roster
    return bench.sort_values("ros_rank", ascending=False, na_position="first").head(n)


def waiver_targets(fas: pd.DataFrame, my_roster: pd.DataFrame,
                   faab_left: int, uses_faab: bool, top_n: int = 15) -> pd.DataFrame:
    """Rank free agents and size a FAAB bid for each."""
    if fas.empty:
        return fas
    needs = positional_needs(my_roster)
    drops = droppables(my_roster)

    pool = fas[fas["pos"].isin({"QB", "RB", "WR", "TE", "K", "DST"})].copy()
    pool = pool[pool["ros_rank"].notna() | (pool["trending"] > 200)
                | pool["weekly_rank"].notna()]

    scores, bids, reasons = [], [], []
    for _, p in pool.iterrows():
        why = []
        ros = p["ros_rank"] if pd.notna(p["ros_rank"]) else 400
        score = max(0.0, 300.0 - ros)

        # How big an upgrade over my worst comparable piece?
        mine = my_roster[my_roster["pos"] == p["pos"]]["ros_rank"].dropna()
        my_worst = mine.max() if len(mine) else 400
        upgrade = my_worst - ros
        if upgrade > 0 and ros < 990:
            score += min(upgrade, 120) * 0.5
            why.append(f"upgrade over your {p['pos']}{len(mine)} (+{upgrade:.0f} ROS spots)")

        need = needs.get(p["pos"])
        if need and need["starter_avg"] > STARTABLE_ROS.get(p["pos"], 150) * 0.8:
            score += 25
            why.append(f"you're thin at {p['pos']}")

        if p["trending"] > 0:
            score += min(p["trending"] / 400.0, 25)
            if p["trending"] >= 1000:
                why.append(f"🔥 {int(p['trending']):,} Sleeper adds/24h")

        if pd.notna(p.get("weekly_rank")) and p["weekly_rank"] <= 30:
            score += 10
            why.append(f"startable this week ({p['pos']}{int(p['weekly_rank'])})")

        # ESPN's projection is computed with THIS league's scoring settings,
        # so it corrects for any custom-scoring quirks the consensus misses.
        proj = p.get("week_proj")
        if proj is not None and pd.notna(proj):
            score += min(float(proj), 22.0) * 0.8
            if proj >= 12:
                why.append(f"projects {proj:.1f} in your scoring this week")

        # FAAB sizing: percent of REMAINING budget by ROS tier, nudged by
        # need and trending, floored for pure streamers.
        if ros <= 60:
            lo, hi = 0.35, 0.55
            why.insert(0, "potential league-winner")
        elif ros <= 100:
            lo, hi = 0.18, 0.30
        elif ros <= 150:
            lo, hi = 0.08, 0.14
        elif ros <= 220 or (pd.notna(p.get("weekly_rank")) and p["weekly_rank"] <= 25):
            lo, hi = 0.02, 0.05
        else:
            lo, hi = 0.0, 0.02
        mult = 1.0
        if need and need["starter_avg"] > 120:
            mult += 0.15
        if p["trending"] >= 2000:
            mult += 0.15
        bid_lo = int(round(faab_left * lo * mult))
        bid_hi = max(int(round(faab_left * hi * mult)), bid_lo + (1 if hi > 0 else 0))
        bid_hi = min(bid_hi, faab_left)
        if uses_faab:
            bids.append(f"${bid_lo}–${bid_hi}" if bid_hi > 0 else "$0–$1")
        else:
            bids.append("high claim" if ros <= 120 else "low claim")

        scores.append(score)
        reasons.append("; ".join(why) if why else "best available")

    pool["score"] = scores
    pool["bid"] = bids
    pool["why"] = reasons
    pool = pool.sort_values("score", ascending=False).head(top_n)

    pool.attrs["drops"] = drops
    return pool


def start_sit(my_roster: pd.DataFrame) -> list[dict]:
    """Flag bench players out-ranking starters on this week's expert ranks."""
    flags = []
    if my_roster.empty or "starter" not in my_roster.columns:
        return flags
    for pos in ("QB", "K", "DST"):
        grp = my_roster[my_roster["pos"] == pos]
        starters = grp[grp["starter"] & grp["weekly_rank"].notna()]
        bench = grp[~grp["starter"] & grp["weekly_rank"].notna()]
        for _, b in bench.iterrows():
            worst = starters.sort_values("weekly_rank", ascending=False).head(1)
            if len(worst) and b["weekly_rank"] + 2 < worst.iloc[0]["weekly_rank"]:
                flags.append({"in": b, "out": worst.iloc[0],
                              "note": f"{pos}{int(b['weekly_rank'])} vs "
                                      f"{pos}{int(worst.iloc[0]['weekly_rank'])} this week"})
    flex = my_roster[my_roster["pos"].isin(FLEX_POSITIONS)]
    starters = flex[flex["starter"] & flex["flex_rank"].notna()]
    bench = flex[~flex["starter"] & flex["flex_rank"].notna()]
    for _, b in bench.iterrows():
        worst = starters.sort_values("flex_rank", ascending=False).head(1)
        if len(worst) and b["flex_rank"] + 3 < worst.iloc[0]["flex_rank"]:
            flags.append({"in": b, "out": worst.iloc[0],
                          "note": f"FLEX{int(b['flex_rank'])} vs "
                                  f"FLEX{int(worst.iloc[0]['flex_rank'])} this week"})
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
