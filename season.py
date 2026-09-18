"""Season-long logic: waiver targets with FAAB bids, trade ideas, start/sit."""

import pandas as pd

STARTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
FLEX_POSITIONS = {"RB", "WR", "TE"}
# ROS overall rank a player must beat to be a plausible starter in 12-team PPR.
STARTABLE_ROS = {"QB": 130, "RB": 110, "WR": 110, "TE": 140, "K": 999, "DST": 999}


def enrich(players: list[dict], ros: pd.DataFrame, weekly: pd.DataFrame,
           sleeper: pd.DataFrame | None = None, trending: dict | None = None) -> pd.DataFrame:
    """Join ESPN players with FantasyPros ROS/weekly ranks and Sleeper data."""
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


def trade_ideas(teams: dict[int, pd.DataFrame], my_id: int,
                team_names: dict[int, str], top_n: int = 8) -> list[dict]:
    """Cross-team surplus/deficit matching → concrete 1-for-1 trade ideas."""
    needs = {tid: positional_needs(df) for tid, df in teams.items()}
    mine = needs.get(my_id, {})
    my_roster = teams.get(my_id, pd.DataFrame())

    # My weak and strong position groups (skip K/DST: nobody trades those).
    weak = sorted(mine.items(), key=lambda kv: kv[1]["starter_avg"], reverse=True)
    strong = [(pos, n) for pos, n in mine.items() if n["depth"] >= 1]

    ideas = []
    for w_pos, w_info in weak[:2]:
        if w_info["starter_avg"] < 60:
            continue  # not actually weak
        for tid, their_needs in needs.items():
            if tid == my_id:
                continue
            th = their_needs.get(w_pos)
            if not th or th["depth"] < 1:
                continue  # they have no surplus where I'm weak
            their_roster = teams[tid]
            grp = their_roster[(their_roster["pos"] == w_pos)
                               & their_roster["ros_rank"].notna()].sort_values("ros_rank")
            n_start = STARTER_SLOTS.get(w_pos, 1)
            targets = grp.iloc[n_start:n_start + 2]  # their depth, not their studs
            for _, tgt in targets.iterrows():
                if tgt["ros_rank"] > STARTABLE_ROS[w_pos]:
                    continue
                # what do I offer? my surplus depth closest in ROS value,
                # ideally at a position where THEY are weak.
                best_offer, offer_gap = None, 1e9
                for s_pos, s_info in strong:
                    if s_pos == w_pos:
                        continue
                    their_s = their_needs.get(s_pos)
                    my_grp = my_roster[(my_roster["pos"] == s_pos)
                                       & my_roster["ros_rank"].notna()].sort_values("ros_rank")
                    depth = my_grp.iloc[STARTER_SLOTS.get(s_pos, 1):]
                    for _, off in depth.iterrows():
                        gap = abs(off["ros_rank"] - tgt["ros_rank"])
                        bonus = -25 if (their_s and their_s["starter_avg"] > 100) else 0
                        if gap + bonus < offer_gap:
                            offer_gap, best_offer = gap + bonus, off
                if best_offer is None:
                    continue
                fairness = best_offer["ros_rank"] - tgt["ros_rank"]  # + = I win
                ideas.append({
                    "team": team_names.get(tid, f"Team {tid}"),
                    "get": tgt, "give": best_offer, "edge": fairness,
                    "why": (f"They're {th['depth']}-deep at {w_pos}, you need one; "
                            f"you're deep at {best_offer['pos']}"),
                })
    ideas.sort(key=lambda i: -i["edge"])
    # keep the best idea per opposing team
    seen, out = set(), []
    for i in ideas:
        if i["team"] not in seen:
            seen.add(i["team"])
            out.append(i)
    return out[:top_n]
