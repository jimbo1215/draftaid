"""Trade finder: deals the other side could plausibly accept, and market movers.

A fair deal has to clear two independent value scales (the real-trade market
and expert rest-of-season consensus) and fill a real need on both rosters.
When the sources disagree hard about a player, nothing is proposed.
"""

import pandas as pd

from core.lineup import DEFAULT_LINEUP, STARTABLE_ROS, positional_needs
from core.values import player_value, rank_value

POSITIONS = ("QB", "RB", "WR", "TE")
FAIR_1FOR1 = (0.88, 1.18)      # give/get market ratio that both sides can accept
FAIR_2FOR1 = (1.02, 1.40)      # consolidating two pieces into one costs a premium
STEAL = (0.45, 0.88)           # tilted the user's way; below 0.45 is an insult
MAX_ECR_OVERPAY_1FOR1 = 1.25   # never overpay on the expert scale either
MAX_ECR_OVERPAY_2FOR1 = 1.45


def _ranked(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    return df[(df["pos"] == pos) & df["ros_rank"].notna()].sort_values("ros_rank")


def _worst_starter(roster: pd.DataFrame, pos: str):
    """(row | None, value) of the weakest starter at a position."""
    starters = _ranked(roster, pos).iloc[:DEFAULT_LINEUP.get(pos, 1)].tail(1)
    if starters.empty:
        return None, 0.0
    return starters.iloc[0], player_value(starters.iloc[0])


def _tradable_depth(roster: pd.DataFrame, limit: int = 6) -> list:
    """Bench-quality-or-better pieces beyond the starters, best first."""
    depth = []
    for pos in POSITIONS:
        for _, p in _ranked(roster, pos).iloc[DEFAULT_LINEUP.get(pos, 1):].iterrows():
            if p["ros_rank"] <= STARTABLE_ROS[pos] * 1.3:
                depth.append(p)
    depth.sort(key=lambda p: p["ros_rank"])
    return depth[:limit]


def _fair_ideas(my_roster, my_depth, my_weak, weak_targets, their, th_weak, th_needs,
                team: str) -> list[dict]:
    ideas = []
    for w_pos in weak_targets:
        grp = _ranked(their, w_pos)
        n_start = DEFAULT_LINEUP.get(w_pos, 1)
        gettable = [r for _, r in grp.iloc[n_start:n_start + 2].iterrows()]
        # If they're 2+ deep beyond starters, even their last starter is in
        # play for the right return.
        if len(grp) - n_start >= 2 and n_start >= 1:
            gettable.insert(0, grp.iloc[n_start - 1])
        worst, worst_v = _worst_starter(my_roster, w_pos)

        for tgt in gettable:
            tv = player_value(tgt)
            if tv <= worst_v + 3:
                continue  # wouldn't move my lineup
            slot_note = (f"slots in over {worst['player']}" if worst is not None
                         else f"becomes your {w_pos}1")
            ecr_tv = rank_value(tgt.get("ros_rank"))  # second opinion on value

            # 1-for-1: near-even value, at a spot where THEY are thin.
            for off in my_depth:
                if off["pos"] == w_pos:
                    continue
                ratio = player_value(off) / tv
                their_gap = th_weak.get(off["pos"], 0.0)
                if not (FAIR_1FOR1[0] <= ratio <= FAIR_1FOR1[1]) or their_gap < 3:
                    continue
                if ecr_tv > 0 and rank_value(off.get("ros_rank")) / ecr_tv > MAX_ECR_OVERPAY_1FOR1:
                    continue
                ideas.append({
                    "team": team, "kind": "1-for-1", "get": tgt, "give": [off],
                    "ratio": ratio,
                    "score": (30 * (1 - abs(ratio - 1.02) / 0.16)
                              + min(their_gap, 30) + min(my_weak[w_pos], 30)),
                    "why_me": f"{tgt['player']} {slot_note}",
                    "why_them": (f"{off['player']} upgrades their {off['pos']} "
                                 f"(their {off['pos']} starters average ROS "
                                 f"{th_needs[off['pos']]['starter_avg']:.0f})"),
                })

            # 2-for-1: two depth pieces for their better player. The side
            # getting two pays a consolidation premium, which is why these
            # get accepted.
            for i in range(len(my_depth)):
                for j in range(i + 1, len(my_depth)):
                    a, b = my_depth[i], my_depth[j]
                    if w_pos in (a["pos"], b["pos"]):
                        continue
                    ratio = (player_value(a) + 0.7 * player_value(b)) / tv
                    their_gap = max(th_weak.get(a["pos"], 0), th_weak.get(b["pos"], 0))
                    if not (FAIR_2FOR1[0] <= ratio <= FAIR_2FOR1[1]) or their_gap < 3:
                        continue
                    if ecr_tv > 0:
                        pkg_ecr = (rank_value(a.get("ros_rank"))
                                   + 0.7 * rank_value(b.get("ros_rank")))
                        if pkg_ecr / ecr_tv > MAX_ECR_OVERPAY_2FOR1:
                            continue
                    ideas.append({
                        "team": team, "kind": "2-for-1", "get": tgt, "give": [a, b],
                        "ratio": ratio,
                        "score": (25 * (1 - abs(ratio - 1.18) / 0.25)
                                  + min(their_gap, 30) + min(my_weak[w_pos], 30) + 6),
                        "why_me": (f"consolidate two bench pieces into one starter: "
                                   f"{tgt['player']} {slot_note}"),
                        "why_them": (f"they turn one player into two rotation pieces "
                                     f"where they're thin ({a['pos']}/{b['pos']})"),
                    })
    return ideas


def _steal_ideas(my_roster, my_depth, their, team: str) -> list[dict]:
    """Offers tilted the user's way: any position upgrade, priced below the
    fair window. The other side often declines, but asking costs nothing."""
    ideas = []
    cheap_first = sorted(my_depth, key=player_value)
    for pos in POSITIONS:
        n_start = DEFAULT_LINEUP.get(pos, 1)
        _, worst_v = _worst_starter(my_roster, pos)
        for _, tgt in _ranked(their, pos).iloc[n_start:n_start + 2].iterrows():
            tv = player_value(tgt)
            if tv <= worst_v + 4:
                continue
            for off in cheap_first:
                if off["pos"] == pos:
                    continue
                ratio = player_value(off) / tv
                if STEAL[0] <= ratio < STEAL[1]:
                    ideas.append({
                        "team": team, "kind": "steal", "get": tgt, "give": [off],
                        "ratio": ratio, "score": 15 + (tv - player_value(off)) * 0.4,
                        "why_me": f"{tgt['player']} upgrades your {pos} at a discount",
                        "why_them": ("honestly, not much. This one's tilted your way; "
                                     "worst case they say no."),
                    })
                    break
    return ideas


def league_weakness(teams: dict[int, pd.DataFrame]) -> tuple[dict, dict]:
    """(needs per team, weakness per team). Weakness = starters' average ROS
    rank minus the league average at that position; positive = weak."""
    needs = {tid: positional_needs(df) for tid, df in teams.items()}
    avg = {pos: sum(n[pos]["starter_avg"] for n in needs.values()) / max(1, len(needs))
           for pos in POSITIONS}
    weak = {tid: {pos: n[pos]["starter_avg"] - avg[pos] for pos in POSITIONS}
            for tid, n in needs.items()}
    return needs, weak


def trade_ideas(teams: dict[int, pd.DataFrame], my_id: int,
                team_names: dict[int, str], top_n: int = 8,
                include_steals: bool = False) -> list[dict]:
    """Ranked trade ideas: fair 1-for-1s and 2-for-1s, plus (optionally)
    steals. At most two ideas per opposing team."""
    if my_id not in teams or len(teams) < 2:
        return []
    needs, weak = league_weakness(teams)
    my_roster, my_weak = teams[my_id], weak[my_id]
    weak_targets = sorted((p for p in POSITIONS if my_weak[p] > 5),
                          key=lambda p: -my_weak[p])[:2]
    if not weak_targets and not include_steals:
        return []
    my_depth = _tradable_depth(my_roster)

    ideas = []
    for tid, their in teams.items():
        if tid == my_id:
            continue
        team = team_names.get(tid, f"Team {tid}")
        found = _fair_ideas(my_roster, my_depth, my_weak, weak_targets, their,
                            weak[tid], needs[tid], team)
        if include_steals and my_depth:
            found += _steal_ideas(my_roster, my_depth, their, team)
        for idea in found:
            idea["team_id"] = tid
        ideas += found

    ideas.sort(key=lambda i: -i["score"])
    out, per_team = [], {}
    for i in ideas:
        if per_team.get(i["team"], 0) >= 2:
            continue
        per_team[i["team"]] = per_team.get(i["team"], 0) + 1
        out.append(i)
    return out[:top_n]


def market_movers(teams: dict[int, pd.DataFrame], my_id: int,
                  team_names: dict[int, str], top_n: int = 4) -> dict:
    """Where the trade market and the experts disagree, with momentum.

    sell: your players the market prices above the experts' rank while their
          value is still climbing (cash out before it corrects).
    buy:  rivals' players the experts rank above a falling market price
          (their manager may sell low)."""
    sell, buy = [], []
    for tid, df in teams.items():
        if df.empty or "mkt_rank" not in df.columns:
            continue
        rated = df[df["mkt_rank"].notna() & df["ros_rank"].notna()
                   & df["pos"].isin(POSITIONS)]
        for _, p in rated.iterrows():
            gap = float(p["ros_rank"]) - float(p["mkt_rank"])   # + = market likes him more
            trend = float(p["mkt_trend"]) if pd.notna(p.get("mkt_trend")) else 0.0
            if tid == my_id and gap >= 10 and trend > 0:
                sell.append({"player": p, "gap": gap, "trend": trend,
                             "score": gap + trend / 40})
            elif tid != my_id and gap <= -10 and trend < 0 and p["ros_rank"] <= 110:
                buy.append({"player": p, "gap": -gap, "trend": trend,
                            "team": team_names.get(tid, f"Team {tid}"), "team_id": tid,
                            "score": -gap - trend / 40})
    sell.sort(key=lambda x: -x["score"])
    buy.sort(key=lambda x: -x["score"])
    return {"sell": sell[:top_n], "buy": buy[:top_n]}
