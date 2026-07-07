"""Snake-draft math, availability probabilities, and pick suggestions."""

import math

import pandas as pd

FLEX_POSITIONS = {"RB", "WR", "TE"}


def my_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    """Overall pick numbers for a draft slot in a snake draft."""
    picks = []
    for rnd in range(1, rounds + 1):
        in_round = slot if rnd % 2 == 1 else teams + 1 - slot
        picks.append((rnd - 1) * teams + in_round)
    return picks


def snake_team_for_pick(overall: int, teams: int) -> int:
    """Draft slot (1-based) that owns a given overall pick in a snake draft."""
    rnd = (overall - 1) // teams + 1
    idx = (overall - 1) % teams
    return idx + 1 if rnd % 2 == 1 else teams - idx


def round_and_pick(overall: int, teams: int) -> tuple[int, int]:
    rnd = (overall - 1) // teams + 1
    pick = (overall - 1) % teams + 1
    return rnd, pick


def next_my_picks(current_overall: int, slot: int, teams: int, rounds: int) -> list[int]:
    return [p for p in my_pick_numbers(slot, teams, rounds) if p >= current_overall]


def survival_probability(adp: float, adp_std: float, pick_number: int) -> float:
    """P(player still on the board when `pick_number` is on the clock),
    modeling his actual selection as Normal(adp, adp_std)."""
    if pd.isna(adp) or pd.isna(adp_std) or adp_std <= 0:
        return float("nan")
    z = (pick_number - adp) / adp_std
    return max(0.0, min(1.0, 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def roster_needs(my_players: pd.DataFrame, roster_spots: dict) -> dict:
    """Unfilled starter slots per position. FLEX absorbs RB/WR/TE overflow."""
    counts = my_players["pos"].value_counts().to_dict() if len(my_players) else {}
    needs, flex_overflow = {}, 0
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        have = counts.get(pos, 0)
        want = roster_spots.get(pos, 0)
        needs[pos] = max(0, want - have)
        if pos in FLEX_POSITIONS:
            flex_overflow += max(0, have - want)
    needs["FLEX"] = max(0, roster_spots.get("FLEX", 0) - flex_overflow)
    return needs


def _need_multiplier(pos: str, needs: dict, rounds_left: int, my_count: int) -> float:
    starters_open = needs.get(pos, 0)
    if pos in ("K", "DST"):
        # Never reach for K/DST: only in play once the draft is nearly done.
        if rounds_left <= starters_open + 1 and starters_open > 0:
            return 1.0
        return 0.05
    if starters_open > 0:
        return 1.0
    if pos in FLEX_POSITIONS and needs.get("FLEX", 0) > 0:
        return 0.9
    # Depth pick: RB/WR benches matter most, backup QB/TE matter less.
    return {"RB": 0.75, "WR": 0.75, "TE": 0.4, "QB": 0.4}.get(pos, 0.5)


def tier_cliffs(available: pd.DataFrame) -> dict:
    """For each position: (current best tier, how many of that tier remain)."""
    out = {}
    for pos, grp in available.groupby("pos"):
        tiers = grp.loc[grp["tier"] > 0, "tier"]
        if tiers.empty:
            continue
        best = int(tiers.min())
        out[pos] = (best, int((tiers == best).sum()))
    return out


def suggest_picks(available: pd.DataFrame, my_players: pd.DataFrame,
                  roster_spots: dict, current_overall: int, my_next_pick: int | None,
                  my_following_pick: int | None, total_rounds: int, top_n: int = 5) -> pd.DataFrame:
    """Rank available players for my current/next pick, with human-readable reasons.

    Blends: overall board value, roster need, tier scarcity at the position,
    market value vs ADP, and how likely the player is gone before my next turn.
    """
    if available.empty:
        return available

    needs = roster_needs(my_players, roster_spots)
    cliffs = tier_cliffs(available)
    rounds_done = len(my_players)
    rounds_left = max(1, total_rounds - rounds_done)

    # Only consider players near the top of the board (no 15th-round reaches).
    pool = available.head(60).copy()

    scores, reasons = [], []
    for _, p in pool.iterrows():
        need_mult = _need_multiplier(p["pos"], needs, rounds_left, rounds_done)
        base = max(0.0, 250.0 - p["consensus"]) * need_mult

        why = []
        tier_info = cliffs.get(p["pos"])
        tier_bonus = 0.0
        if tier_info and p["tier"] == tier_info[0] and tier_info[1] <= 2 and need_mult >= 0.7:
            tier_bonus = 18.0 if tier_info[1] == 1 else 10.0
            why.append(f"last {p['pos']} in tier {p['tier']}" if tier_info[1] == 1
                       else f"only {tier_info[1]} left in {p['pos']} tier {p['tier']}")

        value_bonus = 0.0
        if pd.notna(p["value"]) and p["value"] >= 5:
            value_bonus = min(p["value"], 25.0) * 0.6
            why.append(f"falling {p['value']:+.0f} vs ADP")

        gone_bonus = 0.0
        if my_following_pick is not None:
            surv = survival_probability(p["adp"], p["adp_std"], my_following_pick)
            if not math.isnan(surv) and surv < 0.35:
                gone_bonus = (0.35 - surv) * 30.0
                why.append("almost certainly gone before your next turn" if surv < 0.05
                           else f"only {surv:.0%} chance he lasts to your next turn")

        if needs.get(p["pos"], 0) > 0:
            why.append(f"fills your {p['pos']} starter slot")
        elif p["pos"] in FLEX_POSITIONS and needs.get("FLEX", 0) > 0:
            why.append("fills FLEX")

        if p["injury"] and p["injury"] not in ("", "Questionable"):
            why.append(f"⚠ {p['injury']}")

        scores.append(base + tier_bonus + value_bonus + gone_bonus)
        reasons.append("; ".join(why) if why else "best player available")

    pool["score"] = scores
    pool["why"] = reasons
    return pool.sort_values("score", ascending=False).head(top_n)


TIER_COLORS = [
    "#1f4e79", "#2e6b46", "#7a5c1e", "#6b2e2e", "#4a3a6b",
    "#1e6b6b", "#6b1e4e", "#3e5622", "#5c4033", "#37474f",
]


def tier_color(tier: int) -> str:
    if not tier or tier <= 0:
        return ""
    return TIER_COLORS[(int(tier) - 1) % len(TIER_COLORS)]
