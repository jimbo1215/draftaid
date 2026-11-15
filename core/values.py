"""Player value scales shared by the lineup, waiver, and trade models."""

import math

import pandas as pd


def rank_value(rank) -> float:
    """Value curve over rest-of-season rank. Rank gaps aren't linear: ROS 10 vs
    20 is a chasm, 110 vs 120 is noise. v(1)≈118, v(24)≈85, v(50)≈59, v(100)≈29."""
    if rank is None or pd.isna(rank):
        return 0.0
    return 120.0 * math.exp(-float(rank) / 70.0)


def player_value(p) -> float:
    """Trade value: FantasyCalc market value when available (what real leagues
    actually trade at, scaled onto the curve's range), else the expert curve."""
    mv = p.get("mkt_value")
    if mv is not None and pd.notna(mv):
        return float(mv) / 90.0  # top of market (~10,700) ≈ curve top (~118)
    return rank_value(p.get("ros_rank"))
