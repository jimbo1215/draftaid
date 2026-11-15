"""Join ESPN players with every other source into one enriched table."""

from dataclasses import dataclass, field

import pandas as pd

RANK_COLS = ["ros_rank", "ros_tier", "ros_pos_rank", "bye"]


@dataclass
class Sources:
    """Everything a roster gets enriched with. Any piece may be empty."""
    ros: pd.DataFrame                       # FantasyPros rest-of-season ranks
    weekly: pd.DataFrame                    # FantasyPros this-week ranks
    sleeper: pd.DataFrame = field(default_factory=pd.DataFrame)
    trending: dict = field(default_factory=dict)        # sleeper_id -> adds
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    proj: dict = field(default_factory=dict)            # sleeper_id -> {week: pts}
    opp: dict = field(default_factory=dict)             # sleeper_id -> {week: opponent}
    proj_pos: dict = field(default_factory=dict)        # sleeper_id -> position
    weeks: list = field(default_factory=list)           # weeks with projections loaded
    playoff_weeks: list = field(default_factory=list)   # the league's playoff weeks
    po_rank: dict = field(default_factory=dict)         # sleeper_id -> playoff pos rank

    def __post_init__(self):
        if not self.po_rank:
            self.po_rank = playoff_ranks(self.proj, self.proj_pos, self.playoff_weeks)


def playoff_ranks(proj: dict, proj_pos: dict, playoff_weeks: list) -> dict:
    """sleeper_id -> positional rank by projected points over the playoff
    weeks, across every projected player (not just this league's)."""
    if not playoff_weeks or not proj:
        return {}
    totals: dict[str, list] = {}
    for sid, by_week in proj.items():
        pos = proj_pos.get(sid)
        total = sum(by_week.get(w, 0.0) for w in playoff_weeks)
        if pos and total > 0:
            totals.setdefault(pos, []).append((total, sid))
    ranks = {}
    for rows in totals.values():
        for i, (_, sid) in enumerate(sorted(rows, reverse=True), 1):
            ranks[sid] = i
    return ranks


def _sleeper_ids(df: pd.DataFrame, src: Sources) -> pd.Series:
    """Sleeper id for each row. An ESPN-id match wins when one exists (immune
    to name spelling; both Sleeper and the market feed carry ESPN ids), else
    normalized name + position."""
    ids = pd.Series([None] * len(df), index=df.index, dtype=object)
    by_espn: dict[int, str] = {}
    if not src.sleeper.empty:
        ids = df["key"].map(src.sleeper.drop_duplicates("key").set_index("key")["sleeper_id"])
        if "espn_id" in src.sleeper.columns:
            known = src.sleeper.dropna(subset=["espn_id"])
            by_espn.update(zip(known["espn_id"].astype(int), known["sleeper_id"]))
    if not src.market.empty and "mkt_espn_id" in src.market.columns:
        known = src.market.dropna(subset=["mkt_espn_id"])
        by_espn.update(zip(known["mkt_espn_id"].astype(int), known["sleeper_id"]))
    if by_espn and "espn_id" in df.columns:
        via_espn = df["espn_id"].map(
            lambda v: by_espn.get(int(v)) if v is not None and pd.notna(v) else None)
        ids = via_espn.where(via_espn.notna(), ids)
    return ids


def _injury(row) -> str:
    """Sleeper's status when it has one, else ESPN's (title-cased)."""
    if row["injury"]:
        return row["injury"]
    espn = row.get("espn_injury") or ""
    return "" if espn in ("", "ACTIVE") else espn.replace("_", " ").title()


def enrich(players: list[dict], src: Sources) -> pd.DataFrame:
    """ESPN player dicts -> DataFrame with ranks, ids, injuries, market
    values, and week-by-week projections attached."""
    df = pd.DataFrame(players)
    if df.empty:
        return df
    df = df.merge(src.ros[["key"] + RANK_COLS], on="key", how="left")
    if not src.weekly.empty:
        df = df.merge(src.weekly, on="key", how="left")
    else:
        df["weekly_rank"], df["flex_rank"] = pd.NA, pd.NA

    df["sleeper_id"] = _sleeper_ids(df, src)
    if not src.sleeper.empty:
        extra = src.sleeper.drop_duplicates("sleeper_id").set_index("sleeper_id")
        df["injury"] = df["sleeper_id"].map(extra["injury"]).fillna("")
        df["depth_order"] = (df["sleeper_id"].map(extra["depth_order"])
                             if "depth_order" in extra.columns else pd.NA)
    else:
        df["injury"], df["depth_order"] = "", pd.NA
    df["injury"] = df.apply(_injury, axis=1)

    if src.trending:
        # Raw add counts swing by orders of magnitude (millions on waiver
        # day), so rank within today's list is the stable signal.
        order = {sid: i + 1 for i, sid in enumerate(
            sorted(src.trending, key=lambda k: -src.trending[k]))}
        df["trend_rank"] = df["sleeper_id"].map(order)
    else:
        df["trend_rank"] = pd.NA

    if not src.market.empty and df["sleeper_id"].notna().any():
        values = src.market.drop(columns="mkt_espn_id", errors="ignore")
        df = df.merge(values, on="sleeper_id", how="left")
    else:
        df["mkt_value"], df["mkt_rank"], df["mkt_trend"] = pd.NA, pd.NA, pd.NA

    # Week-by-week projections: a dict per player, plus the playoff window.
    df["wk"] = df["sleeper_id"].map(lambda s: src.proj.get(s) or {})
    df["wk_opp"] = df["sleeper_id"].map(lambda s: src.opp.get(s) or {})
    po_weeks = src.playoff_weeks
    df["po_pts"] = df["wk"].map(
        lambda w: round(sum(w.get(x, 0.0) for x in po_weeks), 1) if w and po_weeks else None)
    df["po_rank"] = df["sleeper_id"].map(src.po_rank)
    return df
