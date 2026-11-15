"""The pre-draft board: expert consensus + real-draft ADP + an optional CSV,
blended into one ranked list for the Draft Room."""

import numpy as np
import pandas as pd

from core import adp as adp_source
from core import fantasypros, sleeper
from core.names import normalize_name

DEFAULT_WEIGHTS = {"ecr": 0.5, "adp": 0.3, "csv": 0.2}
_NAME_COLS = ("player", "name", "player_name", "full_name", "player name")
_RANK_COLS = ("rank", "overall", "overall rank", "adp", "ecr", "rk", "projectedpoints")


def parse_rankings_csv(file) -> pd.DataFrame:
    """Flexible parser for an uploaded rankings CSV (Underdog export or any
    name + rank file). Returns name_key + csv_rank; raises ValueError with a
    readable message when the columns can't be found."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "firstname" in df.columns and "lastname" in df.columns:
        names = df["firstname"].fillna("") + " " + df["lastname"].fillna("")
    else:
        name_col = next((c for c in _NAME_COLS if c in df.columns), None)
        if name_col is None:
            raise ValueError(f"No player-name column found. Columns: {list(df.columns)}")
        names = df[name_col].astype(str)

    rank_col = next((c for c in _RANK_COLS if c in df.columns), None)
    if rank_col is None:
        raise ValueError(f"No rank/ADP column found. Columns: {list(df.columns)}")
    ranks = pd.to_numeric(df[rank_col], errors="coerce")
    if rank_col == "projectedpoints":  # higher points = better -> convert to rank
        ranks = ranks.rank(ascending=False)

    out = pd.DataFrame({"name_key": names.map(normalize_name), "csv_rank": ranks})
    out = out.dropna(subset=["csv_rank"])
    return out.sort_values("csv_rank").drop_duplicates(subset="name_key", keep="first")


def blend(board: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Add `consensus` (weighted average of the source ranks each player has),
    `rank`, and `value` (ADP minus consensus: positive = the market lets him fall)."""
    w = weights or DEFAULT_WEIGHTS
    adp_rank = board["adp"].rank(method="first")
    csv_rank = board["csv_rank"].rank(method="first")
    total = (board["ecr"].fillna(0) * w.get("ecr", 0)
             + adp_rank.fillna(0) * w.get("adp", 0)
             + csv_rank.fillna(0) * w.get("csv", 0))
    weight_sum = (w.get("ecr", 0) * board["ecr"].notna()
                  + w.get("adp", 0) * adp_rank.notna()
                  + w.get("csv", 0) * csv_rank.notna())
    board = board.assign(consensus=total / weight_sum.replace(0, np.nan))
    board = board.sort_values("consensus").reset_index(drop=True)
    board["rank"] = board.index + 1
    board["value"] = (board["adp"] - board["consensus"]).round(1)
    return board


def build_board(csv_ranks: pd.DataFrame | None = None,
                weights: dict | None = None) -> pd.DataFrame:
    """Merge every source into the master draft board, best player first."""
    ecr = fantasypros.fetch_draft_ecr()
    board = ecr.copy()
    meta = {"ecr_updated": ecr.attrs.get("updated", ""),
            "ecr_fetched": ecr.attrs.get("fetched", ""), "adp_drafts": 0, "adp_date": ""}

    try:
        adp = adp_source.fetch_adp()
        board = board.merge(adp[["key", "adp", "adp_std", "adp_pick"]], on="key", how="left")
        meta["adp_drafts"] = adp.attrs.get("drafts", 0)
        meta["adp_date"] = adp.attrs.get("end_date", "")
    except Exception:
        board["adp"], board["adp_std"], board["adp_pick"] = np.nan, np.nan, ""

    try:
        board = board.merge(sleeper.fetch_players(), on="key", how="left")
        trending = sleeper.fetch_trending()
        board["trending"] = board["sleeper_id"].map(trending).fillna(0).astype(int)
    except Exception:
        board["sleeper_id"], board["injury"], board["injury_note"] = None, "", ""
        board["trending"] = 0
    board["injury"] = board["injury"].fillna("")

    if csv_ranks is not None and not csv_ranks.empty:
        board["csv_rank"] = board["player"].map(normalize_name).map(
            csv_ranks.set_index("name_key")["csv_rank"])
    else:
        board["csv_rank"] = np.nan

    board = blend(board, weights)
    board.attrs.update(meta)
    return board
