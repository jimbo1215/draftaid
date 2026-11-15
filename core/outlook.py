"""Season outlook: play the rest of the season thousands of times.

Each team's strength for each remaining week is its best projected lineup
that week (so byes, injuries, and playoff-week matchups are already in the
number), nudged toward how the team has actually scored. The remaining
schedule and the playoff bracket are then simulated to get playoff, bye, and
title odds, and re-simulated with the same random draws to price a what-if
(a trade, a pickup) by how it moves those odds.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.lineup import DEFAULT_LINEUP, week_points

DECIDED = ("HOME", "AWAY", "TIE")
DEFAULT_SIGMA = 24.0     # typical weekly spread of a fantasy team's score
TALENT_SD = 5.0          # uncertainty about each team's true scoring level


# ------------------------------------------------------------------ schedule

def is_decided(m: dict) -> bool:
    return m.get("winner") in DECIDED


def is_regular(m: dict, league: dict) -> bool:
    return (m.get("tier", "NONE") == "NONE"
            and (m.get("week") or 0) <= league.get("regular_season_weeks", 14))


def period_weeks(league: dict, period: int) -> list[int]:
    """NFL weeks a matchup period spans (playoff rounds can be two weeks)."""
    weeks = (league.get("matchup_weeks") or {}).get(period)
    if weeks:
        return list(weeks)
    reg = league.get("regular_season_weeks", 14)
    if period <= reg:
        return [period]
    length = league.get("playoff_round_weeks", 1)
    start = reg + (period - reg - 1) * length + 1
    return list(range(start, start + length))


def playoff_field(league: dict) -> tuple[int, int, int]:
    """(playoff teams, bracket size, rounds)."""
    n_teams = len(league.get("teams") or [])
    teams = league.get("playoff_teams") or (6 if n_teams >= 12 else 4)
    teams = max(2, min(teams, n_teams)) if n_teams >= 2 else 0
    if teams < 2:
        return 0, 0, 0
    size = 1 << math.ceil(math.log2(teams))
    return teams, size, int(math.log2(size))


def playoff_weeks(league: dict) -> list[int]:
    """Every NFL week of the fantasy playoffs, in order."""
    _, _, rounds = playoff_field(league)
    reg = league.get("regular_season_weeks", 14)
    weeks = []
    for r in range(1, rounds + 1):
        weeks += period_weeks(league, reg + r)
    return weeks


def bracket_order(size: int) -> list[int]:
    """Seeds in bracket-slot order: [1, 8, 4, 5, 2, 7, 3, 6] for eight."""
    order = [1]
    while len(order) < size:
        order = [x for s in order for x in (s, 2 * len(order) + 1 - s)]
    return order


# ------------------------------------------------------------------ history

def weekly_scores(league: dict) -> dict:
    """{team_id: {week: points}} for completed regular-season games."""
    out: dict = {}
    for m in league.get("schedule") or []:
        if is_regular(m, league) and is_decided(m):
            out.setdefault(m["home_id"], {})[m["week"]] = m["home_pts"]
            out.setdefault(m["away_id"], {})[m["week"]] = m["away_pts"]
    return out


def all_play(league: dict) -> dict:
    """{team_id: {"w", "l", "pct"}}: each week's score against every other
    team's. Strips schedule luck out of the record."""
    scores = weekly_scores(league)
    weeks = sorted({w for by_week in scores.values() for w in by_week})
    out = {tid: {"w": 0.0, "l": 0.0} for tid in scores}
    for wk in weeks:
        played = [(tid, s[wk]) for tid, s in scores.items() if wk in s]
        for tid, pts in played:
            for other, opts in played:
                if other == tid:
                    continue
                if pts > opts:
                    out[tid]["w"] += 1
                elif pts < opts:
                    out[tid]["l"] += 1
                else:
                    out[tid]["w"] += 0.5
                    out[tid]["l"] += 0.5
    for rec in out.values():
        games = rec["w"] + rec["l"]
        rec["pct"] = rec["w"] / games if games else 0.5
    return out


# ------------------------------------------------------------------ model

@dataclass
class Model:
    tids: list                   # team ids, fixed order
    weeks: list                  # NFL weeks covered by `mu`
    mu: np.ndarray               # (teams, weeks) expected points
    sigma: float = DEFAULT_SIGMA
    tau: float = TALENT_SD
    ppg: dict = field(default_factory=dict)      # season scoring average
    proj: dict = field(default_factory=dict)     # raw lineup projections

    def col(self, week: int) -> np.ndarray:
        """Expected points for one NFL week; a week outside the projection
        window falls back to each team's average."""
        if week in self.weeks:
            return self.mu[:, self.weeks.index(week)]
        return self.mu.mean(axis=1) if self.mu.size else np.zeros(len(self.tids))

    def strength(self, n_weeks: int = 4) -> dict:
        """Average expected points over the next few weeks (bye-smoothed)."""
        if not self.mu.size:
            return {tid: 0.0 for tid in self.tids}
        span = self.mu[:, :max(1, min(n_weeks, self.mu.shape[1]))].mean(axis=1)
        return dict(zip(self.tids, span.tolist()))


def project_teams(teams: dict, lineup: dict, weeks: list[int],
                  current_week: int | None = None) -> dict:
    """{team_id: {week: best projected lineup points}}."""
    lineup = lineup or DEFAULT_LINEUP
    return {tid: {wk: week_points(df, lineup, wk, current_week) for wk in weeks}
            for tid, df in teams.items()}


def build_model(league: dict, proj: dict, weeks: list[int]) -> Model:
    """Blend lineup projections with each team's scoring history."""
    tids = [t["team_id"] for t in league["teams"]]
    raw = np.array([[proj.get(tid, {}).get(wk, 0.0) for wk in weeks] for tid in tids],
                   dtype=float).reshape(len(tids), len(weeks))
    scores = weekly_scores(league)
    all_pts = [p for by_week in scores.values() for p in by_week.values()]
    games = int(np.median([len(scores.get(tid, {})) for tid in tids])) if tids else 0
    ppg = {tid: float(np.mean(list(scores[tid].values()))) for tid in tids
           if scores.get(tid)}

    mu = raw.copy()
    sigma = DEFAULT_SIGMA
    if games >= 2 and raw.size and raw.mean() > 0:
        # Projections run hot or cold versus a league's real scoring; rescale
        # them to this league, then shift each team toward its own history.
        scale = float(np.clip(np.mean(all_pts) / raw.mean(), 0.85, 1.2))
        mu = raw * scale
        trust_history = min(0.35, 0.04 * games)
        for i, tid in enumerate(tids):
            if tid in ppg:
                mu[i] += trust_history * (ppg[tid] - mu[i].mean())
        resid = [p - ppg[tid] for tid in ppg for p in scores[tid].values()]
        dof = len(resid) - len(ppg)
        if dof >= 6:
            sigma = float(np.clip(math.sqrt(sum(r * r for r in resid) / dof), 16.0, 34.0))
    return Model(tids=tids, weeks=list(weeks), mu=mu, sigma=sigma, ppg=ppg, proj=proj)


# ------------------------------------------------------------------ simulation

@dataclass
class Outlook:
    tids: list
    n: int
    playoff_teams: int
    byes: int
    playoff: dict            # team_id -> P(make playoffs)
    bye: dict                # team_id -> P(first-round bye)
    finals: dict             # team_id -> P(reach the final)
    title: dict              # team_id -> P(win the title)
    exp_wins: dict           # team_id -> expected final win total
    avg_seed: dict           # team_id -> mean finishing position
    seed_dist: dict          # team_id -> [P(seed 1), P(seed 2), ...]
    games_left: int          # regular-season matchup periods still to play
    week_win: dict = field(default_factory=dict)   # my games: period -> (opp_id, P(win))
    leverage: dict | None = None                   # my odds if I win / lose this week


def _live(league: dict) -> dict:
    """{team_id: (expected final | None, points so far, remaining share)} for
    games in progress in the current matchup period. ESPN's live projected
    total is the expected final when it's available."""
    out = {}
    current = league.get("matchup_period")
    for m in league.get("schedule") or []:
        if m.get("week") != current or is_decided(m):
            continue
        if not (m.get("home_pts") or m.get("away_pts")):
            continue  # hasn't kicked off: the model's number stands
        for side in ("home", "away"):
            pts, live = m.get(f"{side}_pts") or 0.0, m.get(f"{side}_live")
            if live is not None and live > 0:
                remaining = float(np.clip((live - pts) / live, 0.02, 1.0))
                out[m[f"{side}_id"]] = (float(live), pts, remaining)
            elif pts > 0:
                out[m[f"{side}_id"]] = (None, pts, 0.75)
    return out


def simulate(league: dict, model: Model, n: int = 10000, seed: int = 7,
             my_id=None) -> Outlook:
    """Monte Carlo the remaining regular season and the playoff bracket.
    The same `seed` yields the same random draws, so two calls that differ
    only in `model` (a what-if roster) are directly comparable."""
    tids = model.tids
    index = {tid: i for i, tid in enumerate(tids)}
    n_teams = len(tids)
    reg_weeks = league.get("regular_season_weeks", 14)
    current = league.get("matchup_period") or league.get("week") or 1
    team = {t["team_id"]: t for t in league["teams"]}
    rng = np.random.default_rng(seed)
    talent = rng.normal(0.0, model.tau, (n, n_teams))
    live = _live(league)

    def period_scores(period: int) -> np.ndarray:
        """(n, teams) simulated scores for one matchup period."""
        weeks = period_weeks(league, period)
        mean = sum(model.col(w) for w in weeks).astype(float).copy()
        spread = np.full(n_teams, model.sigma * math.sqrt(len(weeks)))
        skill = np.full(n_teams, float(len(weeks)))
        if period == current:
            for tid, (expected, pts, remaining) in live.items():
                if tid not in index:
                    continue
                i = index[tid]
                mean[i] = expected if expected is not None else max(mean[i], pts)
                spread[i] *= math.sqrt(remaining)
                skill[i] *= remaining
        noise = rng.normal(0.0, 1.0, (n, n_teams))
        return mean[None, :] + talent * skill[None, :] + noise * spread[None, :]

    # ---- regular season
    wins = np.tile(np.array([team[t]["wins"] + 0.5 * team[t].get("ties", 0)
                             for t in tids], dtype=float), (n, 1))
    points = np.tile(np.array([team[t]["points_for"] for t in tids], dtype=float), (n, 1))
    remaining: dict[int, list] = {}
    for m in league.get("schedule") or []:
        if (is_regular(m, league) and not is_decided(m)
                and m["home_id"] in index and m["away_id"] in index):
            remaining.setdefault(m["week"], []).append((index[m["home_id"]], index[m["away_id"]]))

    week_win, my_results = {}, {}
    me = index.get(my_id)
    for period in sorted(remaining):
        scores = period_scores(period)
        for h, a in remaining[period]:
            home_won = scores[:, h] > scores[:, a]
            wins[:, h] += home_won
            wins[:, a] += ~home_won
            points[:, h] += scores[:, h]
            points[:, a] += scores[:, a]
            if me in (h, a):
                i_won = home_won if me == h else ~home_won
                week_win[period] = (tids[a if me == h else h], float(i_won.mean()))
                my_results[period] = i_won

    # ---- seeding: record, then points for; division winners first
    key = wins * 1e5 + points
    divisions = {}
    for tid in tids:
        divisions.setdefault(team[tid].get("division_id"), []).append(index[tid])
    if len(divisions) > 1:
        for members in divisions.values():
            best = key[:, members].max(axis=1, keepdims=True)
            key[:, members] += (key[:, members] == best) * 1e9
    season_over = not remaining and current > reg_weeks
    official = [team[t].get("playoff_seed") for t in tids]
    if season_over and all(official) and len(set(official)) == n_teams:
        order = np.tile(np.argsort(official), (n, 1))     # ESPN's own seeding
    else:
        order = np.argsort(-key, axis=1)
    seed_of = np.argsort(order, axis=1)                   # (n, teams) 0-based seed

    p_teams, size, rounds = playoff_field(league)
    byes = size - p_teams
    made = seed_of < p_teams if p_teams else np.zeros((n, n_teams), dtype=bool)

    # ---- bracket
    if p_teams:
        slots = np.full((n, size), -1, dtype=int)
        for slot, s in enumerate(bracket_order(size)):
            if s <= p_teams:
                slots[:, slot] = order[:, s - 1]
        decided = {(m["week"], frozenset((m["home_id"], m["away_id"]))):
                   (m["home_id"] if m["winner"] == "HOME" else m["away_id"])
                   for m in league.get("schedule") or []
                   if m.get("tier") == "WINNERS_BRACKET" and m.get("winner") in ("HOME", "AWAY")}
        finalists = slots
        for r in range(1, rounds + 1):
            period = reg_weeks + r
            scores = period_scores(period)
            a, b = slots[:, 0::2], slots[:, 1::2]
            sa = np.take_along_axis(scores, np.clip(a, 0, None), axis=1)
            sb = np.take_along_axis(scores, np.clip(b, 0, None), axis=1)
            winner = np.where(b < 0, a, np.where(a < 0, b, np.where(sa >= sb, a, b)))
            for j in range(a.shape[1]):  # lock results that are already final
                a0, b0 = a[0, j], b[0, j]
                if a0 < 0 or b0 < 0 or (a[:, j] != a0).any() or (b[:, j] != b0).any():
                    continue
                known = decided.get((period, frozenset((tids[a0], tids[b0]))))
                if known is not None:
                    winner[:, j] = index[known]
            if r == rounds:
                finalists = slots
            slots = winner
        champion = slots[:, 0]
    else:
        finalists = order[:, :1]
        champion = order[:, 0]

    def share(arr: np.ndarray) -> dict:
        counts = np.bincount(arr[arr >= 0].ravel(), minlength=n_teams) / n
        return dict(zip(tids, counts.tolist()))

    seed_dist = {tid: (np.bincount(seed_of[:, i], minlength=n_teams) / n).tolist()
                 for tid, i in index.items()}
    out = Outlook(
        tids=tids, n=n, playoff_teams=p_teams, byes=byes,
        playoff=dict(zip(tids, made.mean(axis=0).tolist())),
        bye=dict(zip(tids, (seed_of < byes).mean(axis=0).tolist())) if byes
        else {tid: 0.0 for tid in tids},
        finals=share(finalists), title=share(champion),
        exp_wins=dict(zip(tids, wins.mean(axis=0).tolist())),
        avg_seed=dict(zip(tids, (seed_of.mean(axis=0) + 1).tolist())),
        seed_dist=seed_dist, games_left=len(remaining), week_win=week_win)

    # ---- leverage: what this week's result does to my odds
    if me is not None and my_results:
        first = min(my_results)
        won = my_results[first]
        if 0 < won.sum() < n:
            mine_title = champion == me
            out.leverage = {
                "period": first, "opp_id": week_win[first][0], "p_win": float(won.mean()),
                "playoff_win": float(made[won, me].mean()),
                "playoff_loss": float(made[~won, me].mean()),
                "title_win": float(mine_title[won].mean()),
                "title_loss": float(mine_title[~won].mean()),
            }
    return out


# ------------------------------------------------------------------ rankings

def power_rankings(league: dict, model: Model) -> pd.DataFrame:
    """Teams by forward-looking strength, with all-play record and luck
    (actual wins minus the wins an average schedule would have produced)."""
    record = all_play(league)
    strength = model.strength()
    rows = []
    for t in league["teams"]:
        tid = t["team_id"]
        games = t["wins"] + t["losses"] + t.get("ties", 0)
        ap = record.get(tid, {"w": 0.0, "l": 0.0, "pct": 0.5})
        rows.append({
            "team_id": tid, "name": t["name"], "strength": strength.get(tid, 0.0),
            "ppg": model.ppg.get(tid), "wins": t["wins"], "losses": t["losses"],
            "ap_w": ap["w"], "ap_l": ap["l"], "ap_pct": ap["pct"],
            "luck": (t["wins"] + 0.5 * t.get("ties", 0)) - ap["pct"] * games,
        })
    out = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def remaining_sos(league: dict, model: Model) -> dict:
    """{team_id: average strength of remaining regular-season opponents}."""
    strength = model.strength()
    opps: dict = {}
    for m in league.get("schedule") or []:
        if is_regular(m, league) and not is_decided(m):
            opps.setdefault(m["home_id"], []).append(strength.get(m["away_id"], 0.0))
            opps.setdefault(m["away_id"], []).append(strength.get(m["home_id"], 0.0))
    return {tid: float(np.mean(v)) for tid, v in opps.items() if v}
