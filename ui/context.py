"""Shared page context: league + enriched rosters + season outlook, and the
page header every season view starts with."""

import time
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from core import espn, fantasypros, market, sleeper
from core import outlook as sim
from core.players import Sources, enrich
from ui import theme


@dataclass
class Ctx:
    cfg: dict
    league: dict
    teams: dict                     # team_id -> enriched roster DataFrame
    names: dict                     # team_id -> team name
    src: Sources
    my_id: int | None = None
    _memo: dict = field(default_factory=dict)

    @property
    def me(self) -> dict:
        return next(t for t in self.league["teams"] if t["team_id"] == self.my_id)

    @property
    def roster(self) -> pd.DataFrame:
        return self.teams[self.my_id]

    @property
    def lineup(self) -> dict:
        return self.league.get("lineup") or {}

    @property
    def weeks(self) -> list[int]:
        """NFL weeks from now through the end of the fantasy playoffs."""
        return season_weeks(self.league)

    def free_agents(self) -> pd.DataFrame:
        """The league's available players, enriched like the rosters."""
        if "fas" not in self._memo:
            self._memo["fas"] = enrich(espn.get_free_agents(self.cfg, self.league), self.src)
        return self._memo["fas"]

    # ---- season outlook (simulation)

    def model(self, teams: dict | None = None) -> sim.Model:
        """Strength model for the league as it is, or for what-if rosters."""
        if teams is None and "model" in self._memo:
            return self._memo["model"]
        base_proj = self._memo.get("proj")
        if base_proj is None:
            base_proj = sim.project_teams(self.teams, self.lineup, self.weeks,
                                              self.league["week"])
            self._memo["proj"] = base_proj
        proj = base_proj
        if teams is not None:   # only the changed rosters need re-projecting
            proj = {**base_proj, **sim.project_teams(teams, self.lineup, self.weeks,
                                                         self.league["week"])}
        model = sim.build_model(self.league, proj, self.weeks)
        if teams is None:
            self._memo["model"] = model
        return model

    def outlook(self) -> sim.Outlook | None:
        """Playoff / title odds for every team, or None if it can't be run."""
        if "outlook" not in self._memo:
            try:
                self._memo["outlook"] = sim.simulate(self.league, self.model(),
                                                         my_id=self.my_id)
            except Exception:
                self._memo["outlook"] = None
        return self._memo["outlook"]

    def what_if(self, teams: dict) -> sim.Outlook | None:
        """Odds if some rosters changed, on the same random draws as
        `outlook()` so the two are directly comparable."""
        try:
            return sim.simulate(self.league, self.model(teams), my_id=self.my_id)
        except Exception:
            return None


def season_weeks(league: dict) -> list[int]:
    last = max(sim.playoff_weeks(league) or [league.get("regular_season_weeks", 14)])
    return list(range(league["week"], min(18, max(last, league["week"])) + 1))


# ------------------------------------------------------------------ loading

def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def load_sources(league: dict, season: int) -> Sources:
    weeks = season_weeks(league)
    proj = _safe(lambda: sleeper.fetch_projections(season, tuple(weeks)),
                 {"pts": {}, "opp": {}, "pos": {}, "weeks": []})
    return Sources(
        ros=fantasypros.fetch_ros(),
        weekly=fantasypros.fetch_weekly(),
        sleeper=_safe(sleeper.fetch_players, pd.DataFrame()),
        trending=_safe(sleeper.fetch_trending, {}),
        market=_safe(lambda: market.fetch_values(len(league["teams"]) or 12), pd.DataFrame()),
        proj=proj["pts"], opp=proj["opp"], proj_pos=proj["pos"], weeks=proj["weeks"],
        playoff_weeks=sim.playoff_weeks(league),
    )


def _prompt(title: str, body: str, link_label: str):
    theme.page_header(title, ["DraftAid"])
    theme.empty(body)
    st.page_link("views/setup.py", label=link_label, icon=":material/arrow_forward:")
    st.stop()


def _unavailable(title: str, body: str):
    """A data source failed: say so and offer a retry instead of a traceback."""
    theme.page_header(title, ["DraftAid"])
    theme.empty(body)
    if st.button("Try again", icon=":material/refresh:", key="retry_load"):
        st.cache_data.clear()
        st.session_state.pop("_ctx", None)
        st.rerun()
    st.stop()


def my_team_id(cfg: dict, league: dict) -> int | None:
    """Configured team id, else auto-detect from the SWID cookie."""
    tid = cfg.get("my_team_id")
    if tid is not None and any(t["team_id"] == tid for t in league["teams"]):
        return tid
    swid = (cfg.get("swid") or "").strip().upper()
    if swid:
        swid = "{" + swid.strip("{}") + "}"
        for t in league["teams"]:
            if swid in t.get("owners", []):
                return t["team_id"]
    return None


def load(require_team: bool = True) -> Ctx:
    """League + enriched rosters for a season page, or stop with a setup hint."""
    cfg = espn.load_config()
    if not cfg.get("league_id"):
        _prompt("Connect your league",
                "<b>DraftAid isn't connected to your ESPN league yet.</b> Add your "
                "league ID (plus the espn_s2 / SWID cookies if it's private) and "
                "every page here fills in.", "Open Settings")
    try:
        with st.spinner("Syncing with ESPN…"):
            league = espn.get_league(cfg)
    except Exception as e:
        _prompt("League unavailable",
                f"<b>Couldn't load your ESPN league.</b> {theme.esc(e)}", "Check Settings")

    # One memo per ESPN pull: page reruns reuse the sources, joins, and simulation.
    stamp = (cfg.get("league_id"), league.get("fetched"))
    cached = st.session_state.get("_ctx")
    if cached and cached[0] == stamp:
        ctx = cached[1]
    else:
        try:
            with st.spinner("Pulling rankings, projections, and market data…"):
                src = load_sources(league, int(cfg.get("year") or 2026))
            everyone = [{**p, "_team": t["team_id"]}
                        for t in league["teams"] for p in t["roster"]]
            joined = enrich(everyone, src)
        except Exception as e:
            _unavailable("Rankings unavailable",
                         "<b>Couldn't load the expert rankings.</b> FantasyPros may be "
                         f"briefly down; nothing is wrong with your league. ({theme.esc(e)})")
        teams = {t["team_id"]: (joined[joined["_team"] == t["team_id"]]
                                .drop(columns="_team").reset_index(drop=True)
                                if not joined.empty else pd.DataFrame())
                 for t in league["teams"]}
        ctx = Ctx(cfg=cfg, league=league, teams=teams, src=src,
                  names={t["team_id"]: t["name"] for t in league["teams"]},
                  my_id=my_team_id(cfg, league))
        st.session_state["_ctx"] = (stamp, ctx)
    mine = my_team_id(cfg, league)
    if mine != ctx.my_id:
        ctx._memo.pop("outlook", None)   # leverage and my-games odds are per team
    ctx.cfg, ctx.my_id = cfg, mine
    if require_team and ctx.my_id is None:
        _prompt("Which team is yours?",
                "<b>I don't know which team is yours yet.</b> Pick it in Settings and "
                "your roster, waivers, and trades will be tailored to it.", "Open Settings")
    return ctx


# ------------------------------------------------------------------ page chrome

def _synced(league: dict) -> str:
    ts = league.get("fetched")
    if not ts:
        return "Live"
    mins = int((time.time() - ts) // 60)
    return "Synced just now" if mins < 1 else f"Synced {mins} min ago"


def header(ctx: Ctx, title: str, sub: str | None = None):
    """Page title block with league/week context and a refresh button."""
    league = ctx.league
    c1, c2 = st.columns([8, 1], vertical_alignment="top", gap="small")
    with c1:
        theme.page_header(title, [league.get("league_name", ""),
                                  f"Week {league.get('week', '?')}"],
                          sub=sub, live=_synced(league))
    with c2:
        if st.button("", icon=":material/refresh:", key="season_refresh",
                     help="Re-pull ESPN, rankings, projections, and market data now",
                     width="stretch"):
            st.cache_data.clear()
            st.session_state.pop("_ctx", None)
            st.rerun()


def days_to_deadline(league: dict) -> float | None:
    """Days until the league's trade deadline (negative once it has passed)."""
    deadline = league.get("trade_deadline")
    return (deadline - time.time()) / 86400 if deadline else None
