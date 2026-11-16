# DraftAid

A season-long fantasy football assistant for one ESPN league, with the live draft
room it started as. Built with Streamlit; the app entry point is `draftaid.py`.

## What it does

Connect your ESPN league in **Settings** (league ID, plus the `espn_s2` / `SWID`
cookies for a private league; instructions are in the app). Then:

- **My Team**: this week's matchup with a live win probability, your playoff and
  title odds, what this week's game is worth, lineup checks (expert swaps, points
  left on the bench, injuries, upcoming byes), and the roster with rest-of-season
  and playoff-week ranks.
- **Playoff Picture**: the rest of the season simulated 10,000 times. Playoff, bye,
  and title odds for every team, your remaining games, power rankings with all-play
  record and luck, your projected lineup for each playoff week, rivals' players who
  get better in the playoffs, and stashes on the wire.
- **Waiver Wire**: every free agent tried in every add/drop swap against your
  lineup, with the right drop and a FAAB bid priced to the rivals who would start
  him. Plus the backups to your running backs and playoff-week stashes.
- **Trade Finder**: deals that fill a need on both rosters and look fair on two
  independent scales (the real-trade market and expert consensus), each measured by
  what it does to your playoff and title odds. Also a trade-deadline countdown,
  sell-high / buy-low lists, and a league-wide positional strength table.
- **League**: standings with playoff odds and FAAB, the scoreboard, any roster.
- **Draft Room** (archive): the live draft board: blended rankings, one-tap pick
  logging, pick suggestions, tiers, and practice mode.

Three themes (Midnight, Daylight, Dusk) are switchable in Settings.

## How the season simulation works

For every remaining week, each team's best lineup is built from Sleeper's weekly
projections for its current roster, so byes, injuries, and playoff-week matchups are
in the number. Projections are rescaled to how the league actually scores and each
team is nudged toward its own scoring average. The remaining schedule and the
playoff bracket are then played out with a weekly spread measured from the league's
real scores. Games in progress use ESPN's live projected finals. A what-if (a trade)
is re-simulated on the same random draws, so its effect on the odds is directly
comparable. See `core/outlook.py`.

## Layout

```
draftaid.py          entry point: page config, theme, navigation
views/               one file per page
core/                data sources and fantasy logic (no UI)
  espn.py              ESPN league client and parsing
  fantasypros.py       expert consensus: draft, rest-of-season, weekly
  sleeper.py           players, trending adds, weekly projections
  market.py            FantasyCalc trade-market values
  players.py           joins every source onto ESPN rosters
  lineup.py            the lineup model everything else builds on
  waivers.py startsit.py trades.py
  outlook.py           season simulation and power rankings
  draft.py board.py    draft-room math and the pre-draft board
ui/                  design system (theme.py) and shared page context
tests/               unit tests for core/, plus every page run offline
tools/make_fixture.py  builds a demo league from live rankings
```

## Run locally

```
pip install -r requirements.txt
streamlit run draftaid.py
```

No ESPN credentials handy? Build a demo league (real players and ranks, made-up
league) and point the app at it:

```
python tools/make_fixture.py
DRAFTAID_FIXTURE=.fixture streamlit run draftaid.py
```

Tests:

```
pip install -r requirements-dev.txt
python -m pytest
```

## Deploy

This repo backs the **draftaid** Streamlit Community Cloud app; pushing to `main`
redeploys it. Credentials are stored in `league_config.json` (gitignored) or, for a
connection that survives restarts, in the app's Secrets:

```toml
[espn]
league_id = "1234567"
year = 2026
espn_s2 = "..."
swid = "{...}"
my_team_id = 8
theme = "midnight"   # or "daylight", "dusk"
```

## Data sources

All free; none needs an API key.

| Source | What it provides | Cached |
| --- | --- | --- |
| ESPN (your league) | Rosters, lineups, schedule, scores, FAAB, free agents, projections in your scoring | 3 min |
| FantasyPros | Expert consensus: rest-of-season, weekly, and pre-draft ranks | 1 h (15 min pre-draft) |
| Sleeper | Week-by-week projections through the playoffs, injuries, depth charts, trending adds | 3 h / 6 h / 15 min |
| FantasyCalc | Trade-market values and 30-day trend | 6 h |
| FantasyFootballCalculator | Real-draft ADP (draft room) | 15 min |
| Google News | Player headlines | 30 min |

The refresh button on any season page clears every cache and re-pulls.
