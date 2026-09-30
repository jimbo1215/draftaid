"""My Team: matchup, lineup checks, injuries, the roster, and player news."""

import math

import pandas as pd
import streamlit as st

import theme as t
from data_sources import fetch_player_news
from season import start_sit
from season_ui import header, league_ctx, my_team_id, need_my_team

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

me = next(x for x in league["teams"] if x["team_id"] == mid)
roster = teams[mid]
week = league["week"]

standings = sorted(league["teams"], key=lambda x: (-x["wins"], -x["points_for"]))
place = next(i for i, x in enumerate(standings) if x["team_id"] == mid) + 1
record = f"{me['wins']}-{me['losses']}" + (f"-{me['ties']}" if me.get("ties") else "")
header(league, me["name"])


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _proj_total(df: pd.DataFrame) -> float:
    live = df[df["starter"] & (df["slot"] != "IR")]
    return float(live["week_proj"].fillna(0).sum()) if "week_proj" in live else 0.0


# --- this week's matchup, score-bug style
mu = next((m for m in league["schedule"]
           if m["week"] == week and mid in (m["home_id"], m["away_id"])), None)
if mu:
    opp_id = mu["away_id"] if mu["home_id"] == mid else mu["home_id"]
    my_pts = mu["home_pts"] if mu["home_id"] == mid else mu["away_pts"]
    opp_pts = mu["away_pts"] if mu["home_id"] == mid else mu["home_pts"]
    my_proj = _proj_total(roster)
    opp_proj = _proj_total(teams[opp_id]) if opp_id in teams else 0.0
    opp = next((x for x in league["teams"] if x["team_id"] == opp_id), {})
    opp_rec = f"{opp.get('wins', 0)}-{opp.get('losses', 0)}"
    # Pre-game projections: weekly fantasy scores swing ~25 pts per team, so
    # the margin's spread is ~35. Once points are on the board, say less.
    if my_proj and opp_proj and my_pts == 0 and opp_pts == 0:
        wp = 0.5 * (1 + math.erf((my_proj - opp_proj) / (35 * math.sqrt(2))))
        wp_html = (f"<div class='wp'><i style='width:{wp * 100:.0f}%'></i></div>"
                   f"<div class='wpl'><span class='{'lead' if wp >= .5 else ''}'>"
                   f"{wp * 100:.0f}% win prob</span><span>{(1 - wp) * 100:.0f}%</span></div>")
    else:
        wp_html = ("<div class='wpl'><span>Live</span><span>Projections are "
                   "pre-game</span></div>")
    lead_style = " style='color:var(--volt)'" if my_pts > opp_pts else ""
    t.html(
        f"<div class='da-mu'><div class='rowm'>"
        f"<div class='side'><div class='who'>You · {record}</div>"
        f"<div class='tm'>{t.esc(me['name'])}</div>"
        f"<div class='pts da-num'{lead_style}>{my_pts:.1f}</div>"
        f"<div class='pj'>Proj {my_proj:.1f}</div></div>"
        f"<div class='vs'>WK {week}</div>"
        f"<div class='side r'><div class='who'>{opp_rec} · Opponent</div>"
        f"<div class='tm'>{t.esc(names.get(opp_id, 'Opponent'))}</div>"
        f"<div class='pts da-num'>{opp_pts:.1f}</div>"
        f"<div class='pj'>Proj {opp_proj:.1f}</div></div></div>{wp_html}</div>")

t.tiles([
    {"k": "Standing", "v": _ordinal(place), "d": f"of {len(standings)} · {record}"},
    {"k": "Points for", "v": f"{me['points_for']:.0f}",
     "d": f"{me['points_for'] / max(1, week - 1):.1f} per week" if week > 1 else ""},
    {"k": "Points against", "v": f"{me['points_against']:.0f}"},
    {"k": "FAAB left", "v": f"${max(0, league['faab_budget'] - me['faab_spent'])}",
     "d": f"{me['moves']} moves made"} if league["uses_faab"] else
    {"k": "Moves", "v": str(me["moves"])},
])

# --- lineup checks
flags = start_sit(roster, league.get("lineup"))
hurt_starters = roster[(roster["injury"].astype(str).str.len() > 0) & roster["starter"]
                       & (roster["slot"] != "IR")]
t.section("Lineup check", "this week's expert ranks")
if not flags and hurt_starters.empty:
    t.alert("good", "All clear", "Your lineup matches this week's expert ranks and "
            "no starter is on the injury report.")
for f in flags:
    if f["in"] is None:
        t.alert("bad", "Empty", f"<b>{t.esc(f['out']['player'])}</b> is in your lineup",
                t.esc(f["note"]))
    else:
        t.alert("", "Swap", f"Start <b>{t.esc(f['in']['player'])}</b> over "
                f"<b>{t.esc(f['out']['player'])}</b>", t.esc(f["note"]))
for _, p in hurt_starters.iterrows():
    serious = str(p["injury"]).title() not in ("Questionable", "Day To Day")
    t.alert("bad" if serious else "", "Injury" if serious else "Monitor",
            f"<b>{t.esc(p['player'])}</b> is in your lineup", t.esc(p["injury"]))

# --- roster
t.section("Roster", "ROS = rest-of-season positional rank · Proj = ESPN, your scoring")
t.html(t.roster_table(roster))

# --- player news
t.section("Player news")
sel = st.selectbox("Headlines for", roster["player"].tolist(), index=None,
                   placeholder="Pick a player for the latest headlines…",
                   label_visibility="collapsed")
if sel:
    with st.spinner("Grabbing headlines…"):
        news = fetch_player_news(sel)
    if news:
        t.html(t.news_list(news))
    else:
        st.caption("No recent headlines found.")
