"""Waiver wire: ranked free agents with suggested FAAB bids."""

import pandas as pd
import streamlit as st

from espn_league import get_free_agents
from season import waiver_targets
from season_ui import league_ctx, my_team_id, need_my_team, refresh_row

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

me = next(t for t in league["teams"] if t["team_id"] == mid)
refresh_row(league)

budget = league["faab_budget"]
faab_left = max(0, budget - me["faab_spent"])
league_left = [max(0, budget - t["faab_spent"]) for t in league["teams"]]
avg_left = sum(league_left) / len(league_left) if league_left else 0

st.markdown("### 💰 Waivers & FAAB")
if league["uses_faab"]:
    st.markdown(f"**Your FAAB: ${faab_left}** of ${budget} left · "
                f"league average ${avg_left:.0f} left")
    if faab_left > avg_left + 10:
        st.caption("You have more budget than most — you can win any bid you want.")
    elif faab_left < avg_left - 10:
        st.caption("You're below the league average — save bids for real upgrades.")
else:
    st.caption("This league uses waiver priority, not FAAB — suggestions show "
               "claim priority instead of dollars.")

with st.spinner("Ranking the wire…"):
    from data_sources import fetch_fp_ros, fetch_fp_weekly, fetch_sleeper_players, \
        fetch_sleeper_trending
    from season import enrich
    fas_raw = get_free_agents(cfg, league["week"])
    try:
        sleeper, trending = fetch_sleeper_players(), fetch_sleeper_trending()
    except Exception:
        sleeper, trending = pd.DataFrame(), {}
    fas = enrich(fas_raw, fetch_fp_ros(), fetch_fp_weekly(), sleeper, trending)
    targets = waiver_targets(fas, teams[mid], faab_left, league["uses_faab"])

pos_filter = st.multiselect("Position", ["QB", "RB", "WR", "TE", "K", "DST"],
                            default=[], placeholder="All positions",
                            label_visibility="collapsed")
shown = targets[targets["pos"].isin(pos_filter)] if pos_filter else targets

if shown.empty:
    st.caption("No ranked free agents found — is the league connected?")
for _, p in shown.iterrows():
    ros = f"ROS {int(p['ros_rank'])}" if pd.notna(p["ros_rank"]) else "unranked"
    wk = f" · wk {p['pos']}{int(p['weekly_rank'])}" if pd.notna(p["weekly_rank"]) else ""
    own = f" · {p['pct_owned']:.0f}% rostered" if pd.notna(p.get("pct_owned")) else ""
    inj = f" · 🩹 {p['injury']}" if p["injury"] else ""
    with st.container(border=True):
        st.markdown(
            f"<div class='da-sg'><b>{p['player']}</b> "
            f"<img src='https://sleepercdn.com/images/team_logos/nfl/{str(p['team']).lower()}.png' "
            f"width='16' style='vertical-align:-3px'> "
            f"({p['pos']}, {p['team']}) — <b>bid {p['bid']}</b><br>"
            f"<span style='font-size:12.5px;opacity:.8'>{ros}{wk}{own}{inj} · "
            f"{p['why']}</span></div>",
            unsafe_allow_html=True)

drops = targets.attrs.get("drops")
if drops is not None and len(drops):
    st.markdown("##### 🗑 Your most droppable players")
    for _, d in drops.iterrows():
        ros = f"ROS {int(d['ros_rank'])}" if pd.notna(d["ros_rank"]) else "unranked"
        st.caption(f"• {d['player']} ({d['pos']}, {d['team']}) — {ros}")
st.caption("Bids are sized against your **remaining** budget: ~40–55% for a "
           "potential league-winner, ~20–30% for a solid starter, ~10% for depth, "
           "$1–2 for streamers. Adjust for how badly you need the position.")
