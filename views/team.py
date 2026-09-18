"""My Team: roster with ROS/weekly ranks, start-sit flags, matchup, news."""

import pandas as pd
import streamlit as st

from data_sources import fetch_player_news
from espn_league import PRO_TEAMS  # noqa: F401  (import keeps module path warm)
from season import start_sit
from season_ui import league_ctx, my_team_id, need_my_team, refresh_row

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

me = next(t for t in league["teams"] if t["team_id"] == mid)
roster = teams[mid]
week = league["week"]

refresh_row(league)

st.markdown(f"### 🏈 {me['name']} · {me['wins']}-{me['losses']}"
            + (f"-{me['ties']}" if me.get("ties") else "")
            + f" · {me['points_for']} PF")

# --- this week's matchup
mu = next((m for m in league["schedule"]
           if m["week"] == week and mid in (m["home_id"], m["away_id"])), None)
if mu:
    opp_id = mu["away_id"] if mu["home_id"] == mid else mu["home_id"]
    my_pts = mu["home_pts"] if mu["home_id"] == mid else mu["away_pts"]
    opp_pts = mu["away_pts"] if mu["home_id"] == mid else mu["home_pts"]
    st.markdown(f"**Week {week}:** you {my_pts} — {opp_pts} "
                f"{names.get(opp_id, 'opponent')}")

# --- start/sit flags
flags = start_sit(roster)
if flags:
    st.markdown("##### ⚠️ Start/sit checks (this week's expert ranks)")
    for f in flags:
        st.warning(f"Consider **{f['in']['player']}** over "
                   f"**{f['out']['player']}** — {f['note']}")
else:
    st.caption("✅ No start/sit flags — your lineup matches this week's expert ranks.")

# --- injury alerts
hurt = roster[roster["injury"].astype(str).str.len() > 0]
for _, p in hurt.iterrows():
    st.error(f"🩹 **{p['player']}** ({p['pos']}) — {p['injury']}", icon="🩹")

# --- roster table
st.markdown("##### Roster")
show = roster.copy()
show["ROS"] = show["ros_rank"].map(lambda v: int(v) if pd.notna(v) else None)
show["Wk"] = show["weekly_rank"].map(lambda v: int(v) if pd.notna(v) else None)
show["Proj"] = show["week_proj"]
slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4, "OP": 5,
              "DST": 6, "K": 7, "BN": 8, "IR": 9}
show = show.sort_values(["starter", "slot"],
                        key=lambda s: s.map(slot_order).fillna(99) if s.name == "slot"
                        else s, ascending=[False, True])
st.dataframe(
    show[["slot", "player", "pos", "team", "bye", "ROS", "ros_pos_rank", "Wk",
          "Proj", "injury"]],
    hide_index=True, width="stretch",
    column_config={
        "slot": st.column_config.TextColumn("Slot", width="small"),
        "player": st.column_config.TextColumn("Player", width="medium"),
        "pos": st.column_config.TextColumn("Pos", width="small"),
        "team": st.column_config.TextColumn("Team", width="small"),
        "bye": st.column_config.NumberColumn("Bye", width="small"),
        "ROS": st.column_config.NumberColumn(
            "ROS", width="small", help="FantasyPros rest-of-season overall rank"),
        "ros_pos_rank": st.column_config.TextColumn("ROS pos", width="small"),
        "Wk": st.column_config.NumberColumn(
            "Wk rank", width="small", help="This week's positional expert rank"),
        "Proj": st.column_config.NumberColumn(
            "Proj", width="small", format="%.1f", help="ESPN projection this week"),
        "injury": st.column_config.TextColumn("Injury", width="small"),
    })

# --- player news
st.markdown("##### 📰 Player news")
sel = st.selectbox("Get latest headlines for…", roster["player"].tolist(),
                   index=None, placeholder="Pick a player…",
                   label_visibility="collapsed")
if sel:
    with st.spinner("Grabbing headlines…"):
        news = fetch_player_news(sel)
    if news:
        for n in news:
            extras = " · ".join(x for x in (n["source"], n["when"]) if x)
            st.markdown(f"- [{n['title']}]({n['link']})"
                        + (f" <span style='opacity:.6'>({extras})</span>" if extras else ""),
                        unsafe_allow_html=True)
    else:
        st.caption("No recent headlines found.")
