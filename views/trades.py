"""Trade finder: surplus/deficit matching across the league."""

import pandas as pd
import streamlit as st

from season import positional_needs, trade_ideas
from season_ui import league_ctx, my_team_id, need_my_team, refresh_row

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

refresh_row(league)
st.markdown("### 🔁 Trade Finder")

st.caption("A deal only shows up if it looks fair on **two independent value "
           "scales** — FantasyCalc's real-trade market data AND expert "
           "rest-of-season consensus — and fills a real need on both rosters. "
           "When the sources disagree about a player, nothing is proposed. "
           "2-for-1s assume the usual consolidation premium.")
ideas = trade_ideas(teams, mid, names)
if not ideas:
    st.info("No realistic trade fits right now — either your starters are at or "
            "above league average everywhere (nothing worth trading for), or "
            "no opponent is both deep where you're thin AND thin where you're "
            "deep. Check the strength table below to hunt manually.")
for i in ideas:
    get_p = i["get"]
    ratio = i["ratio"]
    if i["kind"] == "1-for-1":
        verdict = ("⚖️ dead even" if 0.97 <= ratio <= 1.07
                   else "👍 slight value win for you" if ratio < 0.97
                   else "🤝 you pay a little extra")
    else:
        verdict = "🤝 you pay the standard 2-for-1 premium"
    gives = "  \n".join(
        f"🔴 **GIVE {g['player']}** ({g['pos']}, {g['team']}, ROS {int(g['ros_rank'])})"
        for g in i["give"])
    inj = f" · 🩹 {get_p['injury']}" if get_p.get("injury") else ""
    with st.container(border=True):
        st.markdown(
            f"**{i['kind']} with {i['team']}** — {verdict}\n\n"
            f"🟢 **GET {get_p['player']}** ({get_p['pos']}, {get_p['team']}, "
            f"ROS {int(get_p['ros_rank'])}{inj})  \n"
            f"{gives}  \n"
            f"<span style='font-size:12.5px;opacity:.85'>"
            f"**For you:** {i['why_me']}<br>"
            f"**For them:** {i['why_them']}</span>",
            unsafe_allow_html=True)

# --- league positional strength map
st.markdown("##### Positional strength around the league")
st.caption("Average ROS rank of each team's starters — lower is stronger. "
           "Find the team weak where you're deep.")
rows = []
for tid, df in teams.items():
    needs = positional_needs(df)
    rows.append({"Team": names[tid] + (" ⭐" if tid == mid else ""),
                 **{pos: round(n["starter_avg"]) for pos, n in needs.items()},
                 "Depth+": sum(n["depth"] for n in needs.values())})
grid = pd.DataFrame(rows)
st.dataframe(grid, hide_index=True, width="stretch",
             height=len(rows) * 35 + 40,
             column_config={"Depth+": st.column_config.NumberColumn(
                 "Depth+", help="Startable players beyond the starting lineup")})
