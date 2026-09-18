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

ideas = trade_ideas(teams, mid, names)
if not ideas:
    st.info("No obvious trade fits right now — your roster is balanced, or "
            "opponents don't have tradable surplus where you're thin.")
for i in ideas:
    get_p, give_p = i["get"], i["give"]
    edge = i["edge"]
    verdict = ("👍 you'd win this" if edge > 8
               else "⚖️ fair swap" if edge > -8 else "needs a sweetener from them")
    with st.container(border=True):
        st.markdown(
            f"**Trade with {i['team']}** — {verdict}\n\n"
            f"🟢 **GET {get_p['player']}** ({get_p['pos']}, {get_p['team']}, "
            f"ROS {int(get_p['ros_rank'])})  \n"
            f"🔴 **GIVE {give_p['player']}** ({give_p['pos']}, {give_p['team']}, "
            f"ROS {int(give_p['ros_rank'])})  \n"
            f"<span style='font-size:12.5px;opacity:.8'>{i['why']}</span>",
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
