"""League: standings, FAAB budgets, this week's matchups, roster viewer."""

import pandas as pd
import streamlit as st

from season_ui import league_ctx, my_team_id, refresh_row

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
refresh_row(league)

st.markdown(f"### 🏆 {league['league_name']}")

# --- standings + FAAB
rows = []
for t in sorted(league["teams"], key=lambda t: (-t["wins"], -t["points_for"])):
    rows.append({
        "Team": t["name"] + (" ⭐" if t["team_id"] == mid else ""),
        "W-L": f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else ""),
        "PF": t["points_for"], "PA": t["points_against"],
        "FAAB left": (max(0, league["faab_budget"] - t["faab_spent"])
                      if league["uses_faab"] else None),
        "Moves": t["moves"],
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             height=len(rows) * 35 + 40,
             column_config={"FAAB left": st.column_config.NumberColumn(
                 "FAAB $", help="Remaining free-agent budget")})

# --- this week's matchups
st.markdown(f"##### Week {league['week']} matchups")
for m in league["schedule"]:
    if m["week"] != league["week"]:
        continue
    h, a = names.get(m["home_id"], "?"), names.get(m["away_id"], "?")
    star = " ⭐" if mid in (m["home_id"], m["away_id"]) else ""
    st.markdown(f"- **{a}** {m['away_pts']} @ **{h}** {m['home_pts']}{star}")

# --- roster viewer
st.markdown("##### Rosters")
sel = st.selectbox("Team", [names[t["team_id"]] for t in league["teams"]],
                   index=None, placeholder="View a team's roster…",
                   label_visibility="collapsed")
if sel:
    tid = next(k for k, v in names.items() if v == sel)
    df = teams[tid].copy()
    df["ROS"] = df["ros_rank"].map(lambda v: int(v) if pd.notna(v) else None)
    st.dataframe(df[["slot", "player", "pos", "team", "bye", "ROS", "injury"]],
                 hide_index=True, width="stretch")
