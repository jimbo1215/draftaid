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

st.caption("Fair deals must look even on **two independent value scales** — "
           "FantasyCalc's real-trade market data AND expert rest-of-season "
           "consensus — and fill a real need on both rosters. "
           "2-for-1s assume the usual consolidation premium.")
steals_on = st.toggle("😈 Include steals — offers tilted your way (they'll "
                      "often decline, but asking is free)", value=True)
ideas = trade_ideas(teams, mid, names, top_n=12 if steals_on else 8,
                    include_steals=steals_on)

needs_all = {tid: positional_needs(df) for tid, df in teams.items()}
if not ideas:
    _positions = ("QB", "RB", "WR", "TE")
    _avg = {p: sum(n[p]["starter_avg"] for n in needs_all.values()) / len(needs_all)
            for p in _positions}
    _weak = [p for p in _positions
             if needs_all[mid][p]["starter_avg"] - _avg[p] > 5]
    if not _weak:
        st.info("**No fair-trade fits, and here's why:** your starters are at "
                "or above league average at every position, so there's no hole "
                "worth paying market price to fix. That changes when injuries "
                "or busts drop one of your position groups below average — or "
                "when an opponent gets desperate.")
    else:
        st.info(f"**No fits right now:** you're thin at {', '.join(_weak)}, but "
                "no opponent is both deep there AND thin where you have spare "
                "depth, at a price both value sources call fair. Check back "
                "after injuries shake up rosters, or hunt manually with the "
                "strength table below.")
for i in ideas:
    get_p = i["get"]
    ratio = i["ratio"]
    if i["kind"] == "steal":
        verdict = "😈 tilted your way — costs nothing to ask"
    elif i["kind"] == "1-for-1":
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
for tid in teams:
    needs = needs_all[tid]
    rows.append({"Team": names[tid] + (" ⭐" if tid == mid else ""),
                 **{pos: round(n["starter_avg"]) for pos, n in needs.items()},
                 "Depth+": sum(n["depth"] for n in needs.values())})
grid = pd.DataFrame(rows)
st.dataframe(grid, hide_index=True, width="stretch",
             height=len(rows) * 35 + 40,
             column_config={"Depth+": st.column_config.NumberColumn(
                 "Depth+", help="Startable players beyond the starting lineup")})
