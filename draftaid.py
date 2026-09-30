"""DraftAid -- season-long fantasy football assistant for Jimmy's league.

Entry point: page config, the shared design system (theme.py), and navigation
between the season tools (ESPN-connected) and the original live Draft Room.
"""

from pathlib import Path

import streamlit as st

import theme

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="DraftAid", page_icon=str(ASSETS / "mark.svg"),
                   layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "mark.svg"), size="large")
theme.inject()

pg = st.navigation({
    "Season": [
        st.Page("views/team.py", title="My Team", icon=":material/shield:", default=True),
        st.Page("views/waivers.py", title="Waiver Wire", icon=":material/person_add:"),
        st.Page("views/trades.py", title="Trade Finder", icon=":material/swap_horiz:"),
        st.Page("views/league.py", title="League", icon=":material/leaderboard:"),
    ],
    "Archive": [
        st.Page("views/draft_room.py", title="Draft Room", icon=":material/grid_view:"),
    ],
    "Settings": [
        st.Page("views/setup.py", title="League Setup", icon=":material/settings:"),
    ],
})
pg.run()
