"""DraftAid: a season-long fantasy football assistant for one ESPN league.

Entry point: page config, the active theme, and navigation. Pages live in
views/, fantasy logic and data sources in core/, the design system in ui/.
"""

from pathlib import Path

import streamlit as st

from core import espn
from ui import theme

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="DraftAid", page_icon=str(ASSETS / "mark.svg"), layout="wide")
theme.setup(saved=espn.load_config().get("theme"))
st.logo(theme.logo_svg(theme.current()),
        icon_image=theme.logo_svg(theme.current(), mark_only=True), size="large")

pg = st.navigation({
    "Season": [
        st.Page("views/team.py", title="My Team", icon=":material/shield:", default=True),
        st.Page("views/playoffs.py", title="Playoff Picture", icon=":material/emoji_events:"),
        st.Page("views/waivers.py", title="Waiver Wire", icon=":material/person_add:"),
        st.Page("views/trades.py", title="Trade Finder", icon=":material/swap_horiz:"),
        st.Page("views/league.py", title="League", icon=":material/leaderboard:"),
    ],
    "Archive": [
        st.Page("views/draft_room.py", title="Draft Room", icon=":material/grid_view:"),
    ],
    "Settings": [
        st.Page("views/setup.py", title="Settings", icon=":material/settings:"),
    ],
})
pg.run()
