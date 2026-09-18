"""DraftAid -- season-long fantasy football assistant for Jimmy's league.

Entry point: page config, shared responsive CSS, and navigation between the
season tools (ESPN-connected) and the original live Draft Room.
"""

import streamlit as st

st.set_page_config(page_title="DraftAid", page_icon="🏈", layout="wide")

# Mobile-first tweaks shared by every page: keep row-action columns
# side-by-side on phones (Streamlit stacks columns vertically below ~640px by
# default), tighten padding, and swap board rows between a wide single-line
# layout (.da-d) and a compact two-line layout (.da-m) based on screen width.
st.markdown("""
<style>
div[data-testid="stColumn"] button { min-width: 42px; }
.da-m { display: none; }
@media (max-width: 700px) {
  div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.3rem !important; }
  div[data-testid="stColumn"] { min-width: 0 !important; }
  .block-container { padding: 0.6rem 0.6rem 3rem !important; }
  div[data-testid="stColumn"] button { padding: 0.3rem 0.45rem !important; }
  .da-d { display: none !important; }
  .da-m { display: flex !important; }
  /* In board rows and suggestion rows, give the action-button columns a fixed
     width and let the text column take the rest, so buttons never overlap. */
  div[data-testid="stHorizontalBlock"]:has(.da-m) > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"]:has(.da-sg) > div[data-testid="stColumn"] {
    flex: 0 0 46px !important; min-width: 46px !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.da-m) > div[data-testid="stColumn"]:first-child,
  div[data-testid="stHorizontalBlock"]:has(.da-sg) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important; min-width: 0 !important;
  }
}
</style>
""", unsafe_allow_html=True)

pg = st.navigation({
    "Season": [
        st.Page("views/team.py", title="My Team", icon="🏈", default=True),
        st.Page("views/waivers.py", title="Waivers & FAAB", icon="💰"),
        st.Page("views/trades.py", title="Trade Finder", icon="🔁"),
        st.Page("views/league.py", title="League", icon="🏆"),
    ],
    "Draft": [
        st.Page("views/draft_room.py", title="Draft Room", icon="🎯"),
    ],
    "Settings": [
        st.Page("views/setup.py", title="League Setup", icon="⚙️"),
    ],
})
pg.run()
