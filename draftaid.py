# DraftAid — streamlined features per your league
# Changes based on your ask:
#   • Default Bench slots = 5
#   • Headshots: improved reliability via Sleeper CDN → ESPN slug → hide on error
#   • College + Experience column (e.g., "5th-year out of Clemson") via Sleeper API lookup (cached)
#   • Compact news icon stays (📰) + on-demand News Preview panel (see headlines before drafting)
#
# Usage: save as draftaid.py and run `streamlit run draftaid.py`. Upload your FantasyPros CSV.
# Note: This app fetches public data from Sleeper and Google News RSS at runtime.

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote_plus

st.set_page_config(page_title="DraftAid+", layout="wide")

# ---------------------------
# Config / Column mapping
# ---------------------------
COLUMN_MAP = {
    "rank": ["Rank", "Overall Rank", "ECR"],
    "player": ["Player", "Name"],
    "team": ["Team"],
    "pos": ["Pos", "Position"],
    "bye": ["Bye", "Bye Week"],
    "adp": ["ADP", "Avg. Draft Pos", "AVG ADP"],
    "ecr": ["ECR", "Expert Consensus Rank"],
    "tier": ["Tier"],
    "posrank": ["PosRank", "Position Rank"],
}

DEFAULT_ROSTER = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,   # counts RB/WR/TE
    "DST": 1,
    "K": 1,
    "Bench": 5,  # ← your league default
}

FLEX_ELIGIBLE = {"RB", "WR", "TE"}

# ---------------------------
# Helpers
# ---------------------------

def coalesce(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    # Normalize columns
    mapping = {coalesce(df, v): k for k, v in COLUMN_MAP.items() if coalesce(df, v)}
    df = df.rename(columns=mapping)
    # Ensure required
    required = ["player", "team", "pos"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV missing required column mapped to '{r}'. Check COLUMN_MAP.")
    # Fill optionals
    for opt in ["rank", "bye", "adp", "ecr", "tier", "posrank"]:
        if opt not in df.columns:
            df[opt] = np.nan
    # Clean types
    for num in ["rank", "bye", "adp", "ecr", "posrank"]:
        df[num] = pd.to_numeric(df[num], errors='coerce')
    df["tier"] = df["tier"].fillna(0).astype(int)
    df["team"] = df["team"].fillna("")
    # Split positions like "WR/RB"
    df["pos"] = df["pos"].astype(str).str.split(r"[/+]?")
    return df

# ---------------------------
# Sleeper players lookup (college, experience, headshots)
# ---------------------------

@st.cache_data(show_spinner=False)
def load_sleeper_players():
    import requests
    try:
        r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {}
    # Build simple name → record map (prefer active players with full_name)
    index = {}
    for pid, rec in data.items():
        nm = (rec.get("full_name") or rec.get("first_name","") + " " + rec.get("last_name"," ")).strip()
        if not nm:
            continue
        key = re.sub(r"\s+", " ", nm.lower())
        # prefer players with years_exp defined
        if key not in index or (rec.get("years_exp") is not None and not index[key].get("years_exp")):
            rec["player_id"] = pid
            index[key] = rec
    return index

SLEEPER = load_sleeper_players()


def normalize_name(name: str):
    return re.sub(r"\s+", " ", name.strip().lower())


def find_sleeper(name: str):
    if not name:
        return None
    key = normalize_name(name)
    rec = SLEEPER.get(key)
    if rec:
        return rec
    # fallback: strip suffixes (Jr., Sr., III), remove punctuation
    key2 = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?$", "", key).strip()
    key2 = re.sub(r"[^a-z0-9 ]", "", key2)
    for k, v in SLEEPER.items():
        k2 = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?$", "", k).strip()
        k2 = re.sub(r"[^a-z0-9 ]", "", k2)
        if k2 == key2:
            return v
    return None


def ordinal(n: int):
    if n % 100 in (11,12,13):
        suffix = "th"
    else:
        suffix = {1:"st",2:"nd",3:"rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

NFL_FALLBACK = "https://static.www.nfl.com/image/private/t_headshot_desktop/league/placeholder"


def headshot_urls(name: str):
    # Try Sleeper thumb by player_id, then ESPN slug, then show nothing (fallback hidden)
    urls = []
    rec = find_sleeper(name)
    if rec and rec.get("player_id"):
        urls.append(f"https://sleepercdn.com/content/nfl/players/thumb/{rec['player_id']}.jpg")
    slug = re.sub(r"[^a-z0-9]", "-", name.lower())
    urls.append(f"https://a.espncdn.com/i/headshots/nfl/players/full/{slug}.png")
    # NFL placeholder left out so we can hide broken <img> via onerror
    return urls


@st.cache_data(show_spinner=False)
def get_bio_for_name(name: str):
    rec = find_sleeper(name)
    college = rec.get("college") if rec else None
    years = rec.get("years_exp") if rec else None
    exp_str = None
    if years is not None:
        try:
            y = int(years)
            exp_str = f"{ordinal(y)}-year"
        except Exception:
            pass
    return college, exp_str


@st.cache_data(show_spinner=False)
def augment_with_bio(df: pd.DataFrame) -> pd.DataFrame:
    colleges = []
    exps = []
    for nm in df["player"].fillna(""):
        c, e = get_bio_for_name(nm)
        colleges.append(c)
        exps.append(e)
    out = df.copy()
    out["college"] = colleges
    out["experience_str"] = exps
    return out

# ---------------------------
# News helpers (compact link + preview panel)
# ---------------------------

@st.cache_data(show_spinner=False)
def make_news_link(name):
    q = quote_plus(f"{name} NFL fantasy injury")
    return f"https://news.google.com/search?q={q}&hl=en-US&gl=US&ceid=US:en"


@st.cache_data(show_spinner=False)
def fetch_news_items(name: str, limit: int = 8):
    import requests
    import xml.etree.ElementTree as ET
    q = quote_plus(f"{name} NFL fantasy OR injury")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.iterfind('.//item'):
            title = item.findtext('title') or ''
            link = item.findtext('link') or ''
            pub = item.findtext('{http://purl.org/dc/elements/1.1/}date') or item.findtext('pubDate') or ''
            items.append({"title": title, "link": link, "pub": pub})
            if len(items) >= limit:
                break
        return items
    except Exception:
        return []

# ---------------------------
# Session state
# ---------------------------
if "picked" not in st.session_state:
    st.session_state.picked = []
if "watch" not in st.session_state:
    st.session_state.watch = set()
if "dnd" not in st.session_state:
    st.session_state.dnd = set()
if "news_player" not in st.session_state:
    st.session_state.news_player = None

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("DraftAid+")
    up = st.file_uploader("Upload FantasyPros CSV", type=["csv"], key="uploader_v3")

    st.subheader("League Setup")
    colA, colB = st.columns(2)
    with colA:
        teams_ct = st.number_input("Teams", 6, 16, 12)
        picks_done = st.number_input("Picks made", 0, 240, 0)
    with colB:
        scoring = st.selectbox("Scoring", ["Standard", "Half-PPR", "PPR"], index=2)
        snake = st.toggle("Snake draft", True)

    st.markdown("---")
    st.subheader("Roster Requirements")
    roster = {}
    for k, v in DEFAULT_ROSTER.items():
        roster[k] = st.number_input(k, 0, 8, v, key=f"rr_{k}")

    st.markdown("---")
    st.subheader("Filters")
    f_pos = st.multiselect("Positions", ["QB","RB","WR","TE","DST","K"], [])
    search = st.text_input("Search player")

# ---------------------------
# Load & prepare board
# ---------------------------
if not up:
    st.info("Upload a FantasyPros CSV to begin.")
    st.stop()

df = load_csv(up).copy()

# Expand positional list (first pos taken as primary display)
primary_pos = df["pos"].apply(lambda x: x[0] if isinstance(x, list) and x else str(x))
df["primary_pos"] = primary_pos

# ADP/ECR delta
if "adp" in df.columns and "ecr" in df.columns:
    df["delta"] = (df["adp"] - df["ecr"]).round(1)
else:
    df["delta"] = np.nan

# Filters
mask = pd.Series(True, index=df.index)
if f_pos:
    mask &= df["primary_pos"].isin(f_pos)
if search:
    mask &= df["player"].str.contains(search, case=False, na=False)

board = df[mask].reset_index(drop=True)

# Bio augmentation (college, experience)
board = augment_with_bio(board)

# ---------------------------
# Roster needs & bye conflicts (lightweight)
# ---------------------------

def current_counts(picked):
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"DST":0,"K":0,"FLEX":0,"Bench":0}
    for p in picked:
        pos = p["pos"]
        if pos in counts:
            counts[pos] += 1
        elif pos in FLEX_ELIGIBLE:
            counts[pos] += 1
    # FLEX/Bench rough calc not critical for your current needs; keep simple
    return counts

counts = current_counts(st.session_state.picked)
needs_cols = st.columns(7)
for i, pos in enumerate(["QB","RB","WR","TE","DST","K","Bench"]):
    have = counts.get(pos,0)
    want = roster.get(pos,0)
    with needs_cols[i]:
        st.metric(pos, f"{have}/{want}")

st.markdown("---")

# ---------------------------
# Board display & interactions
# ---------------------------

def eligibility_badge(pos_list):
    if isinstance(pos_list, list):
        return "/".join(pos_list)
    return str(pos_list)

# Simple sort: Tier asc, ECR asc, then ADP asc
sort_cols = [
    ("tier", True),
    ("ecr", True),
    ("adp", True),
]
for col, asc in reversed(sort_cols):
    if col in board.columns:
        board = board.sort_values(by=col, ascending=asc, na_position='last')

st.subheader("Board")

# Quick actions
a1, a2, a3 = st.columns([2,1,2])
with a1:
    pick_name = st.text_input("Add pick (exact name)")
with a2:
    if st.button("Add Pick") and pick_name:
        m = df[df["player"].str.lower() == pick_name.lower()]
        if not m.empty:
            r = m.iloc[0]
            st.session_state.picked.append({
                "player": r["player"],
                "team": r.get("team",""),
                "pos": r.get("primary_pos",""),
                "bye": int(r["bye"]) if not np.isnan(r.get("bye", np.nan)) else None,
            })
            st.success(f"Added {r['player']} to picks")
        else:
            st.error("Player not found by exact match.")
with a3:
    if st.session_state.news_player:
        st.info(f"News preview loaded for: {st.session_state.news_player}")

# Row renderer

def player_row_html(r):
    name = r["player"]
    team = r.get("team", "") or ""
    pos = eligibility_badge(r.get("pos"))
    bye = int(r["bye"]) if not np.isnan(r.get("bye", np.nan)) else "-"
    tier = r.get("tier", 0)
    delta = r.get("delta", np.nan)
    delta_str = "" if np.isnan(delta) else (f"<span title='ADP - ECR'>{delta:+}</span>")
    college = r.get("college") or "—"
    exp_s = r.get("experience_str") or "—"
    # headshots
    hshots = headshot_urls(name)
    h_html = "".join([f"<img src='{u}' onerror=\"this.style.display='none'\" style='height:38px;width:38px;border-radius:8px;margin-right:8px;object-fit:cover;'/>" for u in hshots])
    # compact news icon
    news_url = make_news_link(name)
    news_html = f"<a href='{news_url}' target='_blank' title='Open news in new tab' style='text-decoration:none;'>📰</a>"
    # one-line bio
    bio_line = f"{exp_s} out of {college}" if college != "—" and exp_s != "—" else (college if college != "—" else exp_s)
    bio_line = bio_line or ""
    return f"""
    <div style='display:flex;align-items:center;gap:8px;'>
      {h_html}
      <div style='flex:1;'>
        <div style='font-weight:600'>{name} <span style='opacity:0.7'>({team} · {pos})</span></div>
        <div style='font-size:12px;opacity:0.8'>Bye {bye} · Tier {tier} · {delta_str} {news_html}</div>
        <div style='font-size:12px;opacity:0.8'>{bio_line}</div>
      </div>
    </div>
    """

# Display top N
for idx, row in board.head(200).iterrows():
    c1, c2, c3 = st.columns([9,1,1])
    with c1:
        st.markdown(player_row_html(row), unsafe_allow_html=True)
    with c2:
        if st.button("Pick", key=f"pick_{idx}"):
            st.session_state.picked.append({
                "player": row["player"],
                "team": row.get("team",""),
                "pos": row.get("primary_pos",""),
                "bye": int(row["bye"]) if not np.isnan(row.get("bye", np.nan)) else None,
            })
            st.rerun()
    with c3:
        if st.button("News", key=f"news_{idx}"):
            st.session_state.news_player = row["player"]
            st.rerun()

st.markdown("---")

# ---------------------------
# Right rail: Picks & News Preview
# ---------------------------
left, right = st.columns([2,2])

with left:
    st.subheader("Your Picks")
    if st.session_state.picked:
        picks_df = pd.DataFrame(st.session_state.picked)
        st.dataframe(picks_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No picks yet")

with right:
    st.subheader("News Preview")
    if st.session_state.news_player:
        items = fetch_news_items(st.session_state.news_player, limit=10)
        if not items:
            st.caption("No recent headlines found.")
        else:
            for it in items:
                st.markdown(f"• [{it['title']}]({it['link']})")
    else:
        st.caption("Click 'News' on any player to preview top headlines here.")

# ---------------------------
# Tips
# ---------------------------
with st.expander("Notes", expanded=False):
    st.markdown(
        """
        - Bench defaults to **5** (adjust in the sidebar if needed).
        - Headshots first try Sleeper's player thumb, then an ESPN-style slug. Some may still not resolve.
        - "Experience" uses Sleeper's `years_exp` when available; college is from Sleeper `college`.
        - Click **News** to preview headlines before drafting; the 📰 icon opens a full Google News search in a new tab.
        """
    )
