import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import hashlib
import urllib.parse

# =========================
# Page & Dark Skin
# =========================
st.set_page_config(page_title="Fantasy Football Draft Aid", layout="wide")

DARK_BG = "#0B0F1A"
CARD_BG = "#12182A"
ACCENT  = "#6EE7F9"   # cyan accent
ACCENT_2= "#A78BFA"   # purple accent
TEXT    = "#E6EAF2"
MUTED   = "#96A2B8"
SUCCESS = "#34D399"
WARN    = "#F59E0B"
DANGER  = "#F43F5E"

def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at 10% 10%, #0c1224, #070a14 60%) !important;
            color:{TEXT};
        }}
        section.main > div.block-container {{ padding-top: 1rem !important; }}
        .skin-card {{
            background: linear-gradient(180deg, {CARD_BG}, #0D1324 80%);
            border: 1px solid rgba(110,231,249,0.08);
            border-radius: 14px;
            padding: 10px 14px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        }}
        .skin-hdr {{ font-weight: 700; letter-spacing: .3px; color: {TEXT}; margin-bottom: .4rem; }}
        .pill {{
            display:inline-block; padding: 2px 8px; border-radius: 999px;
            background: rgba(167,139,250,0.15); color: {ACCENT_2};
            border: 1px solid rgba(167,139,250,0.25); font-size: 12px; margin-left: 6px;
        }}
        .accent {{ color:{ACCENT}; }}
        .success {{ color:{SUCCESS}; }}
        .warn    {{ color:{WARN}; }}
        .danger  {{ color:{DANGER}; }}
        .preview-card {{
            display:flex; gap:14px; align-items:center; border:1px solid rgba(110,231,249,0.12);
            padding:12px; border-radius:12px; background:rgba(255,255,255,0.02);
        }}
        .preview-img {{ width:64px; height:64px; border-radius:50%; object-fit:cover; border:1px solid rgba(255,255,255,0.1); }}
        .btnrow a {{
            text-decoration:none; border:1px solid rgba(150,162,184,0.35); padding:4px 8px; border-radius:8px;
            color:{TEXT}; font-size:12px; margin-right:6px; background:rgba(150,162,184,0.07);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

# =========================
# League / UI constants
# =========================
NUM_TEAMS = 12
ROUNDS = 14
USER_TEAM_ID = 8  # Jimmy

TEAM_NAMES = [
    "Buddy","Giuseppe","Scab","Randino","Nuzzo","Geiger",
    "Bullock","Jimmy (You)","Claire","Dro","Lamart","Logik"
]
TEAM_IDS = {i+1: n for i, n in enumerate(TEAM_NAMES)}

# Preferred build
TARGETS = {'QB':1,'RB':5,'WR':5,'TE':1,'DST':1,'K':1}
POS_ORDER = {'QB':1,'RB':2,'WR':3,'TE':4,'K':5,'DST':6}
POS_MAP = {'DEF':'DST','D/ST':'DST','D':'DST','PK':'K'}

DISPLAY_COLS = ["rank", "name", "team", "position"]

TEAM_COLORS = {
    "ARI":"#97233F","ATL":"#A71930","BAL":"#241773","BUF":"#00338D","CAR":"#0085CA","CHI":"#0B162A",
    "CIN":"#FB4F14","CLE":"#311D00","DAL":"#041E42","DEN":"#FB4F14","DET":"#0076B6","GB":"#203731",
    "HOU":"#03202F","IND":"#002C5F","JAC":"#006778","KC":"#E31837","LAC":"#0080C6",
    "LAR":"#003594","LV":"#000000","MIA":"#008E97","MIN":"#4F2683","NE":"#002244","NO":"#D3BC8D",
    "NYG":"#0B2265","NYJ":"#125740","PHI":"#004C54","PIT":"#FFB612","SF":"#AA0000","SEA":"#002244",
    "TB":"#D50A0A","TEN":"#0C2340","WAS":"#5A1414"
}
TEAM_LOGOS = {abbr: f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png" for abbr in TEAM_COLORS}

# Baseline ESPN IDs for popular players (extend via meta upload)
BASE_ESPN_ID = {
    "Christian McCaffrey": 3117251, "Saquon Barkley": 3929630, "Josh Allen": 3918298,
    "Lamar Jackson": 3916387, "Patrick Mahomes": 3139477, "Joe Burrow": 3915511,
    "Ja'Marr Chase": 4362628, "Justin Jefferson": 4047365, "CeeDee Lamb": 4241389,
    "Amon-Ra St. Brown": 4361429, "Puka Nacua": 4241447, "Jahmyr Gibbs": 4426359,
    "Bijan Robinson": 4685729, "Brock Bowers": 4685715, "Jayden Daniels": 4430784,
    "Jalen Hurts": 4040715, "Tee Higgins": 4242335, "Tyreek Hill": 3116406,
    "A.J. Brown": 4047646, "Jared Goff": 2976212, "Justin Herbert": 4038941, "Kyler Murray": 3917315
}

# =========================
# Utilities
# =========================
def log_event(*args):
    if "logs" not in st.session_state: st.session_state.logs = []
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.logs.append(f"[{stamp}] " + " ".join(str(a) for a in args))

def snake_team_for_pick(pick_no: int) -> int:
    rnd = (pick_no - 1) // NUM_TEAMS + 1
    return ((pick_no - 1) % NUM_TEAMS + 1) if rnd % 2 == 1 else NUM_TEAMS - ((pick_no - 1) % NUM_TEAMS)

def pick_to_round_slot(pick_no: int) -> tuple[int,int]:
    r = (pick_no - 1) // NUM_TEAMS + 1
    slot = ((pick_no - 1) % NUM_TEAMS) + 1
    return r, slot

def round_slot_str(pick_no: int) -> str:
    r, s = pick_to_round_slot(pick_no)
    return f"{r}.{s:02d}"

def normalize_pos(x: str) -> str:
    if pd.isna(x): return 'UNK'
    s = re.sub(r'\d+', '', str(x).upper()).strip()
    return POS_MAP.get(s, s)

def normalize_team(x: str) -> str:
    if pd.isna(x): return 'UNK'
    s = str(x).upper().strip()
    s = (s.replace('JAX','JAC').replace('SFO','SF').replace('SF 49ERS','SF')
           .replace('KANSAS CITY','KC').replace('N.Y. JETS','NYJ').replace('N.Y. GIANTS','NYG'))
    s = re.sub(r'[^A-Z]', '', s)
    return s if s else 'UNK'

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "RK":"rank", "Rk":"rank", "Rank":"rank",
        "PLAYER NAME":"name","Player":"name","PLAYER":"name","Player Name":"name",
        "TEAM":"team","Tm":"team","Team":"team",
        "POS":"position","Pos":"position","Position":"position"
    }
    for src, dst in rename_map.items():
        if src in df.columns: df = df.rename(columns={src: dst})

    lower = {c.lower(): c for c in df.columns}
    def pick(col_names):
        for c in col_names:
            if c in df.columns: return c
            if c.lower() in lower: return lower[c.lower()]
        return None

    rank_col = pick(["rank","rk","#","overall","ovr"])
    name_col = pick(["name","player","player name","playername","player/team","player (team)"])
    team_col = pick(["team","tm"])
    pos_col  = pick(["position","pos"])

    if name_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == 'O']
        name_col = obj_cols[0] if obj_cols else df.columns[0]

    if rank_col is None:
        rank_col = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[rank_col]):
            df["rank"] = np.arange(1, len(df) + 1)
            rank_col = "rank"

    if team_col is None:
        df["team"] = "UNK"; team_col = "team"
    if pos_col is None:
        df["position"] = "UNK"; pos_col = "position"

    out = pd.DataFrame({
        "rank": pd.to_numeric(df[rank_col], errors="coerce").fillna(9999),
        "name": df[name_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "team": df[team_col].astype(str).map(normalize_team),
        "position": df[pos_col].astype(str).map(normalize_pos),
    })
    out = out.dropna(subset=["name"]).reset_index(drop=True)

    # Parse embedded "Name POS, TEAM"
    mask_emb = (out["position"].eq("UNK")) & out["name"].str.contains(r"\b(QB|RB|WR|TE|K|DST)\b", regex=True)
    if mask_emb.any():
        parts = out.loc[mask_emb, "name"].str.extract(r"^(?P<n>[A-Za-z\.\'\-\s]+)\s+(?P<p>[A-Z]{1,3})(?:,\s*(?P<t>[A-Z]+))?")
        out.loc[mask_emb, "name"] = parts["n"].fillna(out.loc[mask_emb, "name"])
        out.loc[mask_emb, "position"] = parts["p"].map(normalize_pos).fillna(out.loc[mask_emb, "position"])
        out.loc[mask_emb, "team"] = parts["t"].map(normalize_team).fillna(out.loc[mask_emb, "team"])

    out = out.sort_values("rank").drop_duplicates(subset=["name","team","position"], keep="first").reset_index(drop=True)
    out.insert(0, "pid", range(1, len(out) + 1))
    return out

def team_counts(team_df: pd.DataFrame) -> dict:
    return team_df["position"].value_counts().to_dict() if not team_df.empty else {}

def remaining_needs(team_df: pd.DataFrame, targets: dict) -> dict:
    counts = team_counts(team_df)
    return {p: max(0, t - counts.get(p, 0)) for p, t in targets.items()}

def need_weight(pos: str) -> float:
    return {'RB':9.0,'WR':9.0,'QB':5.5,'TE':5.0,'DST':2.0,'K':1.5}.get(pos,1.0)

def suggest_best(available: pd.DataFrame, team_df: pd.DataFrame, targets: dict):
    if available.empty: return None
    needs = remaining_needs(team_df, targets)
    w = available.copy()
    r = w["rank"].astype(float)
    mu, sd = r.mean(), r.std(ddof=0)
    w["score_rank"] = -r if (sd == 0 or not np.isfinite(mu) or not np.isfinite(sd)) else (mu - r) / (sd + 1e-6)
    w["need_bonus"] = w["position"].map(lambda p: need_weight(p) * needs.get(p,0))
    w["zero_bonus"] = w["position"].map(lambda p: 0.25 if (needs.get(p,0)>0 and (team_df["position"].eq(p).sum() if not team_df.empty else 0)==0) else 0.0)
    w["score"] = w["score_rank"] + w["need_bonus"] + w["zero_bonus"]
    row = w.sort_values(["score","score_rank"], ascending=False).head(1)
    return None if row.empty else row.iloc[0]

# ---------- Headshots + Meta ----------
def initials_avatar(name: str, hex_bg: str) -> str:
    if not isinstance(name, str) or name.strip() == "": name = "?"
    bg = hex_bg.lstrip("#")
    return f"https://ui-avatars.com/api/?name={urllib.parse.quote(name)}&background={bg}&color=ffffff&size=128&bold=true&format=png"

def espn_headshot_from_id(espn_id: int | None) -> str | None:
    if not espn_id: return None
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png"

def player_image_url(name: str, team_abbr: str, use_headshots: bool) -> str:
    abbr = str(team_abbr).upper().strip()
    if use_headshots:
        # priority: meta override -> base map -> initials
        eid = st.session_state.player_meta_map.get(name) if "player_meta_map" in st.session_state else None
        if not eid: eid = st.session_state.base_espn_map.get(name)
        url = espn_headshot_from_id(eid)
        if url: return url
        color = TEAM_COLORS.get(abbr, "#222222")
        return initials_avatar(name, color)
    return TEAM_LOGOS.get(abbr, "")

def parse_player_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept flexible headers
    name_col = None
    for c in df.columns:
        if c in ("name","player","player name","player_name"):
            name_col = c; break
    if name_col is None: raise ValueError("player_meta.csv needs a 'name' column")
    df = df.rename(columns={name_col:"name"})
    if "espn_id" in df.columns:
        df["espn_id"] = pd.to_numeric(df["espn_id"], errors="coerce").astype("Int64")
    if "college" not in df.columns: df["college"] = ""
    if "exp" not in df.columns and "experience" in df.columns:
        df = df.rename(columns={"experience":"exp"})
    if "exp" not in df.columns: df["exp"] = ""
    df["name"] = df["name"].astype(str).str.strip()
    return df[["name","espn_id","college","exp"]]

# =========================
# Draft actions
# =========================
def draft_pid(pid: int):
    next_pick = st.session_state.pick_counter + 1
    on_clock = snake_team_for_pick(next_pick)
    if pid is None: return False, "No player selected."
    pool = st.session_state.available
    mask = pool["pid"].eq(pid); idx = pool.index[mask]
    if len(idx) == 0: return False, "Player already drafted or not found."
    row = pool.loc[idx[0]]
    if len(st.session_state.teams[on_clock]) >= ROUNDS:
        return False, f"Team {TEAM_IDS[on_clock]} roster full."
    row = row.copy()
    row["pick_no"] = next_pick
    row["pick_str"] = round_slot_str(next_pick)
    abbr = str(row.get("team", "")).upper().strip()
    row["logo"]  = TEAM_LOGOS.get(abbr, "")
    row["color"] = TEAM_COLORS.get(abbr, "#222222")
    row["img"]   = player_image_url(row["name"], abbr, st.session_state.get("show_headshots", False))
    # attach meta (best-effort)
    meta = st.session_state.player_meta_df
    if not meta.empty:
        mrow = meta[meta["name"].str.lower()==row["name"].lower()].head(1)
        if not mrow.empty:
            row["college"] = mrow.iloc[0]["college"]
            row["exp"] = mrow.iloc[0]["exp"]
    st.session_state.teams[on_clock] = pd.concat([st.session_state.teams[on_clock], row.to_frame().T], ignore_index=True)
    st.session_state.available = pool.drop(index=idx).reset_index(drop=True)
    st.session_state.pick_counter = next_pick
    st.session_state.draft_log.append({"pick": next_pick,"team": TEAM_IDS[on_clock],"name": row["name"],"pos": row["position"],"pick_str": row["pick_str"]})
    return True, f"Pick {next_pick} ({row['pick_str']}): {row['name']} → {TEAM_IDS[on_clock]}"

def run_one_cpu_pick():
    total = NUM_TEAMS * ROUNDS
    if st.session_state.pick_counter >= total: return False, "Draft complete."
    next_pick = st.session_state.pick_counter + 1
    team_id = snake_team_for_pick(next_pick)
    if team_id == USER_TEAM_ID: return False, "Your pick."
    ai_team = st.session_state.teams[team_id]
    best = suggest_best(st.session_state.available, ai_team, TARGETS)
    if best is None:
        if st.session_state.available.empty: return False, "No players."
        pid = int(st.session_state.available.sort_values("rank").iloc[0]["pid"])
    else:
        pid = int(best["pid"])
    return draft_pid(pid)

def sim_to_me():
    moved = 0; total = NUM_TEAMS * ROUNDS
    while st.session_state.pick_counter < total:
        next_pick = st.session_state.pick_counter + 1
        if snake_team_for_pick(next_pick) == USER_TEAM_ID: break
        ok, _ = run_one_cpu_pick()
        if not ok: break
        moved += 1
    return moved

def filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q: return df
    q = q.strip().lower()
    m = (df["name"].str.lower().str.contains(q, na=False) |
         df["team"].str.lower().str.contains(q, na=False) |
         df["position"].str.lower().str.contains(q, na=False))
    return df[m]

# =========================
# Session state defaults
# =========================
if "available" not in st.session_state:
    st.session_state.available = pd.DataFrame(columns=["pid","rank","name","team","position"])
if "teams" not in st.session_state:
    base_cols = ["pid","rank","name","team","position","pick_no","pick_str","logo","color","img","college","exp"]
    st.session_state.teams = {i: pd.DataFrame(columns=base_cols) for i in range(1, NUM_TEAMS+1)}
if "draft_log" not in st.session_state: st.session_state.draft_log = []
if "pick_counter" not in st.session_state: st.session_state.pick_counter = 0
if "selected_pid" not in st.session_state: st.session_state.selected_pid = None
if "loaded_file_hash" not in st.session_state: st.session_state.loaded_file_hash = None
if "fancy_mode" not in st.session_state: st.session_state.fancy_mode = True
if "show_headshots" not in st.session_state: st.session_state.show_headshots = True  # try heads again
if "player_meta_df" not in st.session_state: st.session_state.player_meta_df = pd.DataFrame(columns=["name","espn_id","college","exp"])
if "player_meta_map" not in st.session_state: st.session_state.player_meta_map = {}
# base headshot map
st.session_state.base_espn_map = BASE_ESPN_ID.copy()

# =========================
# Sidebar: load + controls
# =========================
st.sidebar.header("Load Rankings")
up = st.sidebar.file_uploader("Upload FantasyPros CSV", type=["csv"], key="uploader_main")

if st.sidebar.button("Load / Reset from CSV", use_container_width=True, key="btn_load"):
    if up is None:
        st.sidebar.error("Please upload a CSV first.")
    else:
        try:
            data = up.getvalue()
            file_hash = hashlib.md5(data).hexdigest()
            if st.session_state.loaded_file_hash == file_hash:
                st.sidebar.info("Same file already loaded.")
            else:
                raw = pd.read_csv(up)
                pool = ensure_cols(raw).sort_values("rank").reset_index(drop=True)
                st.session_state.available = pool
                base_cols = ["pid","rank","name","team","position","pick_no","pick_str","logo","color","img","college","exp"]
                st.session_state.teams = {i: pd.DataFrame(columns=base_cols) for i in range(1, NUM_TEAMS+1)}
                st.session_state.draft_log = []
                st.session_state.pick_counter = 0
                st.session_state.selected_pid = None
                st.session_state.loaded_file_hash = file_hash
                st.sidebar.success(f"Loaded {len(pool)} players.")
        except Exception as e:
            st.sidebar.error(f"Read error: {e}")

st.sidebar.header("Player Meta (optional)")
meta_up = st.sidebar.file_uploader("Upload player_meta.csv (name, espn_id, college, exp)", type=["csv"], key="uploader_meta")
if st.sidebar.button("Load Meta", use_container_width=True, key="btn_load_meta"):
    if meta_up is None:
        st.sidebar.error("Upload player_meta.csv first.")
    else:
        try:
            meta_raw = pd.read_csv(meta_up)
            meta_df = parse_player_meta(meta_raw)
            st.session_state.player_meta_df = meta_df
            # build map name -> espn_id
            st.session_state.player_meta_map = {n: (int(e) if pd.notna(e) else None) for n, e in zip(meta_df["name"], meta_df["espn_id"])}
            st.sidebar.success(f"Meta loaded for {len(meta_df)} players.")
        except Exception as e:
            st.sidebar.error(f"Meta parse error: {e}")

st.sidebar.header("Draft Controls")
c1, c2 = st.sidebar.columns(2)
if c1.button("Sim one pick", use_container_width=True, key="btn_sim_one"):
    ok, msg = run_one_cpu_pick(); st.sidebar.info(msg); st.rerun()
if c2.button("Sim to my turn", use_container_width=True, key="btn_sim_me"):
    moved = sim_to_me(); st.sidebar.info(f"Advanced {moved} picks."); st.rerun()
if st.sidebar.button("Clear Draft", type="secondary", use_container_width=True, key="btn_clear"):
    base_cols = ["pid","rank","name","team","position","pick_no","pick_str","logo","color","img","college","exp"]
    st.session_state.teams = {i: pd.DataFrame(columns=base_cols) for i in range(1, NUM_TEAMS+1)}
    st.session_state.draft_log = []; st.session_state.pick_counter = 0; st.session_state.selected_pid = None
    st.sidebar.success("Cleared."); st.rerun()

st.sidebar.header("Appearance")
st.session_state.fancy_mode = st.sidebar.toggle("Fancy mode (logos / color accents)", value=st.session_state.fancy_mode, key="toggle_fancy")
st.session_state.show_headshots = st.sidebar.toggle("Show headshots (ESPN/initials)", value=st.session_state.show_headshots, key="toggle_heads")
show_debug = st.sidebar.checkbox("Show debug panel", value=False, key="toggle_debug")

# =========================
# Layout
# =========================
left, right = st.columns([0.62, 0.38], gap="large")

# ===== LEFT: Board & Tables =====
with left:
    st.markdown(f"""<div class="skin-card"><div class="skin-hdr">Best Available Players <span class="pill">Live</span></div></div>""", unsafe_allow_html=True)
    next_pick = st.session_state.pick_counter + 1
    on_clock_id = snake_team_for_pick(next_pick)
    st.caption(f"On the clock: **{TEAM_IDS[on_clock_id]}** • Pick {next_pick} ({round_slot_str(next_pick)})")

    suggested = None
    if on_clock_id == USER_TEAM_ID and not st.session_state.available.empty:
        suggested = suggest_best(st.session_state.available, st.session_state.teams[USER_TEAM_ID], TARGETS)

    tabs = st.tabs(["Overall","QB","RB","WR","TE","K","DST"])

    def table_with_select(df, key_prefix, star_pid=None):
        if df.empty:
            st.info("No players."); return
        df_show = df.copy()
        if star_pid is not None and "pid" in df_show.columns:
            df_show.insert(0, "⭐", np.where(df_show["pid"].eq(star_pid), "⭐", ""))
        if st.session_state.fancy_mode:
            df_show["img"] = df_show.apply(lambda r: player_image_url(r.get("name",""), r.get("team",""), st.session_state.show_headshots), axis=1)
            cols_to_show = (["⭐","img"] if "⭐" in df_show.columns else ["img"])
        else:
            cols_to_show = (["⭐"] if "⭐" in df_show.columns else [])
        cols_to_show += [c for c in DISPLAY_COLS if c in df_show.columns and c not in cols_to_show]
        if st.session_state.fancy_mode and "img" in df_show.columns:
            st.dataframe(df_show[cols_to_show], use_container_width=True, hide_index=True,
                         column_config={"img": st.column_config.ImageColumn(" ", help="Player/Team")})
        else:
            st.dataframe(df_show[cols_to_show], use_container_width=True, hide_index=True)

        # Dropdown to set selected_pid
        labels = ["— Select —"]; pid_by_label = {}
        for _, row in df_show.iterrows():
            if "pid" not in df_show.columns: continue
            pid = int(row["pid"]); rnk = row.get("rank","?")
            try: rnk = int(rnk)
            except: rnk = rnk if rnk != "" else "?"
            name = row.get("name","?"); team = (row.get("team","") or "").strip(); pos = (row.get("position","") or "").strip()
            tail = " ".join([x for x in [team, pos] if x]); label = f"{rnk}. {name}" + (f" ({tail})" if tail else "")
            labels.append(label); pid_by_label[label] = pid
        choice = st.selectbox("Select (ready for right-panel Draft)", labels, key=f"{key_prefix}_selectbox")
        if choice != "— Select —":
            pid = pid_by_label.get(choice)
            if pid is not None:
                st.session_state.selected_pid = pid
                st.success(f"Selected: {df_show.loc[df_show['pid'].eq(pid), 'name'].iloc[0]}")

    with tabs[0]:
        q = st.text_input("Search (overall)", value="", key="q_all")
        view = filter_df(st.session_state.available, q).sort_values("rank").head(150).reset_index(drop=True)
        star_pid = int(suggested["pid"]) if suggested is not None else None
        table_with_select(view, "all", star_pid=star_pid)

    def render_pos_tab(pos, idx):
        with tabs[idx]:
            q = st.text_input(f"Search ({pos})", value="", key=f"q_{pos}")
            sub = st.session_state.available.query("position == @pos")
            view = filter_df(sub, q).sort_values("rank").head(150).reset_index(drop=True)
            star_pid = int(suggested["pid"]) if suggested is not None and suggested["position"] == pos else None
            table_with_select(view, f"{pos}", star_pid=star_pid)

    render_pos_tab("QB", 1); render_pos_tab("RB", 2); render_pos_tab("WR", 3)
    render_pos_tab("TE", 4); render_pos_tab("K", 5); render_pos_tab("DST", 6)

# ===== RIGHT: Pick & Teams =====
with right:
    st.markdown(f"""<div class="skin-card"><div class="skin-hdr">On the Clock</div></div>""", unsafe_allow_html=True)
    st.info(f"**Pick {st.session_state.pick_counter + 1}** ({round_slot_str(st.session_state.pick_counter + 1)}) — {TEAM_IDS[snake_team_for_pick(st.session_state.pick_counter + 1)]}")

    pool = st.session_state.available
    if pool.empty:
        st.warning("No players left."); pid_choice = None
    else:
        label_to_pid = {f"{int(r.rank)}. {r.name} ({r.team} {r.position})": int(r.pid) for r in pool.itertuples()}
        labels = ["— Select —"] + list(label_to_pid.keys())
        if st.session_state.selected_pid in pool["pid"].values:
            current_label = next((lbl for lbl, pid in label_to_pid.items() if pid == st.session_state.selected_pid), None)
            init_idx = labels.index(current_label) if current_label in labels else 0
        else:
            init_idx = 0
        chosen_label = st.selectbox("Select player to DRAFT", labels, index=init_idx, key="global_draft_box")
        pid_choice = None if chosen_label == "— Select —" else label_to_pid[chosen_label]
        if pid_choice is not None: st.session_state.selected_pid = pid_choice

    # ===== Pre-draft Preview (headshot + college + exp + quick news) =====
    sel = None
    if st.session_state.selected_pid and not pool.empty:
        match = pool[pool["pid"].eq(st.session_state.selected_pid)]
        if not match.empty:
            sel = match.iloc[0]
    if sel is not None:
        name = sel["name"]; team = sel["team"]; pos = sel["position"]; rnk = int(sel["rank"])
        img = player_image_url(name, team, st.session_state.show_headshots)
        # meta lookup
        college = ""; exp = ""
        meta = st.session_state.player_meta_df
        if not meta.empty:
            mr = meta[meta["name"].str.lower()==str(name).lower()].head(1)
            if not mr.empty:
                college = str(mr.iloc[0]["college"] or "")
                exp = str(mr.iloc[0]["exp"] or "")
        # compact preview card
        st.markdown(
            f"""
            <div class="preview-card">
                <img src="{img}" class="preview-img" alt="headshot"/>
                <div>
                    <div style="font-weight:700;font-size:16px">{name} <span style="opacity:.75">({team} {pos})</span></div>
                    <div style="opacity:.8">Rank #{rnk}{' • ' + college if college else ''}{' • ' + exp + ' exp' if exp else ''}</div>
                    <div class="btnrow" style="margin-top:6px;">
                        <a href="https://news.google.com/search?q={urllib.parse.quote(name + ' fantasy')}" target="_blank">📰 News</a>
                        <a href="https://www.nbcsports.com/fantasy/football/player/{urllib.parse.quote(name.replace(' ', '-').lower())}" target="_blank">📚 Edge</a>
                        <a href="https://www.rotowire.com/football/search.php?search={urllib.parse.quote(name)}" target="_blank">🧠 RotoWire</a>
                        <a href="https://x.com/search?q={urllib.parse.quote(name + ' fantasy')}" target="_blank">𝕏 Search</a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Suggestion (your turn)
    on_clock_now = snake_team_for_pick(st.session_state.pick_counter + 1)
    if on_clock_now == USER_TEAM_ID and not st.session_state.available.empty:
        sug = suggest_best(st.session_state.available, st.session_state.teams[USER_TEAM_ID], TARGETS)
        if sug is not None: st.info(f"Suggested: **{sug['name']}** ({sug['team']} {sug['position']})")

    if st.button("🚀 Draft Selected", type="primary", use_container_width=True, key="btn_draft"):
        pid_to_use = st.session_state.selected_pid if pid_choice is None else pid_choice
        ok, msg = draft_pid(pid_to_use)
        if ok:
            st.success(msg); st.session_state.selected_pid = None; st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.markdown(f"""<div class="skin-card"><div class="skin-hdr">Draft Log</div></div>""", unsafe_allow_html=True)
    if st.session_state.draft_log:
        log_df = pd.DataFrame(st.session_state.draft_log)
        log_df["pick_str"] = log_df["pick"].apply(lambda p: round_slot_str(int(p)))
        st.dataframe(log_df.sort_values("pick"), use_container_width=True, hide_index=True)
    else:
        st.write("No picks yet.")

    st.markdown(f"""<div class="skin-card"><div class="skin-hdr">Team Viewer</div></div>""", unsafe_allow_html=True)
    viewer = st.selectbox("View team:", [f"{i}. {TEAM_IDS[i]}" for i in range(1, NUM_TEAMS+1)], index=USER_TEAM_ID-1, key="viewer_team")
    view_tid = int(viewer.split(".")[0])
    tdf = st.session_state.teams[view_tid]

    if tdf.empty:
        st.write("No players yet.")
    else:
        tmp = tdf.copy()
        tmp["pos_order"] = tmp["position"].map(lambda p: POS_ORDER.get(p, 9))
        tmp = tmp.sort_values(["pick_no","pos_order","rank"]).drop(columns=["pos_order"])
        if "pick_no" in tmp.columns:
            tmp["pick_str"] = tmp["pick_no"].apply(lambda p: round_slot_str(int(p)) if pd.notna(p) else "")
        if "img" not in tmp.columns or (tmp["img"] == "").all():
            tmp["img"] = tmp.apply(lambda r: player_image_url(r.get("name",""), r.get("team",""), st.session_state.show_headshots), axis=1)

        if st.session_state.fancy_mode:
            show_cols = ["pick_str","img","name","position","rank"]
            # show college/exp if present
            if "college" in tmp.columns or "exp" in tmp.columns:
                if "college" not in tmp.columns: tmp["college"] = ""
                if "exp" not in tmp.columns: tmp["exp"] = ""
                show_cols += ["college","exp"]
            tmp["news"] = tmp["name"].map(lambda n: f"https://news.google.com/search?q={urllib.parse.quote(n + ' fantasy')}")
            show_cols.append("news")
            st.dataframe(
                tmp[show_cols], use_container_width=True, hide_index=True,
                column_config={
                    "pick_str": "Pick",
                    "img": st.column_config.ImageColumn(" ", help="Player/Team"),
                    "news": st.column_config.LinkColumn("📰 News", help="Latest news")
                }
            )
        else:
            show_cols = ["pick_str","name","position","team","rank"]
            if "college" in tmp.columns or "exp" in tmp.columns:
                if "college" not in tmp.columns: tmp["college"] = ""
                if "exp" not in tmp.columns: tmp["exp"] = ""
                show_cols += ["college","exp"]
            st.dataframe(tmp[show_cols], use_container_width=True, hide_index=True)

    # Position counts & needs
    st.caption("Position counts & needs")
    order = ['QB','RB','WR','TE','DST','K']
    counts_dict = team_counts(tdf) if not tdf.empty else {}
    needs_dict  = remaining_needs(tdf if not tdf.empty else pd.DataFrame(columns=["position"]), TARGETS)
    counts_df = pd.DataFrame([[int(counts_dict.get(c, 0)) for c in order]], columns=order)
    needs_df  = pd.DataFrame([[int(needs_dict.get(c, 0)) for c in order]], columns=order)
    st.write("Current:"); st.dataframe(counts_df, use_container_width=True, hide_index=True)
    st.write("To target:"); st.dataframe(needs_df, use_container_width=True, hide_index=True)

# =========================
# Debug
# =========================
if show_debug:
    if "logs" in st.session_state and st.session_state.logs:
        st.code("\n".join(st.session_state.logs), language="text")
    else:
        st.write("No logs.")
