"""DraftAid -- live fantasy football draft assistant.

12-team 1QB full-PPR by default. Blends FantasyPros expert consensus, live
real-draft ADP, Sleeper injury/trending data, and an optional Underdog CSV
into one tier-colored board with pick suggestions and snake-draft tracking.
"""

import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

import draft_logic as dl
from data_sources import build_board, fetch_player_news, parse_rankings_csv

st.set_page_config(page_title="DraftAid", page_icon="🏈", layout="wide")

AUTOSAVE = Path(__file__).parent / "draft_autosave.json"
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
DEFAULT_TEAMS = ["Buddy", "Giuseppe", "Scab", "Randino", "Nuzzo", "Geiger",
                 "Bullock", "Jimmy (You)", "Claire", "Dro", "Lamart", "Logik"]

# ---------------------------------------------------------------- state

if "picks" not in st.session_state:
    st.session_state.picks = []          # [{key, player, pos, team, bye, mine, overall}]
if "phantom" not in st.session_state:
    st.session_state.phantom = 0         # unlogged picks (counter resync)
if "csv_ranks" not in st.session_state:
    st.session_state.csv_ranks = None
if "csv_name" not in st.session_state:
    st.session_state.csv_name = ""
if "board_nonce" not in st.session_state:
    st.session_state.board_nonce = 0     # bumping it resets the board table selection
if "league_nonce" not in st.session_state:
    st.session_state.league_nonce = 0    # bumping it re-seeds the league editor


def state_payload() -> dict:
    return {
        "picks": st.session_state.picks,
        "phantom": st.session_state.phantom,
        "league": st.session_state.get("_league", {}),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def autosave():
    try:
        AUTOSAVE.write_text(json.dumps(state_payload()))
    except OSError:
        pass


def load_state(payload: dict):
    st.session_state.picks = payload.get("picks", [])
    st.session_state.phantom = payload.get("phantom", 0)
    league = payload.get("league") or {}
    if league.get("team_names"):
        st.session_state.league_seed = league["team_names"]
        if league.get("slot"):
            st.session_state.me_seed = league["team_names"][league["slot"] - 1]
        st.session_state.league_nonce += 1
    st.session_state.board_nonce += 1


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🏈 DraftAid")

    with st.expander("League setup", expanded=False):
        tc1, tc2 = st.columns(2)
        teams = tc1.number_input("Teams", 4, 20, 12)
        rounds = tc2.number_input("Rounds", 8, 25, 14)

        st.caption("Edit names for this year's league. To change the draft order, "
                   "renumber the **Pick** column — the board sorts by it.")
        seed = list(st.session_state.get("league_seed") or DEFAULT_TEAMS)
        seed = (seed + [f"Team {i}" for i in range(len(seed) + 1, int(teams) + 1)])[:int(teams)]
        edited = st.data_editor(
            pd.DataFrame({"Pick": range(1, int(teams) + 1), "Team": seed}),
            key=f"league_editor_{int(teams)}_{st.session_state.league_nonce}",
            hide_index=True, width="stretch", num_rows="fixed",
            column_config={
                "Pick": st.column_config.NumberColumn("Pick", min_value=1,
                                                      max_value=int(teams), step=1),
                "Team": st.column_config.TextColumn("Team", required=True),
            })
        ordered = edited.sort_values("Pick", kind="stable")["Team"].tolist()
        team_names = [(str(n).strip() or f"Team {i + 1}") for i, n in enumerate(ordered)]

        me_default = st.session_state.get("me_seed") or next(
            (n for n in team_names if "(you)" in n.lower()), team_names[0])
        me = st.selectbox("Which team is you?", team_names,
                          index=team_names.index(me_default) if me_default in team_names else 0,
                          key=f"me_team_{int(teams)}_{st.session_state.league_nonce}")
        slot = team_names.index(me) + 1
        st.caption(f"You pick {slot}{'st' if slot == 1 else 'nd' if slot == 2 else 'rd' if slot == 3 else 'th'} "
                   f"in odd rounds.")

        st.markdown("**Starting lineup**")
        c1, c2, c3 = st.columns(3)
        spots = {
            "QB": c1.number_input("QB", 0, 3, 1),
            "RB": c2.number_input("RB", 0, 5, 2),
            "WR": c3.number_input("WR", 0, 5, 2),
            "TE": c1.number_input("TE", 0, 3, 1),
            "FLEX": c2.number_input("FLEX", 0, 4, 1),
            "K": c3.number_input("K", 0, 2, 1),
            "DST": c1.number_input("DST", 0, 2, 1),
        }
    st.session_state._league = {"team_names": team_names, "slot": int(slot)}

    with st.expander("Ranking sources & weights", expanded=False):
        st.caption("How much each source counts toward the blended rank.")
        w_ecr = st.slider("FantasyPros expert consensus", 0, 100, 50)
        w_adp = st.slider("Live ADP (FFCalculator)", 0, 100, 30)
        w_csv = st.slider("Uploaded CSV (Underdog)", 0, 100, 20)
        up = st.file_uploader("Underdog / rankings CSV", type=["csv"],
                              help="On Underdog: Rankings → Download CSV. Any CSV "
                                   "with a player-name and rank/ADP column works.")
        if up is not None and up.name != st.session_state.csv_name:
            try:
                st.session_state.csv_ranks = parse_rankings_csv(up)
                st.session_state.csv_name = up.name
                st.success(f"Loaded {len(st.session_state.csv_ranks)} players from {up.name}")
            except ValueError as e:
                st.error(str(e))

    if st.button("🔄 Refresh live rankings", width="stretch",
                 help="Re-pulls FantasyPros, ADP, and Sleeper data right now."):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.download_button(
        "💾 Export draft state",
        json.dumps(state_payload()),
        file_name="draft_state.json", mime="application/json",
        width="stretch",
    )
    restore = st.file_uploader("Import draft state", type=["json"], key="import_state")
    if restore is not None and st.button("Load imported state", width="stretch"):
        load_state(json.load(restore))
        autosave()
        st.rerun()

    with st.popover("🗑 Reset draft", width="stretch"):
        st.write("This clears every logged pick.")
        if st.button("Yes, reset everything", type="primary"):
            st.session_state.picks = []
            st.session_state.phantom = 0
            st.session_state.board_nonce += 1
            AUTOSAVE.unlink(missing_ok=True)
            st.rerun()

# Offer to resume an autosaved draft after a crash/refresh.
if not st.session_state.picks and AUTOSAVE.exists():
    payload = json.loads(AUTOSAVE.read_text())
    if payload.get("picks"):
        st.info(f"Found an autosaved draft from {payload.get('saved_at', '?')} "
                f"with {len(payload['picks'])} picks.")
        cA, cB = st.columns([1, 1])
        if cA.button("▶ Resume saved draft", type="primary"):
            load_state(payload)
            st.rerun()
        if cB.button("Discard it"):
            AUTOSAVE.unlink(missing_ok=True)
            st.rerun()

# ---------------------------------------------------------------- board data

total = w_ecr + w_adp + w_csv
weights = ({"ecr": w_ecr / total, "adp": w_adp / total, "csv": w_csv / total}
           if total else {"ecr": 1.0, "adp": 0.0, "csv": 0.0})

try:
    board = build_board(st.session_state.csv_ranks, weights)
except Exception as e:
    st.error(f"Could not load rankings: {e}")
    st.stop()

picks = st.session_state.picks
drafted_keys = {p["key"] for p in picks}
taken_by_key = {p["key"]: ("Me" if p["mine"] else "Taken") for p in picks}
my_players = pd.DataFrame([p for p in picks if p["mine"]],
                          columns=["key", "player", "pos", "team", "bye", "mine"])

# Who took each drafted player, e.g. "Randino 3.02".
status_by_key = {}
for i, p in enumerate(picks):
    o = p.get("overall", i + 1)
    r_, k_ = dl.round_and_pick(o, int(teams))
    status_by_key[p["key"]] = (
        f"{team_names[dl.snake_team_for_pick(o, int(teams)) - 1]} {r_}.{k_:02d}")

current_overall = len(picks) + st.session_state.phantom + 1
rnd, pick_in_rnd = dl.round_and_pick(current_overall, int(teams))
upcoming = dl.next_my_picks(current_overall, int(slot), int(teams), int(rounds))
on_clock = bool(upcoming) and upcoming[0] == current_overall
next_turn = (upcoming[1] if on_clock else upcoming[0]) if len(upcoming) > (1 if on_clock else 0) else None

if next_turn is not None:
    board["lasts"] = [dl.survival_probability(a, s, next_turn)
                      for a, s in zip(board["adp"], board["adp_std"])]
else:
    board["lasts"] = float("nan")
available = board[~board["key"].isin(drafted_keys)]

# --------------------------------------------------------- pick callbacks


def _apply_pick(row: pd.Series, mine: bool):
    st.session_state.picks.append({
        "key": row["key"], "player": row["player"], "pos": row["pos"],
        "team": row["team"], "bye": int(row["bye"]), "mine": mine,
        "overall": len(st.session_state.picks) + st.session_state.phantom + 1,
    })
    st.session_state.board_nonce += 1
    autosave()


def record_quick(mine: bool):
    label = st.session_state.get("quick_pick")
    options = st.session_state.get("_quick_map", {})
    if label and label in options:
        _apply_pick(options[label], mine)
        st.session_state.quick_pick = None


def request_card():
    """Open the player card for whoever is typed in the quick-entry box."""
    label = st.session_state.get("quick_pick")
    options = st.session_state.get("_quick_map", {})
    if label and label in options:
        st.session_state.card_request = options[label]["key"]


def undo_pick():
    if st.session_state.picks:
        st.session_state.picks.pop()
        st.session_state.board_nonce += 1
        autosave()


def remove_pick_by_key(key: str):
    idx = next((i for i, p in enumerate(st.session_state.picks) if p["key"] == key), None)
    if idx is None:
        return
    st.session_state.picks.pop(idx)
    for later in st.session_state.picks[idx:]:
        if "overall" in later:
            later["overall"] -= 1
    st.session_state.board_nonce += 1
    autosave()


def _img_url(r) -> str | None:
    if r["pos"] == "DST":
        return f"https://sleepercdn.com/images/team_logos/nfl/{str(r['team']).lower()}.png"
    sid = r.get("sleeper_id")
    return (f"https://sleepercdn.com/content/nfl/players/thumb/{sid}.jpg"
            if pd.notna(sid) else None)


def team_roster(team_slot: int) -> list[dict]:
    return [p for i, p in enumerate(st.session_state.picks)
            if dl.snake_team_for_pick(p.get("overall", i + 1), int(teams)) == team_slot]


def sim_picks(to_my_turn: bool):
    """Practice mode: auto-draft for the other teams until it's my turn."""
    total_picks = int(teams) * int(rounds)
    for _ in range(int(teams)):
        overall = len(st.session_state.picks) + st.session_state.phantom + 1
        if overall > total_picks:
            break
        team_slot = dl.snake_team_for_pick(overall, int(teams))
        if team_slot == int(slot):
            break
        drafted = {p["key"] for p in st.session_state.picks}
        avail = board[~board["key"].isin(drafted)]
        if avail.empty:
            break
        tdf = pd.DataFrame(team_roster(team_slot),
                           columns=["key", "player", "pos", "team", "bye", "mine", "overall"])
        sugg = dl.suggest_picks(avail, tdf, spots, overall, None, None, int(rounds), top_n=3)
        choice = sugg.sample(1).iloc[0] if len(sugg) else avail.iloc[0]
        _apply_pick(choice, False)
        if not to_my_turn:
            break


# ---------------------------------------------------------------- player card

@st.dialog("Player card", width="large")
def player_card(row: pd.Series):
    c_img, c_head = st.columns([1, 5])
    img = _img_url(row)
    if img:
        c_img.image(img, width=90)
    with c_head:
        st.markdown(f"### {row['player']}")
        bits = [str(row["pos_rank"]), str(row["team"]), f"Bye {row['bye']}"]
        if pd.notna(row.get("age")):
            bits.append(f"Age {int(row['age'])}")
        if pd.notna(row.get("years_exp")):
            exp = int(row["years_exp"])
            bits.append("Rookie" if exp == 0 else f"Year {exp + 1}")
        if isinstance(row.get("college"), str) and row["college"]:
            bits.append(row["college"])
        st.caption(" · ".join(bits))
        flags = []
        if row.get("injury"):
            note = f" ({row['injury_note']})" if isinstance(row.get("injury_note"), str) and row["injury_note"] else ""
            flags.append(f"🩹 **{row['injury']}**{note}")
        if row.get("trending", 0) > 0:
            flags.append(f"🔥 {int(row['trending']):,} Sleeper adds in 24h")
        if flags:
            st.markdown(" · ".join(flags))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Board rank", f"#{int(row['rank'])}", f"Tier {int(row['tier'])}" if row["tier"] else None,
              delta_color="off")
    m2.metric("Experts (ECR)", f"#{int(row['ecr'])}", f"{int(row['ecr_best'])}–{int(row['ecr_worst'])}",
              delta_color="off")
    adp_pick = row["adp_pick"] if isinstance(row["adp_pick"], str) and row["adp_pick"] else "—"
    m3.metric("ADP", adp_pick,
              f"{row['adp']:.1f} overall" if pd.notna(row["adp"]) else None, delta_color="off")
    m4.metric("Value vs ADP", f"{row['value']:+.1f}" if pd.notna(row["value"]) else "—",
              "market lets him fall" if pd.notna(row["value"]) and row["value"] >= 5 else None,
              delta_color="off")
    lasts = row.get("lasts")
    m5.metric("Lasts to your turn", f"{lasts:.0%}" if pd.notna(lasts) else "—",
              f"pick #{next_turn}" if next_turn else None, delta_color="off")

    already = status_by_key.get(row["key"])
    if already:
        st.warning(f"Already drafted: **{already}**")
        if st.button("↩ Un-draft (this was logged by mistake)", width="stretch"):
            remove_pick_by_key(row["key"])
            st.rerun()
    else:
        b1, b2 = st.columns(2)
        if b1.button("✅ Draft to MY team", type="primary", width="stretch"):
            _apply_pick(row, True)
            st.rerun()
        if b2.button(f"🚫 Taken by {on_clock_name}", width="stretch"):
            _apply_pick(row, False)
            st.rerun()

    with st.spinner("Grabbing latest headlines…"):
        news = fetch_player_news(row["player"])
    if news:
        st.markdown("##### 📰 Latest news")
        for n in news:
            extras = " · ".join(x for x in (n["source"], n["when"]) if x)
            st.markdown(f"- [{n['title']}]({n['link']})"
                        + (f" <span style='opacity:.6'>({extras})</span>" if extras else ""),
                        unsafe_allow_html=True)
    l1, l2, _ = st.columns([1, 1, 2])
    l1.link_button("More news", "https://news.google.com/search?q="
                   + quote(f"{row['player']} fantasy"), width="stretch")
    l2.link_button("RotoWire", "https://www.rotowire.com/football/search.php?search="
                   + quote(str(row["player"])), width="stretch")


# ---------------------------------------------------------------- header

on_clock_name = (team_names[dl.snake_team_for_pick(current_overall, int(teams)) - 1]
                 if current_overall <= int(teams) * int(rounds) else "—")

h1, h2, h3, h4 = st.columns([2, 2, 2, 3])
h1.metric("On the clock", f"Round {rnd}, Pick {pick_in_rnd}",
          f"#{current_overall} · {on_clock_name}", delta_color="off")
if on_clock:
    h2.metric("Your turn", "🚨 NOW", "make your pick", delta_color="off")
elif upcoming:
    h2.metric("Your next pick", f"#{upcoming[0]}", f"in {upcoming[0] - current_overall} picks",
              delta_color="off")
else:
    h2.metric("Your next pick", "—", "draft complete", delta_color="off")
h3.metric("Your roster", f"{len(my_players)}/{int(rounds)}")
with h4:
    drafts = board.attrs.get("adp_drafts", 0)
    st.caption(f"Blend: {w_ecr}% FantasyPros · {w_adp}% ADP · {w_csv}% CSV"
               + (f" ({st.session_state.csv_name})" if st.session_state.csv_name else ""))
    if drafts:
        st.caption(f"ADP from {drafts:,} real 12-team PPR drafts through "
                   f"{board.attrs.get('adp_date', '')} · 🔄 in sidebar re-pulls everything")
    with st.expander("Fix pick counter"):
        st.caption("If the board got ahead of you, set the true overall pick number.")
        fixed = st.number_input("Current overall pick", 1, int(teams) * int(rounds),
                                current_overall, key="fix_overall")
        if fixed != current_overall and st.button("Apply", key="apply_fix"):
            st.session_state.phantom += fixed - current_overall
            autosave()
            st.rerun()

if on_clock:
    st.success("**You're on the clock!** Suggestions below ⬇")

if picks:
    ticker = []
    for i, p in list(enumerate(picks))[-6:][::-1]:
        o = p.get("overall", i + 1)
        r_, k_ = dl.round_and_pick(o, int(teams))
        who = team_names[dl.snake_team_for_pick(o, int(teams)) - 1]
        ticker.append(f"{r_}.{k_:02d} {who}: **{p['player']}**"
                      + (" 🟢" if p["mine"] else ""))
    st.caption("🕘 " + " &nbsp;·&nbsp; ".join(ticker), unsafe_allow_html=True)

tab_board, tab_tiers, tab_team, tab_league, tab_log, tab_full = st.tabs(
    ["🎯 Draft Board", "📊 Tiers", "🧑‍🤝‍🧑 My Team", "🏟 League", "📜 Pick Log",
     "📖 Full Rankings"])

# ---------------------------------------------------------------- draft board

with tab_board:
    # --- suggestions
    my_pick_now = upcoming[0] if upcoming else None
    following = upcoming[1] if len(upcoming) > 1 else None
    sugg = dl.suggest_picks(available, my_players, spots, current_overall,
                            my_pick_now, following, int(rounds))
    with st.container(border=True):
        st.markdown("##### 💡 Suggested for your next pick")
        for i, (_, s) in enumerate(sugg.iterrows()):
            marker = "🥇🥈🥉４５"[i] if i < 5 else "·"
            st.markdown(
                f"{marker} **{s['player']}** ({s['pos_rank']}, {s['team']}, bye {s['bye']})"
                f" — {s['why']}")

    # --- quick entry
    st.markdown("##### Log a pick")
    quick_map = {
        f"{int(r['rank'])}. {r['player']} ({r['pos']} · {r['team']})": r
        for _, r in available.iterrows()
    }
    st.session_state._quick_map = quick_map
    qc1, qc2, qc3, qc4, qc5 = st.columns([3.6, 1, 1.2, 1.2, 1])
    qc1.selectbox("Type a name…", quick_map.keys(), index=None, key="quick_pick",
                  placeholder="Type a player name…", label_visibility="collapsed")
    qc2.button("ℹ️ Card", key="btn_card", on_click=request_card, width="stretch",
               help="Open the player card: news, ranks, and draft buttons")
    qc3.button("🚫 Taken", key="btn_taken", on_click=record_quick, args=(False,),
               width="stretch",
               help="Someone else drafted this player")
    qc4.button("✅ My pick", key="btn_mine", type="primary", on_click=record_quick, args=(True,),
               width="stretch")
    qc5.button("↩ Undo", key="btn_undo", on_click=undo_pick, width="stretch",
               disabled=not picks, help="Remove the last logged pick")

    with st.expander("🤖 Practice mode (mock draft)"):
        st.caption("Simulates the other teams' picks so you can rehearse before draft day. "
                   "Don't use this during the real draft — log real picks instead.")
        pm1, pm2 = st.columns(2)
        pm1.button("Sim 1 pick", key="btn_sim1", on_click=sim_picks, args=(False,),
                   width="stretch", disabled=on_clock)
        pm2.button("Sim to my turn", key="btn_simme", on_click=sim_picks, args=(True,),
                   width="stretch", disabled=on_clock)

    # --- filters + table
    fc1, fc2, fc3 = st.columns([2, 2.6, 1.2])
    pos_filter = fc1.multiselect("Position", POSITIONS, default=[],
                                 placeholder="All positions")
    search = fc2.text_input("Search", placeholder="Filter by player or team…")
    show_drafted = fc3.toggle("Show drafted", value=False,
                              help="Keep drafted players on the board, struck through, "
                                   "with who took them")

    # Reset the row selection whenever the view changes so a stale selection
    # can't point at a different player.
    view_sig = (tuple(sorted(pos_filter)), search.strip().lower(), bool(show_drafted))
    if st.session_state.get("_view_sig") != view_sig:
        st.session_state._view_sig = view_sig
        st.session_state.board_nonce += 1

    shown = board if show_drafted else available
    if pos_filter:
        shown = shown[shown["pos"].isin(pos_filter)]
    if search:
        q = search.strip().lower()
        shown = shown[shown["player"].str.lower().str.contains(q, regex=False)
                      | shown["team"].str.lower().str.contains(q, regex=False)]
    shown = shown.head(250).reset_index(drop=True)
    shown = shown.assign(img=shown.apply(_img_url, axis=1),
                         status=shown["key"].map(status_by_key).fillna(""))

    display_cols = ["img", "rank", "player", "pos_rank", "team", "bye", "tier",
                    "ecr", "adp", "value", "lasts", "injury", "trending"]
    if show_drafted:
        display_cols.insert(3, "status")
    disp = shown[display_cols]

    def _row_style(row):
        if shown.loc[row.name, "status"]:
            return ["color: #8a8a8a; text-decoration: line-through"] * len(row)
        color = dl.tier_color(shown.loc[row.name, "tier"])
        return [f"background-color: {color}66" if color else ""] * len(row)

    styled = disp.style.apply(_row_style, axis=1).format(
        {"adp": "{:.1f}", "value": "{:+.1f}", "ecr": "{:.0f}"}, na_rep="—")

    st.caption("👆 Tap the ⚪ at the left edge of a row to open the player card — news, "
               "ranks, and draft buttons. (Or type a name above and hit ℹ️ Card.)")
    event = st.dataframe(
        styled, key=f"board_{st.session_state.board_nonce}",
        on_select="rerun", selection_mode="single-row",
        width="stretch", height=560, hide_index=True,
        column_config={
            "img": st.column_config.ImageColumn(" ", width="small"),
            "rank": st.column_config.NumberColumn("#", width="small"),
            "player": st.column_config.TextColumn("Player", width="medium"),
            "status": st.column_config.TextColumn("Drafted by", width="small"),
            "pos_rank": st.column_config.TextColumn("Pos", width="small"),
            "team": st.column_config.TextColumn("Team", width="small"),
            "bye": st.column_config.NumberColumn("Bye", width="small"),
            "tier": st.column_config.NumberColumn("Tier", width="small"),
            "ecr": st.column_config.NumberColumn("ECR", width="small",
                                                 help="FantasyPros expert consensus rank"),
            "adp": st.column_config.NumberColumn("ADP", width="small",
                                                 help="Average draft position in real drafts"),
            "value": st.column_config.NumberColumn("Val", width="small",
                                                   help="ADP minus blended rank: positive = market lets him fall"),
            "lasts": st.column_config.ProgressColumn(
                "Lasts to your turn", min_value=0, max_value=1, format="percent",
                help="Chance he's still available at your next turn"),
            "injury": st.column_config.TextColumn("Injury", width="small"),
            "trending": st.column_config.NumberColumn(
                "🔥", width="small", help="Sleeper adds in the last 24h"),
        },
    )

    # Open the player card: from the ℹ️ Card button, or by clicking a table row.
    card_key = st.session_state.pop("card_request", None)
    if card_key is not None:
        match = board[board["key"] == card_key]
        if len(match):
            player_card(match.iloc[0])
    elif event.selection.rows:
        sel_row = shown.iloc[event.selection.rows[0]]
        marker = (sel_row["key"], st.session_state.board_nonce)
        if st.session_state.get("_card_marker") != marker:
            st.session_state._card_marker = marker
            player_card(sel_row)

# ---------------------------------------------------------------- tiers

with tab_tiers:
    st.caption("Strikethrough = drafted · **bold green** = yours. "
               "When a tier is nearly empty, the position is about to cliff.")
    cols = st.columns(4)
    for col, pos in zip(cols, ["RB", "WR", "TE", "QB"]):
        with col:
            st.markdown(f"#### {pos}")
            pos_board = board[(board["pos"] == pos) & (board["tier"] > 0)].head(60)
            for tier_num, grp in pos_board.groupby("tier"):
                remaining = (~grp["key"].isin(drafted_keys)).sum()
                color = dl.tier_color(int(tier_num))
                st.markdown(
                    f"<div style='background:{color};padding:2px 8px;border-radius:6px;"
                    f"margin-top:6px;font-weight:600'>Tier {int(tier_num)} "
                    f"<span style='opacity:.7'>({remaining} left)</span></div>",
                    unsafe_allow_html=True)
                lines = []
                for _, p in grp.iterrows():
                    status = taken_by_key.get(p["key"])
                    name = f"{p['player']}"
                    if status == "Me":
                        lines.append(f"<b style='color:#4caf50'>{name} ✔</b>")
                    elif status == "Taken":
                        lines.append(f"<s style='opacity:.45'>{name}</s>")
                    else:
                        lines.append(name)
                st.markdown("<br>".join(lines), unsafe_allow_html=True)

# ---------------------------------------------------------------- my team

with tab_team:
    needs = dl.roster_needs(my_players, spots)
    st.markdown("#### Roster")
    slots_rows = []
    assigned = set()
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        pool = [p for p in picks if p["mine"] and p["pos"] == pos]
        for i in range(int(spots[pos])):
            if i < len(pool):
                p = pool[i]
                assigned.add(p["key"])
                slots_rows.append({"Slot": f"{pos}{i + 1}", "Player": p["player"],
                                   "Team": p["team"], "Bye": str(p["bye"])})
            else:
                slots_rows.append({"Slot": f"{pos}{i + 1}", "Player": "—", "Team": "", "Bye": ""})
    flex_pool = [p for p in picks
                 if p["mine"] and p["pos"] in dl.FLEX_POSITIONS and p["key"] not in assigned]
    for i in range(int(spots["FLEX"])):
        if i < len(flex_pool):
            p = flex_pool[i]
            assigned.add(p["key"])
            slots_rows.append({"Slot": f"FLEX{i + 1}", "Player": p["player"],
                               "Team": p["team"], "Bye": str(p["bye"])})
        else:
            slots_rows.append({"Slot": f"FLEX{i + 1}", "Player": "—", "Team": "", "Bye": ""})
    bench = [p for p in picks if p["mine"] and p["key"] not in assigned]
    for i, p in enumerate(bench):
        slots_rows.append({"Slot": f"BN{i + 1}", "Player": p["player"],
                           "Team": p["team"], "Bye": str(p["bye"])})
    st.dataframe(pd.DataFrame(slots_rows), hide_index=True, width="stretch")

    open_needs = {k: v for k, v in needs.items() if v > 0}
    if open_needs:
        st.info("Still need: " + ", ".join(f"{v}× {k}" for k, v in open_needs.items()))
    if len(my_players):
        bye_counts = my_players["bye"].value_counts()
        stacked = bye_counts[bye_counts >= 3]
        if not stacked.empty:
            for bye_wk, n in stacked.items():
                st.warning(f"⚠ {n} of your players share bye week {bye_wk}")

# ---------------------------------------------------------------- league

with tab_league:
    lc1, lc2 = st.columns([2, 3])
    with lc1:
        st.markdown("#### Team roster")
        viewer = st.selectbox("Team", [f"{i}. {team_names[i - 1]}" for i in range(1, int(teams) + 1)],
                              index=int(slot) - 1, label_visibility="collapsed")
        vslot = int(viewer.split(".")[0])
        roster = team_roster(vslot)
        if roster:
            rows = []
            for p in roster:
                r, k = dl.round_and_pick(p.get("overall", 1), int(teams))
                rows.append({"Pick": f"{r}.{k:02d}", "Player": p["player"],
                             "Pos": p["pos"], "Team": p["team"], "Bye": str(p["bye"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No picks logged for this team yet.")
    with lc2:
        st.markdown("#### Who has what")
        st.caption("Position counts per team — spot who still needs a QB/TE before a run starts.")
        grid = []
        for i in range(1, int(teams) + 1):
            counts = pd.Series([p["pos"] for p in team_roster(i)]).value_counts()
            grid.append({"Team": team_names[i - 1] + (" ⭐" if i == int(slot) else ""),
                         **{pos: int(counts.get(pos, 0)) for pos in POSITIONS}})
        st.dataframe(pd.DataFrame(grid), hide_index=True, width="stretch",
                     height=int(teams) * 35 + 40)

# ---------------------------------------------------------------- pick log

with tab_log:
    if not picks:
        st.caption("No picks logged yet.")
    else:
        log_rows = []
        for i, p in enumerate(picks):
            overall = p.get("overall", i + 1)
            r, k = dl.round_and_pick(overall, int(teams))
            who = team_names[dl.snake_team_for_pick(overall, int(teams)) - 1]
            log_rows.append({"Pick": f"{r}.{k:02d}", "Overall": overall,
                             "Player": p["player"], "Pos": p["pos"], "Team": p["team"],
                             "Who": f"🟢 {who}" if p["mine"] else who})
        st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")
        del_labels = [f"{i + 1}: {r['Player']} (pick {r['Pick']})"
                      for i, r in enumerate(log_rows)]
        del_choice = st.selectbox(
            "Remove a specific pick (fixes mistakes anywhere in the log)",
            del_labels, index=None, placeholder="Choose a pick to remove…")
        if del_choice and st.button("Remove pick"):
            idx = del_labels.index(del_choice)
            st.session_state.picks.pop(idx)
            for later in st.session_state.picks[idx:]:
                if "overall" in later:
                    later["overall"] -= 1
            autosave()
            st.rerun()

# ---------------------------------------------------------------- full rankings

with tab_full:
    full = board.copy()
    full["status"] = full["key"].map(status_by_key).fillna("")
    st.download_button("⬇ Download board as CSV",
                       full.drop(columns=["key", "sleeper_id"], errors="ignore")
                           .to_csv(index=False),
                       file_name="draftaid_board.csv", mime="text/csv")
    st.dataframe(
        full[["rank", "player", "pos_rank", "team", "bye", "tier", "ecr", "ecr_best",
              "ecr_worst", "adp", "adp_pick", "value", "injury", "trending", "status"]],
        hide_index=True, width="stretch", height=600)
