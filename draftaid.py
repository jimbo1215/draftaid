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
from data_sources import build_board, parse_rankings_csv

st.set_page_config(page_title="DraftAid", page_icon="🏈", layout="wide")

AUTOSAVE = Path(__file__).parent / "draft_autosave.json"
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
DEFAULT_TEAM_NAMES = ("Buddy\nGiuseppe\nScab\nRandino\nNuzzo\nGeiger\n"
                      "Bullock\nJimmy (You)\nClaire\nDro\nLamart\nLogik")

# ---------------------------------------------------------------- state

if "picks" not in st.session_state:
    st.session_state.picks = []          # [{key, player, pos, team, bye, mine}]
if "phantom" not in st.session_state:
    st.session_state.phantom = 0         # unlogged picks (counter resync)
if "csv_ranks" not in st.session_state:
    st.session_state.csv_ranks = None
if "csv_name" not in st.session_state:
    st.session_state.csv_name = ""


def autosave():
    try:
        AUTOSAVE.write_text(json.dumps({
            "picks": st.session_state.picks,
            "phantom": st.session_state.phantom,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }))
    except OSError:
        pass


def load_state(payload: dict):
    st.session_state.picks = payload.get("picks", [])
    st.session_state.phantom = payload.get("phantom", 0)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🏈 DraftAid")

    with st.expander("League settings", expanded=False):
        teams = st.number_input("Teams", 4, 20, 12)
        slot = st.number_input("Your draft slot", 1, int(teams), 8)
        rounds = st.number_input("Rounds", 8, 25, 14)
        names_raw = st.text_area("Team names (draft order, one per line)",
                                 DEFAULT_TEAM_NAMES, height=170)
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
    team_names = [n.strip() for n in names_raw.replace(",", "\n").splitlines() if n.strip()]
    team_names = (team_names
                  + [f"Team {i}" for i in range(len(team_names) + 1, int(teams) + 1)])[:int(teams)]

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
        json.dumps({"picks": st.session_state.picks, "phantom": st.session_state.phantom}),
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

current_overall = len(picks) + st.session_state.phantom + 1
rnd, pick_in_rnd = dl.round_and_pick(current_overall, int(teams))
upcoming = dl.next_my_picks(current_overall, int(slot), int(teams), int(rounds))
on_clock = bool(upcoming) and upcoming[0] == current_overall
next_turn = (upcoming[1] if on_clock else upcoming[0]) if len(upcoming) > (1 if on_clock else 0) else None

available = board[~board["key"].isin(drafted_keys)].copy()
if next_turn is not None:
    available["lasts"] = [dl.survival_probability(a, s, next_turn)
                          for a, s in zip(available["adp"], available["adp_std"])]
else:
    available["lasts"] = float("nan")

# --------------------------------------------------------- pick callbacks


def _apply_pick(row: pd.Series, mine: bool):
    st.session_state.picks.append({
        "key": row["key"], "player": row["player"], "pos": row["pos"],
        "team": row["team"], "bye": int(row["bye"]), "mine": mine,
        "overall": len(st.session_state.picks) + st.session_state.phantom + 1,
    })
    autosave()


def record_quick(mine: bool):
    label = st.session_state.get("quick_pick")
    options = st.session_state.get("_quick_map", {})
    if label and label in options:
        _apply_pick(options[label], mine)
        st.session_state.quick_pick = None


def record_selected(mine: bool):
    table_key = st.session_state.get("_table_key", "")
    sel = st.session_state.get(table_key)
    shown = st.session_state.get("_shown_rows")
    if sel is None or shown is None:
        return
    for r in sel["selection"]["rows"]:
        if r < len(shown):
            _apply_pick(shown.iloc[r], mine)


def undo_pick():
    if st.session_state.picks:
        st.session_state.picks.pop()
        autosave()


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
    qc1, qc2, qc3, qc4 = st.columns([4, 1.2, 1.2, 1])
    qc1.selectbox("Type a name…", quick_map.keys(), index=None, key="quick_pick",
                  placeholder="Type a player name…", label_visibility="collapsed")
    qc2.button("🚫 Taken", key="btn_taken", on_click=record_quick, args=(False,),
               width="stretch",
               help="Someone else drafted this player")
    qc3.button("✅ My pick", key="btn_mine", type="primary", on_click=record_quick, args=(True,),
               width="stretch")
    qc4.button("↩ Undo", key="btn_undo", on_click=undo_pick, width="stretch",
               disabled=not picks, help="Remove the last logged pick")

    if picks:
        last = ", ".join(f"{p['player']}{' (you)' if p['mine'] else ''}"
                         for p in picks[-4:][::-1])
        st.caption(f"Recent: {last}")

    with st.expander("🤖 Practice mode (mock draft)"):
        st.caption("Simulates the other teams' picks so you can rehearse before draft day. "
                   "Don't use this during the real draft — log real picks instead.")
        pm1, pm2 = st.columns(2)
        pm1.button("Sim 1 pick", key="btn_sim1", on_click=sim_picks, args=(False,),
                   width="stretch", disabled=on_clock)
        pm2.button("Sim to my turn", key="btn_simme", on_click=sim_picks, args=(True,),
                   width="stretch", disabled=on_clock)

    # --- filters + table
    fc1, fc2 = st.columns([2, 3])
    pos_filter = fc1.multiselect("Position", POSITIONS, default=[],
                                 placeholder="All positions")
    search = fc2.text_input("Search", placeholder="Filter by player or team…")

    shown = available
    if pos_filter:
        shown = shown[shown["pos"].isin(pos_filter)]
    if search:
        q = search.strip().lower()
        shown = shown[shown["player"].str.lower().str.contains(q, regex=False)
                      | shown["team"].str.lower().str.contains(q, regex=False)]
    shown = shown.head(200).reset_index(drop=True)

    def _img_url(r):
        if r["pos"] == "DST":
            return f"https://sleepercdn.com/images/team_logos/nfl/{str(r['team']).lower()}.png"
        sid = r.get("sleeper_id")
        return (f"https://sleepercdn.com/content/nfl/players/thumb/{sid}.jpg"
                if pd.notna(sid) else None)

    shown = shown.assign(img=shown.apply(_img_url, axis=1))
    st.session_state._shown_rows = shown

    display_cols = ["img", "rank", "player", "pos_rank", "team", "bye", "tier",
                    "ecr", "adp", "value", "lasts", "injury", "trending"]
    disp = shown[display_cols]

    def _tier_row_style(row):
        color = dl.tier_color(shown.loc[row.name, "tier"])
        return [f"background-color: {color}66" if color else ""] * len(row)

    styled = disp.style.apply(_tier_row_style, axis=1).format(
        {"adp": "{:.1f}", "value": "{:+.1f}", "ecr": "{:.0f}"}, na_rep="—")

    table_key = f"board_{len(picks)}_{len(shown)}"
    st.session_state._table_key = table_key
    event = st.dataframe(
        styled, key=table_key, on_select="rerun", selection_mode="single-row",
        width="stretch", height=560, hide_index=True,
        column_config={
            "img": st.column_config.ImageColumn(" ", width="small"),
            "rank": st.column_config.NumberColumn("#", width="small"),
            "player": st.column_config.TextColumn("Player", width="medium"),
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
    if event.selection.rows:
        sel_row = shown.iloc[event.selection.rows[0]]
        sc1, sc2, sc3, sc4, sc5 = st.columns([2.6, 1.2, 1.2, 0.8, 0.9])
        sc1.markdown(f"Selected: **{sel_row['player']}** ({sel_row['pos']} · {sel_row['team']})")
        sc2.button("🚫 Taken ", on_click=record_selected, args=(False,),
                   width="stretch", key="sel_taken")
        sc3.button("✅ My pick ", type="primary", on_click=record_selected, args=(True,),
                   width="stretch", key="sel_mine")
        sc4.link_button("📰 News", "https://news.google.com/search?q="
                        + quote(f"{sel_row['player']} fantasy"), width="stretch")
        sc5.link_button("🧠 Roto", "https://www.rotowire.com/football/search.php?search="
                        + quote(sel_row["player"]), width="stretch")

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
    full["status"] = full["key"].map(taken_by_key).fillna("")
    st.download_button("⬇ Download board as CSV",
                       full.drop(columns=["key", "sleeper_id"], errors="ignore")
                           .to_csv(index=False),
                       file_name="draftaid_board.csv", mime="text/csv")
    st.dataframe(
        full[["rank", "player", "pos_rank", "team", "bye", "tier", "ecr", "ecr_best",
              "ecr_worst", "adp", "adp_pick", "value", "injury", "trending", "status"]],
        hide_index=True, width="stretch", height=600)
