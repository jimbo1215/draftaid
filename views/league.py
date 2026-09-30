"""League: standings, FAAB budgets, scoreboard by week, roster viewer."""

import streamlit as st

import theme as t
from season_ui import header, league_ctx, my_team_id

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
header(league, "League")

# --- standings + FAAB
standings = sorted(league["teams"], key=lambda x: (-x["wins"], -x["points_for"]))
budget = league["faab_budget"] or 1
cut = league.get("playoff_teams") or 0
rows = []
for i, x in enumerate(standings, 1):
    rec = f"{x['wins']}-{x['losses']}" + (f"-{x['ties']}" if x["ties"] else "")
    faab = ""
    if league["uses_faab"]:
        left = max(0, league["faab_budget"] - x["faab_spent"])
        faab = (f"<td class='r hide-m'><div style='display:flex;align-items:center;gap:8px;"
                f"justify-content:flex-end'><span>${left}</span><div class='da-bar' "
                f"style='width:54px;margin:0'><i style='width:{left / budget * 100:.0f}%'></i>"
                f"</div></div></td>")
    cls = " ".join(c for c, on in (("me", x["team_id"] == mid), ("cut", cut and i == cut)) if on)
    rows.append(
        f"<tr class='{cls}'>"
        f"<td class='c da-dim'>{i}</td>"
        f"<td><span class='da-strong'>{t.esc(x['name'])}</span></td>"
        f"<td class='c'>{rec}</td><td class='r'>{x['points_for']:.1f}</td>"
        f"<td class='r hide-m da-dim'>{x['points_against']:.1f}</td>{faab}"
        f"<td class='r hide-m da-dim'>{x['moves']}</td></tr>")
faab_head = "<th class='r hide-m'>FAAB</th>" if league["uses_faab"] else ""
t.section("Standings", f"top {cut} make the playoffs" if cut else "")
t.html(f"<div class='da-wrap'><table class='da-table'><thead><tr><th class='c'>#</th>"
       f"<th>Team</th><th class='c'>W-L</th><th class='r'>PF</th>"
       f"<th class='r hide-m'>PA</th>{faab_head}<th class='r hide-m'>Moves</th></tr></thead>"
       f"<tbody>{''.join(rows)}</tbody></table></div>")

# --- scoreboard
weeks = sorted({m["week"] for m in league["schedule"]
                if m["week"] and m["week"] <= league["week"]})
c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom", gap="small")
with c1:
    t.section("Scoreboard")
wk = c2.selectbox("Week", weeks, index=len(weeks) - 1 if weeks else 0,
                  format_func=lambda w: f"Week {w}", label_visibility="collapsed")
cards = []
games = [m for m in league["schedule"] if m["week"] == wk]
for m in sorted(games, key=lambda m: mid not in (m["home_id"], m["away_id"])):
    final = m["winner"] in ("HOME", "AWAY")
    lines = []
    for side in ("away", "home"):
        tid, pts = m[f"{side}_id"], m[f"{side}_pts"]
        won = final and m["winner"] == side.upper()
        cls = "w" if won else "l" if final else ""
        lines.append(f"<div class='ln {cls}'><span>{t.esc(names.get(tid, '?'))}</span>"
                     f"<span class='p'>{pts:.1f}</span></div>")
    status = "Final" if final else ("Not started" if all(
        m[f"{s}_pts"] == 0 for s in ("home", "away")) else "In progress")
    me_cls = " me" if mid in (m["home_id"], m["away_id"]) else ""
    cards.append(f"<div class='da-sb{me_cls}'>{''.join(lines)}"
                 f"<div class='ft'>{status}</div></div>")
if cards:
    t.html(f"<div class='da-grid'>{''.join(cards)}</div>")

# --- roster viewer
t.section("Rosters")
order = sorted(league["teams"], key=lambda x: x["team_id"] != mid)
sel = st.selectbox("Team", [x["name"] for x in order], index=None,
                   placeholder="View any team's roster…", label_visibility="collapsed")
if sel:
    tid = next(k for k, v in names.items() if v == sel)
    t.html(t.roster_table(teams[tid]))
