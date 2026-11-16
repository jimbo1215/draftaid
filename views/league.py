"""League: standings with playoff odds and FAAB, the scoreboard, any roster."""

import streamlit as st

from core.outlook import DECIDED
from ui import theme as t
from ui.context import header, load

ctx = load(require_team=False)
league, names, mid = ctx.league, ctx.names, ctx.my_id
out = ctx.outlook()
header(ctx, "League")

# --- standings
standings = sorted(league["teams"], key=lambda x: (-x["wins"], -x["points_for"]))
budget = league["faab_budget"] or 1
cut = (out.playoff_teams if out else league.get("playoff_teams")) or 0
head = [("#", "c"), ("Team", ""), ("W-L", "c"), ("PF", "r"), ("PA", "r hide-m")]
if out:
    head.append(("Playoff odds", "r"))
if league["uses_faab"]:
    head.append(("FAAB left", "r hide-m"))
head.append(("Moves", "r hide-m"))
rows = []
for i, x in enumerate(standings, 1):
    rec = f"{x['wins']}-{x['losses']}" + (f"-{x['ties']}" if x["ties"] else "")
    cls = " ".join(c for c, on in (("me", x["team_id"] == mid), ("cut", cut and i == cut)) if on)
    cells = (f"<td class='c da-dim'>{i}</td>"
             f"<td><span class='da-strong'>{t.esc(x['name'])}</span></td>"
             f"<td class='c'>{rec}</td><td class='r'>{x['points_for']:.1f}</td>"
             f"<td class='r hide-m da-dim'>{x['points_against']:.1f}</td>")
    if out:
        cells += f"<td class='r'>{t.odds(out.playoff.get(x['team_id']))}</td>"
    if league["uses_faab"]:
        left = max(0, league["faab_budget"] - x["faab_spent"])
        cells += f"<td class='r hide-m'>${left}</td>"
    cells += f"<td class='r hide-m da-dim'>{x['moves']}</td>"
    rows.append(f"<tr class='{cls}'>{cells}</tr>")
t.section("Standings", f"the line marks the playoff cut: top {cut} get in" if cut else "")
t.html(t.table(head, rows))

# --- scoreboard
weeks = sorted({m["week"] for m in league["schedule"]
                if m["week"] and m["week"] <= league["matchup_period"]})
c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom", gap="small")
with c1:
    t.section("Scoreboard")
wk = c2.selectbox("Week", weeks, index=len(weeks) - 1 if weeks else 0,
                  format_func=lambda w: f"Week {w}", label_visibility="collapsed")
cards = []
games = [m for m in league["schedule"] if m["week"] == wk]
for m in sorted(games, key=lambda m: mid not in (m["home_id"], m["away_id"])):
    final = m["winner"] in DECIDED
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
    t.html(t.roster_table(ctx.teams[tid], playoffs=bool(ctx.src.playoff_weeks)))
