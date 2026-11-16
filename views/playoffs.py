"""Playoff Picture: odds from simulating the rest of the season, what this
week is worth, who is really good, and how your lineup projects in the
weeks that decide the title."""

import pandas as pd
import streamlit as st

from core import outlook as sim
from core.lineup import week_lineup
from core.values import pos_rank_number
from core.waivers import playoff_stashes
from ui import theme as t
from ui.blocks import stash_table
from ui.context import header, load

ctx = load()
league, names, mid, roster = ctx.league, ctx.names, ctx.my_id, ctx.roster
out = ctx.outlook()
header(ctx, "Playoff Picture",
       "The rest of the season, played out 10,000 times from every team's projected "
       "lineup each week, so byes, injuries, and playoff-week matchups are in the numbers.")

if out is None:
    t.empty("<b>The season simulation couldn't run.</b> It needs the league schedule from "
            "ESPN and weekly projections from Sleeper. Try the refresh button; the rest of "
            "the app works without it.")
    st.stop()

reg_weeks = league["regular_season_weeks"]
team_by_id = {x["team_id"]: x for x in league["teams"]}
n_teams = len(team_by_id)
me = team_by_id[mid]


def _status(tid) -> str:
    if out.games_left == 0:
        return ""
    if out.playoff[tid] >= 1:
        return " " + t.tag("Clinched", "good")
    if out.playoff[tid] <= 0:
        return " " + t.tag("Eliminated")
    return ""


# --- your odds
ties = me.get("ties", 0)
exp_w = out.exp_wins[mid]
tiles = [{"k": "Make the playoffs", "v": t.fmt_pct(out.playoff[mid]), "bar": out.playoff[mid],
          "lead": True, "d": f"Top {out.playoff_teams} of {n_teams} qualify"}]
if out.byes:
    tiles.append({"k": "First-round bye", "v": t.fmt_pct(out.bye[mid]), "bar": out.bye[mid],
                  "d": f"Top {out.byes} seeds skip a round"})
tiles += [
    {"k": "Win the title", "v": t.fmt_pct(out.title[mid]), "bar": out.title[mid],
     "d": f"Reach the final: {t.fmt_pct(out.finals[mid])}"},
    {"k": "Projected record", "v": f"{exp_w:.1f}<small>– {max(0.0, reg_weeks - exp_w - ties):.1f}</small>",
     "d": f"Most likely finish: {t.ordinal(out.seed_dist[mid].index(max(out.seed_dist[mid])) + 1)}"},
]
t.tiles(tiles)

# --- this week's swing
lev = out.leverage
if lev:
    t.section("What this week is worth",
              f"Week {lev['period']} vs {t.esc(names.get(lev['opp_id'], '?'))} · "
              f"{t.fmt_pct(lev['p_win'])} to win")
    t.tiles([
        {"k": "If you win", "v": t.fmt_pct(lev["playoff_win"]), "bar": lev["playoff_win"],
         "d": f"to make the playoffs · title {t.fmt_pct(lev['title_win'])}"},
        {"k": "If you lose", "v": t.fmt_pct(lev["playoff_loss"]), "bar": lev["playoff_loss"],
         "d": f"to make the playoffs · title {t.fmt_pct(lev['title_loss'])}"},
    ])

# --- remaining schedule
if out.week_win:
    rows = []
    for period, (opp_id, p) in sorted(out.week_win.items()):
        opp = team_by_id.get(opp_id, {})
        rows.append(f"<tr><td class='da-dim'>Week {period}</td>"
                    f"<td><span class='da-strong'>{t.esc(names.get(opp_id, '?'))}</span></td>"
                    f"<td class='c hide-m'>{opp.get('wins', 0)}-{opp.get('losses', 0)}</td>"
                    f"<td class='r'>{t.odds(p)}</td></tr>")
    t.section("Your remaining games", "chance to win each")
    t.html(t.table([("", ""), ("Opponent", ""), ("Record", "c hide-m"),
                    ("Win probability", "r")], rows))

# --- league odds
order = sorted(out.tids, key=lambda tid: (-out.playoff[tid], -out.title[tid]))
head = [("Team", ""), ("Record", "c"), ("Proj. wins", "r hide-m"), ("Playoffs", "r")]
if out.byes:
    head.append(("Bye", "r hide-m"))
head += [("Title", "r"), ("Finish", "hide-m")]
rows = []
for tid in order:
    x = team_by_id[tid]
    rec = f"{x['wins']}-{x['losses']}" + (f"-{x['ties']}" if x.get("ties") else "")
    cells = (f"<td><span class='da-strong'>{t.esc(names[tid])}</span>{_status(tid)}</td>"
             f"<td class='c'>{rec}</td><td class='r hide-m'>{out.exp_wins[tid]:.1f}</td>"
             f"<td class='r'>{t.odds(out.playoff[tid])}</td>")
    if out.byes:
        cells += f"<td class='r hide-m'>{t.fmt_pct(out.bye[tid])}</td>"
    cells += (f"<td class='r'>{t.fmt_pct(out.title[tid])}</td>"
              f"<td class='hide-m' style='min-width:110px'>"
              f"{t.seed_bars(out.seed_dist[tid], out.playoff_teams)}</td>")
    rows.append(f"<tr class='{'me' if tid == mid else ''}'>{cells}</tr>")
t.section("Every team's odds",
          "Finish = odds of each final standing, 1st on the left; gray is outside the playoffs")
t.html(t.table(head, rows))

# --- power rankings
model = ctx.model()
power = sim.power_rankings(league, model)
sos = sim.remaining_sos(league, model)
avg_strength = float(power["strength"].mean()) or 1.0
head = [("#", "c"), ("Team", ""), ("Lineup strength", "r"), ("Scored / wk", "r hide-m"),
        ("All-play", "c hide-m"), ("Luck", "r")]
if sos:
    head.append(("Schedule left", "c hide-m"))
rows = []
for _, p in power.iterrows():
    tid = p["team_id"]
    cells = (f"<td class='c da-dim'>{p['rank']}</td>"
             f"<td><span class='da-strong'>{t.esc(p['name'])}</span></td>"
             f"<td class='r'>{p['strength']:.1f}</td>"
             f"<td class='r hide-m da-dim'>{t.fmt_pts(p['ppg'])}</td>"
             f"<td class='c hide-m'>{p['ap_w']:.0f}-{p['ap_l']:.0f}</td>"
             f"<td class='r'>{t.delta(p['luck'], ' W')}</td>")
    if sos:
        opp = sos.get(tid)
        if opp is None:
            cells += "<td class='heat hide-m da-dim'>–</td>"
        else:   # weaker opponents = an easier road = favorable
            cells += t.heat(f"{opp:.1f}", max(-1.0, min(1.0, (avg_strength - opp) / 12.0)),
                            "hide-m")
    rows.append(f"<tr class='{'me' if tid == mid else ''}'>{cells}</tr>")
t.section("Power rankings",
          "Lineup strength = projected points per week from here · All-play = record if "
          "you'd played every team every week · Luck = real wins above that pace")
t.html(t.table(head, rows))

# --- the playoff weeks
po_weeks = [w for w in ctx.src.playoff_weeks if w in ctx.src.weeks]
if po_weeks:
    # Everyone who would start in at least one playoff week, by total projection.
    starters: dict = {}
    for wk in po_weeks:
        for _, p in week_lineup(roster, ctx.lineup, wk):
            starters[p["id"]] = p["row"]
    span_weeks = [w for w in ctx.src.weeks]
    rows = []
    for row in sorted(starters.values(), key=lambda r: -(r.get("po_pts") or 0)):
        wk_pts = row.get("wk") or {}
        usual = [v for w, v in wk_pts.items() if w in span_weeks and v > 0]
        base = sum(usual) / len(usual) if usual else 0.0
        cells = f"<td>{t.who(row, small=True)}</td>"
        for wk in po_weeks:
            pts, opp = wk_pts.get(wk), (row.get("wk_opp") or {}).get(wk)
            if not pts:
                cells += "<td class='heat da-dim'>out</td>"
                continue
            diff = max(-1.0, min(1.0, (pts - base) / base / 0.3)) if base else 0.0
            label = f"{pts:.1f}" + (f" <span class='da-dim hide-m'>{t.esc(opp)}</span>"
                                    if opp else "")
            cells += t.heat(label, diff)
        rank = f"{row['pos']}{int(row['po_rank'])}" if t.isnum(row.get("po_rank")) else "–"
        cells += (f"<td class='r'><span class='da-strong'>{t.fmt_pts(row.get('po_pts'))}</span></td>"
                  f"<td class='r hide-m'>{rank}</td>")
        rows.append(f"<tr>{cells}</tr>")
    head = ([("Player", "")] + [((f"Week {w}", f"W{w}"), "c") for w in po_weeks]
            + [("Total", "r"), ("Playoff rank", "r hide-m")])
    t.section(f"Your playoff lineup, weeks {po_weeks[0]}–{po_weeks[-1]}",
              "Projected points and opponent · blue = better than his usual week, red = worse")
    t.html(t.table(head, rows))

    proj = model.proj
    tiles = []
    for wk in po_weeks:
        mine_pts = proj.get(mid, {}).get(wk, 0.0)
        rank = 1 + sum(1 for tid in proj if proj[tid].get(wk, 0.0) > mine_pts)
        tiles.append({"k": f"Week {wk} lineup", "v": f"{mine_pts:.1f}",
                      "d": f"{t.ordinal(rank)} of {n_teams} by projection"})
    t.tiles(tiles)

    # --- who gets better in the playoffs
    risers = []
    for tid, df in ctx.teams.items():
        if tid == mid or df.empty:
            continue
        for _, p in df[df["po_rank"].notna() & df["pos"].isin(("QB", "RB", "WR", "TE"))].iterrows():
            ros_n = pos_rank_number(p.get("ros_pos_rank"))
            limit = 12 if p["pos"] in ("QB", "TE") else 24
            if ros_n and p["po_rank"] <= limit and ros_n - p["po_rank"] >= 5:
                risers.append((ros_n - p["po_rank"], names[tid], p))
    risers.sort(key=lambda r: -r[0])
    if risers:
        rows = [f"<tr><td>{t.who(p, small=True)}</td><td class='hide-m da-dim'>{t.esc(owner)}</td>"
                f"<td class='r da-dim'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
                f"<td class='r'><span class='da-strong'>{p['pos']}{int(p['po_rank'])}</span></td>"
                f"<td class='r hide-m'>{t.delta(gain, '', 0)}</td></tr>"
                for gain, owner, p in risers[:6]]
        t.section("Trade targets with a playoff boost",
                  "Rivals' players who rank higher in the playoff weeks than over the rest "
                  "of the season")
        t.html(t.table([("Player", ""), ("Team", "hide-m"), (("Rest of season", "Season"), "r"),
                        ("Playoffs", "r"), ("Spots gained", "r hide-m")], rows))

    try:
        stashes = playoff_stashes(ctx.free_agents(), roster, ctx.lineup, po_weeks)
    except Exception:
        stashes = pd.DataFrame()
    if not stashes.empty:
        t.section("Playoff stashes on the wire",
                  "Free agents who would raise your projected playoff lineups")
        t.html(stash_table(stashes))
        st.page_link("views/waivers.py", label="Price them on the Waiver Wire",
                     icon=":material/arrow_forward:")
else:
    t.section("The playoff weeks")
    t.empty("Week-by-week projections for the playoff weeks aren't available right now, "
            "so the playoff lineup view is hidden. The odds above still run.")

with st.expander("How these numbers are made"):
    st.markdown(
        f"- **Team strength, week by week.** For every remaining week, each team's best "
        f"lineup is built from Sleeper's weekly projections for its current roster, so a "
        f"bye week or an injured starter lowers that week's number. Projections are "
        f"rescaled to how your league actually scores, then each team is nudged toward "
        f"its own scoring average.\n"
        f"- **10,000 seasons.** Every remaining game is played with a weekly spread of "
        f"about {model.sigma:.0f} points per team (measured from your league's scores), "
        f"plus uncertainty about how good each team really is. Seeds go by record, then "
        f"points for; the bracket is played out with each team's projection for those "
        f"exact playoff weeks.\n"
        f"- **Games in progress** use ESPN's live projected finals, so the odds tighten "
        f"as Sunday goes on.\n"
        f"- **What it can't see:** future trades and pickups, head-to-head tiebreakers, "
        f"and players beating their projections. Treat a few points of difference as a "
        f"tie.")
