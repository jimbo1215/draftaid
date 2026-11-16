"""My Team: this week's matchup, where you stand, lineup checks, and the roster."""

import math

import streamlit as st

from core.news import fetch_player_news
from core.outlook import DECIDED
from core.startsit import bench_points, bye_watch, start_sit
from ui import theme as t
from ui.context import header, load

ctx = load()
league, me, roster, names = ctx.league, ctx.me, ctx.roster, ctx.names
mid, week, period = ctx.my_id, league["week"], league["matchup_period"]
out = ctx.outlook()

standings = sorted(league["teams"], key=lambda x: (-x["wins"], -x["points_for"]))
place = next(i for i, x in enumerate(standings) if x["team_id"] == mid) + 1
record = f"{me['wins']}-{me['losses']}" + (f"-{me['ties']}" if me.get("ties") else "")
header(ctx, me["name"])


def _proj_total(df) -> float:
    live = df[df["starter"] & (df["slot"] != "IR")] if not df.empty else df
    return float(live["week_proj"].fillna(0).sum()) if "week_proj" in live else 0.0


# --- this week's matchup
mu = next((m for m in league["schedule"]
           if m["week"] == period and mid in (m["home_id"], m["away_id"])), None)
if mu:
    mine, theirs = ("home", "away") if mu["home_id"] == mid else ("away", "home")
    opp_id = mu[f"{theirs}_id"]
    my_pts, opp_pts = mu[f"{mine}_pts"], mu[f"{theirs}_pts"]
    my_proj = mu.get(f"{mine}_live") or _proj_total(roster)
    opp_proj = mu.get(f"{theirs}_live") or _proj_total(ctx.teams.get(opp_id, roster.iloc[0:0]))
    opp = next((x for x in league["teams"] if x["team_id"] == opp_id), {})
    final = mu["winner"] in DECIDED
    started = bool(my_pts or opp_pts)

    win_p = None
    if not final:
        if out and period in out.week_win:
            win_p = out.week_win[period][1]
        elif my_proj and opp_proj:   # no simulation: a normal margin on the projections
            win_p = 0.5 * (1 + math.erf((my_proj - opp_proj) / (35 * math.sqrt(2))))
    if win_p is not None:
        foot = (f"<div class='wp'>{t.split_bar(win_p)}</div><div class='wpl'>"
                f"<span><b>{t.fmt_pct(win_p)}</b> win probability</span>"
                f"<span>{t.fmt_pct(1 - win_p)}</span></div>")
    else:
        foot = f"<div class='wpl'><span>{'Final' if final else 'Projections unavailable'}</span></div>"
    proj_label = "Projected final" if started and not final else "Projected"
    t.html(
        f"<div class='da-mu'><div class='rowm'>"
        f"<div class='side'><div class='who'>You · {record}</div>"
        f"<div class='tm'>{t.esc(me['name'])}</div>"
        f"<div class='pts'>{my_pts:.1f}</div>"
        f"<div class='pj'>{proj_label} {my_proj:.1f}</div></div>"
        f"<div class='vs'>WEEK {period}</div>"
        f"<div class='side r'><div class='who'>{opp.get('wins', 0)}-{opp.get('losses', 0)} · Opponent</div>"
        f"<div class='tm'>{t.esc(names.get(opp_id, 'Opponent'))}</div>"
        f"<div class='pts'>{opp_pts:.1f}</div>"
        f"<div class='pj'>{proj_label} {opp_proj:.1f}</div></div></div>{foot}</div>")

# --- where you stand
tiles = [{"k": "Standing", "v": t.ordinal(place), "d": f"of {len(standings)} · {record}"}]
if out:
    tiles += [
        {"k": "Playoff odds", "v": t.fmt_pct(out.playoff[mid]), "bar": out.playoff[mid],
         "lead": True, "d": f"Projected finish {t.ordinal(round(out.avg_seed[mid]))} · "
                            f"top {out.playoff_teams} get in"},
        {"k": "Title odds", "v": t.fmt_pct(out.title[mid]), "bar": out.title[mid],
         "d": f"An average team: {100 / len(standings):.0f}%"},
    ]
games = max(1, me["wins"] + me["losses"] + me.get("ties", 0))
tiles.append({"k": "Points for", "v": f"{me['points_for']:.0f}",
              "d": f"{me['points_for'] / games:.1f} per week"})
if league["uses_faab"]:
    tiles.append({"k": "FAAB left", "v": f"${max(0, league['faab_budget'] - me['faab_spent'])}",
                  "d": f"{me['moves']} moves made"})
t.tiles(tiles)

lev = out.leverage if out else None
if lev and lev["period"] == period:
    swing = (lev["playoff_win"] - lev["playoff_loss"]) * 100
    t.alert("info", "This week",
            f"Beat <b>{t.esc(names.get(lev['opp_id'], 'your opponent'))}</b> and your playoff "
            f"odds are <b>{t.fmt_pct(lev['playoff_win'])}</b>. Lose and they're "
            f"<b>{t.fmt_pct(lev['playoff_loss'])}</b>.", f"{swing:.0f}-point swing")
    st.page_link("views/playoffs.py", label="See the full playoff picture",
                 icon=":material/arrow_forward:")

# --- lineup checks
lineup = league.get("lineup")
flags = start_sit(roster, lineup, week)
gap = bench_points(roster, lineup)
hurt_starters = roster[(roster["injury"].astype(str).str.len() > 0) & roster["starter"]
                       & (roster["slot"] != "IR")]
byes = bye_watch(roster, lineup, [w for w in range(week + 1, week + 4)
                                  if w <= league.get("final_week", 17)])
t.section("Lineup check", "expert ranks from FantasyPros · projections from ESPN in your scoring")
if not flags and hurt_starters.empty and gap["gap"] < 1:
    t.alert("good", "All clear", "Your lineup matches this week's expert ranks and "
            "no starter is on the injury report.")
for f in flags:
    if f["in"] is None:
        t.alert("bad", "Empty", f"<b>{t.esc(f['out']['player'])}</b> is in your lineup",
                t.esc(f["note"]))
    else:
        t.alert("", "Swap", f"Start <b>{t.esc(f['in']['player'])}</b> over "
                f"<b>{t.esc(f['out']['player'])}</b>", t.esc(f["note"]))
if gap["gap"] >= 1:
    t.alert("", "Points", f"Your best available lineup projects <b>{gap['gap']:.1f} more "
            f"points</b> than the one you've set", f"{gap['best']:.1f} vs {gap['set']:.1f}")
for _, p in hurt_starters.iterrows():
    serious = str(p["injury"]).title() not in ("Questionable", "Day To Day")
    t.alert("bad" if serious else "", "Injury" if serious else "Monitor",
            f"<b>{t.esc(p['player'])}</b> is in your lineup", t.esc(p["injury"]))
for b in byes:
    who = ", ".join(t.esc(p["player"]) for p in b["players"])
    t.alert("info", f"Week {b['week']}", f"On bye: <b>{who}</b>. Line up a fill-in before "
            "waivers run.", f"{len(b['players'])} starter{'s' if len(b['players']) > 1 else ''}")

# --- roster
has_playoffs = bool(ctx.src.playoff_weeks) and roster["po_rank"].notna().any()
note = "Rest of season = expert positional rank"
if has_playoffs:
    po = ctx.src.playoff_weeks
    note += f" · Playoffs = projected positional rank, weeks {po[0]}–{po[-1]}"
t.section("Roster", note)
t.html(t.roster_table(roster, playoffs=has_playoffs))

# --- player news
t.section("Player news")
sel = st.selectbox("Headlines for", roster["player"].tolist(), index=None,
                   placeholder="Pick a player for the latest headlines…",
                   label_visibility="collapsed")
if sel:
    with st.spinner("Grabbing headlines…"):
        news = fetch_player_news(sel)
    if news:
        t.html(t.news_list(news))
    else:
        st.caption("No recent headlines found.")
