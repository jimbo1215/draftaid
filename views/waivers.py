"""Waiver wire: free agents ranked by what they do for YOUR lineup, each paired
with the right drop and a FAAB bid sized to the rivals who'd compete for him."""

import pandas as pd
import streamlit as st

import theme as t
from season import drop_costs, waiver_targets
from season_ui import free_agents, header, league_ctx, my_team_id, need_my_team

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

me = next(x for x in league["teams"] if x["team_id"] == mid)
uses_faab = league["uses_faab"]
budget = league["faab_budget"]
left = {x["team_id"]: max(0, budget - x["faab_spent"]) for x in league["teams"]}
my_left = left[mid]
rivals = [v for k, v in left.items() if k != mid]

header(league, "Waiver Wire",
       "Every free agent scored against your actual lineup, with the drop that "
       "makes room and a bid priced to beat the rivals who'd want him.")

if uses_faab:
    rank = sorted(left.values(), reverse=True).index(my_left) + 1
    t.tiles([
        {"k": "Your FAAB", "v": f"${my_left}<small>/ ${budget}</small>", "accent": True,
         "bar": my_left / budget if budget else 0},
        {"k": "League avg left", "v": f"${sum(rivals) / len(rivals):.0f}" if rivals else "–",
         "d": f"You rank #{rank} of {len(left)} in budget"},
        {"k": "Richest rival", "v": f"${max(rivals)}" if rivals else "–",
         "d": "The most anyone can outbid you by"},
        {"k": "Season", "v": f"Wk {league['week']}<small>/ {league['regular_season_weeks']}</small>",
         "d": "Bids get bolder as FAAB loses value",
         "bar": min(1, (league["week"] - 1) / max(1, league["regular_season_weeks"]))},
    ])
else:
    t.tiles([
        {"k": "Waiver type", "v": "Priority", "d": "This league doesn't use FAAB"},
        {"k": "Season", "v": f"Wk {league['week']}"},
    ])

with st.spinner("Scoring the wire against your lineup…"):
    fas = free_agents(cfg, league)
    targets = waiver_targets(fas, teams[mid], league, mid, teams)

pos = st.segmented_control("Position", ["All", "QB", "RB", "WR", "TE", "K", "DST"],
                           default="All", label_visibility="collapsed", key="wv_pos")
shown = targets if targets.empty or pos in (None, "All") else targets[targets["pos"] == pos]

t.section("Recommended adds", f"{len(fas)} available players scanned")

TAG_TONE = {"Priority add": "volt", "Solid add": "info", "Stream": "dim", "Depth": "dim"}


def _card(i: int, p: pd.Series) -> str:
    hot = p["tag"] == "Priority add" or (i == 0 and p["tag"] == "Solid add")
    wk = f"{p['pos']}{int(p['weekly_rank'])}" if t.isnum(p.get("weekly_rank")) else "–"
    own = f"{p['pct_owned']:.0f}%" if t.isnum(p.get("pct_owned")) else "–"
    chg = p.get("own_change")
    own_d = (f" <span style='color:var(--good)'>+{chg:.1f}</span>" if t.isnum(chg) and chg >= 1
             else "")
    stats = (f"<span>ROS<b>{t.esc(p.get('ros_pos_rank') or '–')}</b></span>"
             f"<span>Overall<b>{t.fmt_int(p.get('ros_rank'))}</b></span>"
             f"<span>This week<b>{wk}</b></span>"
             f"<span>Proj<b>{t.fmt_pts(p.get('week_proj'))}</b></span>"
             f"<span>Rostered<b>{own}</b>{own_d}</span>")
    if p.get("on_waivers"):
        stats += "<span><b style='color:var(--warn);margin:0'>On waivers</b></span>"
    d = p["drop"]
    if d is not None and not (isinstance(d, float) and pd.isna(d)):
        swap = (f"<div class='da-swap'><span class='lab add'>ADD</span>"
                f"<span class='da-strong'>{t.esc(p['player'])}</span>"
                f"<span class='arrow'>/</span><span class='lab drop'>DROP</span>"
                f"<span>{t.esc(d['player'])}</span>"
                f"<span class='da-dim' style='font-size:12px'>{t.esc(d['pos'])} · "
                f"ROS {t.esc(d.get('ros_pos_rank') or '–')}</span></div>")
    else:
        swap = ("<div class='da-swap'><span class='lab add'>ADD</span>"
                "<span>You have an open roster spot, no drop needed</span></div>")
    why = "".join(f"<li>{t.esc(w)}</li>" for w in p["why"])
    if uses_faab:
        amt = f"${int(p['bid'])}"
        rng = (f"${int(p['bid_lo'])}–{int(p['bid_hi'])}" if p["bid_hi"] > p["bid_lo"]
               else "min bid")
        bid = (f"<div class='da-bid'><div class='lbl'>Bid</div><div class='amt'>{amt}</div>"
               f"<div class='rng'>{rng}</div></div>")
    else:
        bid = f"<div class='da-bid'><div class='rng'>{t.esc(p['claim'])}</div></div>"
    return (f"<div class='da-card{' hot' if hot else ''}'><div class='da-wv'>"
            f"<div class='da-rank'>{i + 1}</div>"
            f"<div style='min-width:0'><div class='top'>{t.who(p)}"
            f"{t.tag(p['tag'], TAG_TONE.get(p['tag'], 'dim'))}</div>"
            f"<div class='da-stats'>{stats}</div>{swap}<ul class='da-why'>{why}</ul></div>"
            f"{bid}</div></div>")


if shown.empty:
    best = fas[fas["ros_rank"].notna()].sort_values("ros_rank").head(6)
    if pos not in (None, "All"):
        best = fas[(fas["pos"] == pos) & fas["ros_rank"].notna()].sort_values("ros_rank").head(6)
    t.empty("<b>Nothing on the wire beats what you already have.</b> Your roster "
            "(including your bench depth) out-values every available player"
            + (f" at {pos}" if pos not in (None, "All") else "")
            + ", so the smart move is to hold your FAAB. The best names out there "
              "are below, in case an injury changes the math.")
    if not best.empty:
        rows = "".join(
            f"<tr><td>{t.who(p, small=True)}</td><td class='r'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
            f"<td class='r'>{t.fmt_pts(p.get('week_proj'))}</td></tr>"
            for _, p in best.iterrows())
        t.html(f"<div class='da-wrap' style='margin-top:10px'><table class='da-table'><thead>"
               f"<tr><th>Watch list</th><th class='r'>ROS</th><th class='r'>Proj</th></tr>"
               f"</thead><tbody>{rows}</tbody></table></div>")
else:
    t.html("".join(_card(i, p) for i, (_, p) in enumerate(shown.iterrows())))

# --- drop candidates, by how little they'd cost your lineup
t.section("Your cut list", "least lineup value lost first")
costs = drop_costs(teams[mid], league.get("lineup"))
if not costs.empty:
    worst = max(costs["drop_cost"].max(), 1)
    rows = []
    for _, p in costs.head(6).iterrows():
        pct = p["drop_cost"] / worst
        label = ("Free cut" if p["drop_cost"] < 0.5 else "Low cost" if pct < 0.25
                 else "Keep")
        tone = "good" if label == "Free cut" else "dim" if label == "Low cost" else "warn"
        rows.append(
            f"<tr><td>{t.who(p, small=True)}</td>"
            f"<td class='r'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
            f"<td class='r hide-m'>{t.fmt_pts(p.get('week_proj'))}</td>"
            f"<td class='r'>{t.tag(label, tone)}</td></tr>")
    t.html(f"<div class='da-wrap'><table class='da-table'><thead><tr><th>Player</th>"
           f"<th class='r'>ROS</th><th class='r hide-m'>Proj</th><th class='r'>Verdict</th>"
           f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")

with st.expander("How recommendations and bids are calculated"):
    st.markdown(
        "- **Tailored to your roster.** Each free agent is tried in every possible "
        "add/drop swap with your roster, using your league's real lineup slots. "
        "Rest-of-season value comes from FantasyPros consensus, and this week's points "
        "come from ESPN's projections in *your league's scoring*. The best swap is the "
        "one shown.\n"
        "- **Bids are priced to the competition.** A bid is a share of what the *rivals "
        "who would actually start him* have left, not a share of your own budget. It "
        "grows with how much he improves your lineup, gets bumped when he's trending on "
        "Sleeper or rising in ESPN adds, and never exceeds the richest interested rival's "
        "budget + $1.\n"
        "- **The season clock.** Unspent FAAB loses value every week, so bids scale up as "
        "the regular season winds down. Kickers and defenses are always $0–1 swaps.\n"
        "- **Always current.** The wire is cross-checked against every roster in your "
        "league, so a player who was just picked up never shows as available.")
