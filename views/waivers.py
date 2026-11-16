"""Waiver wire: free agents ranked by what they do for YOUR lineup, each paired
with the right drop and a FAAB bid sized to the rivals who'd compete for him."""

import pandas as pd
import streamlit as st

from core.lineup import drop_costs
from core.waivers import handcuffs, playoff_stashes, waiver_targets
from ui import theme as t
from ui.blocks import stash_table
from ui.context import header, load

ctx = load()
league, mid, roster = ctx.league, ctx.my_id, ctx.roster
uses_faab = league["uses_faab"]
budget = league["faab_budget"]
left = {x["team_id"]: max(0, budget - x["faab_spent"]) for x in league["teams"]}
my_left = left[mid]
rivals = [v for k, v in left.items() if k != mid]

header(ctx, "Waiver Wire",
       "Every free agent scored against your actual lineup, with the drop that "
       "makes room and a bid priced to beat the rivals who'd want him.")

if uses_faab:
    rank = sorted(left.values(), reverse=True).index(my_left) + 1
    t.tiles([
        {"k": "Your FAAB", "v": f"${my_left}<small>of ${budget}</small>", "lead": True,
         "bar": my_left / budget if budget else 0},
        {"k": "League average left", "v": f"${sum(rivals) / len(rivals):.0f}" if rivals else "–",
         "d": f"You rank {t.ordinal(rank)} of {len(left)} in budget"},
        {"k": "Richest rival", "v": f"${max(rivals)}" if rivals else "–",
         "d": "The most anyone can outbid you by"},
        {"k": "Season", "v": f"Week {league['week']}<small>of {league['regular_season_weeks']}</small>",
         "d": "Bids get bolder as FAAB loses value",
         "bar": min(1, (league["week"] - 1) / max(1, league["regular_season_weeks"]))},
    ])
else:
    t.tiles([
        {"k": "Waiver type", "v": "Priority", "d": "This league doesn't use FAAB"},
        {"k": "Season", "v": f"Week {league['week']}"},
    ])

with st.spinner("Scoring the wire against your lineup…"):
    fas = ctx.free_agents()
    targets = waiver_targets(fas, roster, league, mid, ctx.teams)

pos = st.segmented_control("Position", ["All", "QB", "RB", "WR", "TE", "K", "DST"],
                           default="All", label_visibility="collapsed", key="wv_pos")
shown = targets if targets.empty or pos in (None, "All") else targets[targets["pos"] == pos]

t.section("Recommended adds", f"{len(fas)} available players scanned")

TAG_TONE = {"Priority add": "accent", "Solid add": "info"}


def _card(i: int, p: pd.Series) -> str:
    lead = p["tag"] == "Priority add" or (i == 0 and p["tag"] == "Solid add")
    wk = f"{p['pos']}{int(p['weekly_rank'])}" if t.isnum(p.get("weekly_rank")) else "–"
    own = f"{p['pct_owned']:.0f}%" if t.isnum(p.get("pct_owned")) else "–"
    chg = p.get("own_change")
    own_d = f" <span class='da-up'>{chg:.1f}</span>" if t.isnum(chg) and chg >= 1 else ""
    stats = (f"<span>Rest of season<b>{t.esc(p.get('ros_pos_rank') or '–')}</b></span>"
             f"<span>This week<b>{wk}</b></span>"
             f"<span>Projected<b>{t.fmt_pts(p.get('week_proj'))}</b></span>")
    if t.isnum(p.get("po_rank")):
        stats += f"<span>Playoffs<b>{p['pos']}{int(p['po_rank'])}</b></span>"
    stats += f"<span>Rostered<b>{own}</b>{own_d}</span>"
    tags = t.tag(p["tag"], TAG_TONE.get(p["tag"], ""))
    if p.get("on_waivers"):
        tags += " " + t.tag("On waivers", "warn")
    d = p["drop"]
    if d is not None and not (isinstance(d, float) and pd.isna(d)):
        swap = (f"<div class='da-swap'><span class='lab add'>ADD</span>"
                f"<span class='da-strong'>{t.esc(p['player'])}</span>"
                f"<span class='arrow'>/</span><span class='lab drop'>DROP</span>"
                f"<span>{t.esc(d['player'])}</span>"
                f"<span class='da-dim' style='font-size:12px'>{t.esc(d['pos'])} · "
                f"rest of season {t.esc(d.get('ros_pos_rank') or '–')}</span></div>")
    else:
        swap = ("<div class='da-swap'><span class='lab add'>ADD</span>"
                "<span>You have an open roster spot, no drop needed</span></div>")
    why = "".join(f"<li>{t.esc(w)}</li>" for w in p["why"])
    if uses_faab:
        rng = (f"${int(p['bid_lo'])}–{int(p['bid_hi'])}" if p["bid_hi"] > p["bid_lo"]
               else "minimum bid")
        bid = (f"<div class='da-bid'><div class='lbl'>Bid</div>"
               f"<div class='amt'>${int(p['bid'])}</div><div class='rng'>{rng}</div></div>")
    else:
        bid = f"<div class='da-bid'><div class='rng'>{t.esc(p['claim'])}</div></div>"
    return (f"<div class='da-card{' lead' if lead else ''}'><div class='da-wv'>"
            f"<div class='da-rank'>{i + 1}</div>"
            f"<div class='body' style='min-width:0'><div class='top'>{t.who(p)}{tags}</div>"
            f"<div class='da-stats'>{stats}</div>{swap}<ul class='da-why'>{why}</ul></div>"
            f"{bid}</div></div>")


if shown.empty:
    best = fas[fas["ros_rank"].notna()] if not fas.empty else fas
    if pos not in (None, "All") and not best.empty:
        best = best[best["pos"] == pos]
    best = best.sort_values("ros_rank").head(6) if not best.empty else best
    t.empty("<b>Nothing on the wire beats what you already have.</b> Your roster "
            "(including your bench depth) out-values every available player"
            + (f" at {pos}" if pos not in (None, "All") else "")
            + ", so the smart move is to hold your FAAB. The best names out there "
              "are below, in case an injury changes the math.")
    if not best.empty:
        rows = [f"<tr><td>{t.who(p, small=True)}</td>"
                f"<td class='r'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
                f"<td class='r'>{t.fmt_pts(p.get('week_proj'))}</td></tr>"
                for _, p in best.iterrows()]
        t.html("<div style='margin-top:10px'>"
               + t.table([("Watch list", ""), ("Rest of season", "r"), ("Projected", "r")], rows)
               + "</div>")
else:
    t.html("".join(_card(i, p) for i, (_, p) in enumerate(shown.iterrows())))

# --- insurance: the direct backups to the backs you rely on
cuffs = handcuffs(roster, fas)
if cuffs:
    rows = []
    for h in cuffs:
        b, s = h["backup"], h["starter"]
        depth = f"No. {int(b['depth_order'])} on the {t.esc(b['team'])} depth chart"
        own = f"{b['pct_owned']:.0f}%" if t.isnum(b.get("pct_owned")) else "–"
        rows.append(f"<tr><td>{t.who(b, small=True)}</td>"
                    f"<td>Backs up <span class='da-strong'>{t.esc(s['player'])}</span>"
                    f"<div class='da-dim' style='font-size:12px'>{depth}</div></td>"
                    f"<td class='r hide-m'>{own}</td></tr>")
    t.section("Insurance for your backs",
              "If your starter goes down, his backup is the hottest add in the league. "
              "He's free today.")
    t.html(t.table([("Available backup", ""), ("Why", ""), ("Rostered", "r hide-m")], rows))

# --- playoff stashes
po_weeks = [w for w in ctx.src.playoff_weeks if w in ctx.src.weeks]
stashes = playoff_stashes(fas, roster, ctx.lineup, po_weeks) if po_weeks else pd.DataFrame()
if not stashes.empty:
    t.section(f"Playoff stashes, weeks {po_weeks[0]}–{po_weeks[-1]}",
              "Free agents who would raise your projected lineups when the title is decided")
    t.html(stash_table(stashes))

# --- drop candidates, by how little they'd cost your lineup
t.section("Your cut list", "least lineup value lost first")
costs = drop_costs(roster, league.get("lineup"))
if not costs.empty:
    worst = max(costs["drop_cost"].max(), 1)
    rows = []
    for _, p in costs.head(6).iterrows():
        pct = p["drop_cost"] / worst
        label = ("Free cut" if p["drop_cost"] < 0.5 else "Low cost" if pct < 0.25
                 else "Keep")
        tone = "good" if label == "Free cut" else "" if label == "Low cost" else "warn"
        rows.append(
            f"<tr><td>{t.who(p, small=True)}</td>"
            f"<td class='r'>{t.esc(p.get('ros_pos_rank') or '–')}</td>"
            f"<td class='r hide-m'>{t.fmt_pts(p.get('week_proj'))}</td>"
            f"<td class='r'>{t.tag(label, tone)}</td></tr>")
    t.html(t.table([("Player", ""), ("Rest of season", "r"), ("Projected", "r hide-m"),
                    ("Verdict", "r")], rows))

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
        "- **Playoff view.** \"Playoffs\" is a player's positional rank by Sleeper's "
        "projections for your league's playoff weeks. Stashes are measured by the points "
        "they'd add to your best lineup in those weeks; backups are found from NFL depth "
        "charts.\n"
        "- **Always current.** The wire is cross-checked against every roster in your "
        "league, so a player who was just picked up never shows as available.")
