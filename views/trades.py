"""Trade finder: surplus/deficit matching across the league."""

import streamlit as st

import theme as t
from season import player_value, positional_needs, trade_ideas
from season_ui import header, league_ctx, my_team_id, need_my_team

cfg, league, teams, names = league_ctx()
mid = my_team_id(cfg, league)
if mid is None:
    need_my_team()

header(league, "Trade Finder",
       "Deals that fill a real need on both rosters and look fair on two "
       "independent scales: FantasyCalc's real-trade market and expert "
       "rest-of-season consensus.")

steals_on = st.toggle("Include steals: offers tilted your way (they'll often "
                      "decline, but asking is free)", value=True)
ideas = trade_ideas(teams, mid, names, top_n=12 if steals_on else 8,
                    include_steals=steals_on)
needs_all = {tid: positional_needs(df) for tid, df in teams.items()}
positions = ("QB", "RB", "WR", "TE")
avg = {p: sum(n[p]["starter_avg"] for n in needs_all.values()) / len(needs_all)
       for p in positions}

t.section("Trade ideas", f"{len(ideas)} found")
if not ideas:
    weak = [p for p in positions if needs_all[mid][p]["starter_avg"] - avg[p] > 5]
    if not weak:
        t.empty("<b>No fair-trade fits, and here's why:</b> your starters are at or "
                "above league average at every position, so there's no hole worth "
                "paying market price to fix. That changes when injuries or busts drop "
                "a position group below average, or when an opponent gets desperate.")
    else:
        t.empty(f"<b>No fits right now.</b> You're thin at {', '.join(weak)}, but no "
                "opponent is both deep there <i>and</i> thin where you have spare "
                "depth, at a price both value sources call fair. Check back after "
                "injuries shake up rosters, or hunt manually with the heatmap below.")


def _pl(p) -> str:
    return (f"<div class='pl'>{t.who(p, small=True)}"
            f"<span class='da-num da-dim' style='margin-left:auto;font-size:12px'>"
            f"{t.esc(p.get('ros_pos_rank') or '')}</span></div>")


cards = []
for i in ideas:
    ratio = i["ratio"]
    if i["kind"] == "steal":
        verdict = t.tag("Tilted your way", "volt")
    elif i["kind"] == "1-for-1":
        verdict = (t.tag("Dead even", "info") if 0.97 <= ratio <= 1.07
                   else t.tag("Slight win for you", "good") if ratio < 0.97
                   else t.tag("You pay a little extra", "warn"))
    else:
        verdict = t.tag("Standard 2-for-1 premium", "warn")
    give_v = sum(player_value(g) for g in i["give"])
    get_v = player_value(i["get"]) or 1
    pos = min(100, max(0, (give_v / get_v - 0.5) * 100))
    cards.append(
        f"<div class='da-card{' hot' if i['kind'] == 'steal' else ''}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;"
        f"flex-wrap:wrap'><div><span class='da-eyebrow' style='display:inline'>"
        f"{t.esc(i['kind'])} with</span> <span class='da-strong' style='font-size:16px'>"
        f"{t.esc(i['team'])}</span></div>{verdict}</div>"
        f"<div class='da-tr'><div class='col get'><div class='h'>YOU GET</div>{_pl(i['get'])}</div>"
        f"<div class='col give'><div class='h'>YOU GIVE</div>"
        f"{''.join(_pl(g) for g in i['give'])}</div></div>"
        f"<div class='da-meter'><div class='track'><i style='left:calc({pos:.0f}% - 1px)'></i></div>"
        f"<div class='lg'><span>WIN FOR YOU</span><span>EVEN</span><span>OVERPAY</span></div></div>"
        f"<ul class='da-why'><li><b style='color:var(--text)'>For you:</b> {t.esc(i['why_me'])}</li>"
        f"<li><b style='color:var(--text)'>For them:</b> {t.esc(i['why_them'])}</li></ul></div>")
if cards:
    t.html("".join(cards))

# --- league positional strength heatmap
t.section("Positional strength", "avg rest-of-season rank of each team's starters · "
          "green = stronger than league average")


def _cell(v: float, league_avg: float) -> str:
    diff = (league_avg - v) / max(league_avg, 1)  # + = stronger than average
    a = min(0.55, abs(diff) * 0.9)
    rgb = "47,214,163" if diff > 0 else "255,93,93"
    return (f"<td class='hc' style='background:rgba({rgb},{a:.2f})'>{v:.0f}</td>")


rows = []
for tid in sorted(teams, key=lambda k: k != mid):
    n = needs_all[tid]
    rows.append(f"<tr class='{'me' if tid == mid else ''}'>"
                f"<td><span class='da-strong'>{t.esc(names[tid])}</span></td>"
                + "".join(_cell(n[p]["starter_avg"], avg[p]) for p in positions)
                + f"<td class='c da-dim'>{sum(x['depth'] for x in n.values())}</td></tr>")
t.html("<div class='da-wrap'><table class='da-table da-heat'><thead><tr><th>Team</th>"
       + "".join(f"<th class='c'>{p}</th>" for p in positions)
       + "<th class='c' title='Startable players beyond the lineup'>Depth</th></tr></thead>"
       f"<tbody>{''.join(rows)}</tbody></table></div>")
