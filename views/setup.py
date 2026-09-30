"""League Setup: connect the app to your ESPN league."""

import streamlit as st

import theme as t
from espn_league import get_league, load_config, save_config

t.page_header("League Setup", ["Settings"],
              sub="Connect DraftAid to your ESPN league. Credentials stay on this app's "
                  "server (league_config.json, never committed to git) and are only ever "
                  "sent to ESPN.")

cfg = load_config()

with st.form("espn_connect", border=True):
    league_id = st.text_input("ESPN league ID", value=str(cfg.get("league_id", "")),
                              help="The number after leagueId= in your league's URL")
    year = st.number_input("Season", 2020, 2030, int(cfg.get("year") or 2026))
    st.caption("Private league? Paste both cookies (see instructions below). "
               "Public league? Leave them blank.")
    espn_s2 = st.text_input("espn_s2 cookie", value=cfg.get("espn_s2", ""),
                            type="password")
    swid = st.text_input("SWID cookie", value=cfg.get("swid", ""), type="password",
                         help="Looks like {ABCD1234-...}")
    submitted = st.form_submit_button("Save and test connection", type="primary",
                                      icon=":material/link:", width="stretch")

if submitted:
    if not league_id.strip():
        st.error("League ID is required.")
    else:
        new_cfg = {**cfg, "league_id": league_id.strip(), "year": int(year),
                   "espn_s2": espn_s2.strip(), "swid": swid.strip()}
        save_config(new_cfg)
        st.cache_data.clear()
        try:
            league = get_league(new_cfg)
            t.alert("good", "Connected",
                    f"<b>{t.esc(league['league_name'])}</b>: {len(league['teams'])} teams, "
                    f"NFL week {league['week']}"
                    + (f", ${league['faab_budget']} FAAB" if league["uses_faab"]
                       else ", waiver-priority league"))
        except Exception as e:
            t.alert("bad", "Failed", t.esc(e))

# --- pick my team (after a successful connect)
cfg = load_config()
if cfg.get("league_id"):
    try:
        league = get_league(cfg)
        team_opts = {x["name"]: x["team_id"] for x in league["teams"]}
        current = next((x["name"] for x in league["teams"]
                        if x["team_id"] == cfg.get("my_team_id")), None)
        t.section("Your team")
        pick = st.selectbox("Which team is yours?", list(team_opts.keys()),
                            index=(list(team_opts).index(current)
                                   if current in team_opts else None),
                            placeholder="Pick your team…", label_visibility="collapsed")
        if pick and team_opts[pick] != cfg.get("my_team_id"):
            cfg["my_team_id"] = team_opts[pick]
            save_config(cfg)
            t.alert("good", "Saved", f"You are <b>{t.esc(pick)}</b>.")
    except Exception:
        pass

t.section("Help")
with st.expander("How to find your league ID and cookies"):
    st.markdown("""
**League ID**: open your league on espn.com; the URL contains
`leagueId=XXXXXXX`. That number is your league ID.

**Private leagues need two cookies** (they let the app read your league as you,
nothing more):

*On a computer:* log in at espn.com → press F12 (DevTools) → **Application**
tab → **Cookies** → `https://www.espn.com` → copy the values of `espn_s2`
(very long) and `SWID` (like `{ABC...}`).

*On your phone:* easiest is to do the above once on any computer. The cookies
last for months. Paste them here and you're set.

**Heads-up on hosting:** on Streamlit Community Cloud the saved config file can
reset when the app restarts. For a permanent connection, add this to the app's
**Settings → Secrets** on share.streamlit.io:

```toml
[espn]
league_id = "1234567"
year = 2026
espn_s2 = "..."
swid = "{...}"
my_team_id = 8
```
""")
