# 🏈 DraftAid

A live draft assistant for 12-team, 1-QB, full-PPR snake drafts. Built with Streamlit.
Main app file: `draftaid.py`.

## What it does

- **Blended live rankings** — pulls FantasyPros expert-consensus PPR rankings (with their
  tiers), real-draft ADP from FantasyFootballCalculator (12-team PPR), and Sleeper
  injury/trending data every time you hit **🔄 Refresh live rankings**. Optionally blend in
  Underdog's rankings via CSV upload. Source weights are adjustable in the sidebar.
- **One-tap draft tracking** — every board row has its own ℹ️ / 🚫 / ✅ buttons: one tap
  opens the **player card** (headshot, team logo, age/college, expert rank range, ADP,
  live news headlines, and big *Draft to MY team* / *Taken by …* buttons), one tap logs a
  pick to the on-clock team, one tap drafts him to you. Type-ahead quick entry, one-tap
  draft straight from the suggestions, undo, and full pick-log editing included. A "Show
  drafted" toggle keeps taken players visible (struck through, with who took them).
- **League setup** — an editable table of team names and pick order (change it each year
  in seconds), with a "which team is you?" selector that derives your draft slot.
- **Snake-draft awareness** — always shows the current round/pick, when your next turn is,
  and each player's probability of *lasting until your next turn* (from ADP spread).
- **Pick suggestions** — top-5 recommendations for your next pick, scored on board value,
  roster need, tier scarcity, ADP value, and gone-by-your-next-turn risk, each with a
  plain-English reason.
- **Tier view** — tier-colored board and a by-position tier sheet so you can see cliffs
  coming before your leaguemates do.
- **League tracking** — every pick is attributed to the right team by snake order (with your
  leaguemates' names), so the 🏟 League tab shows each team's roster and a who-has-what grid
  to spot positional runs before they start.
- **Practice mode** — simulate the other teams' picks to rehearse your draft plan.
- **Crash-proof** — every pick autosaves to disk; if the page refreshes mid-draft you get a
  one-click *Resume saved draft*. You can also export/import the draft state as JSON.

## Run locally

```
pip install -r requirements.txt
streamlit run draftaid.py
```

## Deploy to Streamlit Cloud

This repo backs the existing **draftaid** Streamlit Cloud app: pushing to `main` redeploys
it automatically (the main file is still `draftaid.py`, so no app-settings change is
needed). No API keys or secrets required; all data sources are free and public.

## Draft-day workflow

1. Before the draft: sidebar → **League settings** → set your draft slot (and roster spots
   if they differ from the default 1QB/2RB/2WR/1TE/1FLEX/1K/1DST).
2. Optional: on [Underdog](https://underdogfantasy.com/rankings), download your rankings CSV
   and upload it under **Ranking sources & weights**.
3. Hit **🔄 Refresh live rankings** right before the draft starts.
4. As each pick is announced, type a few letters of the name and tap **🚫 Taken** — or
   **✅ My pick** when it's yours. That's the whole job; the board, suggestions, tiers, and
   your-turn countdown update automatically.
5. Missed some picks? Open **Fix pick counter** and set the true overall pick number.
   Logged the wrong player? Remove any pick from the **📜 Pick Log** tab.

## Data sources

| Source | What it provides | Refresh |
| --- | --- | --- |
| FantasyPros | Expert-consensus PPR ranks, tiers, byes, rank spread | 15 min cache, or 🔄 button |
| FantasyFootballCalculator | ADP + stdev from real 12-team PPR mock drafts | 15 min cache, or 🔄 button |
| Sleeper | Injury status, 24h trending adds | 15 min (trending) / 6 h (players) |
| Underdog CSV (optional) | Underdog rankings/ADP | On upload |

Sleeper and Underdog don't publish free ADP APIs, which is why Underdog comes in as a CSV
and Sleeper contributes injury/trending data rather than ranks.
