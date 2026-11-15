"""Build a demo league from live rankings, shaped like raw ESPN payloads.

    python tools/make_fixture.py            # writes .fixture/league.json + fas.json
    DRAFTAID_FIXTURE=.fixture streamlit run draftaid.py

Twelve teams snake-draft the current rest-of-season board, play a simulated
season up to this week, and leave everyone else on the wire. It exists so the
app can be run, demoed, and visually checked without ESPN credentials; the
players, ranks, and projections are real, the league is not.
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import fantasypros, http, sleeper  # noqa: E402
from core.espn import PRO_TEAMS  # noqa: E402
from core.names import normalize_name, normalize_team  # noqa: E402

TEAM_NAMES = ["Buddy", "Giuseppe", "Scab", "Randino", "Nuzzo", "Geiger",
              "Bullock", "Jimmy", "Claire", "Dro", "Lamart", "Logik"]
MY_TEAM = 8
POS_ID = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DST": 16}
SLOT_ID = {"QB": 0, "RB": 2, "WR": 4, "TE": 6, "FLEX": 23, "DST": 16, "K": 17,
           "BN": 20, "IR": 21}
PRO_ID = {abbr: pid for pid, abbr in PRO_TEAMS.items()}
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
ROUNDS, REG_WEEKS, BUDGET = 14, 14, 100
CAPS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}


def player_pool(week: int, season: int) -> list[dict]:
    """Skill players from the ROS board plus kickers and defenses from this
    week's projections, each with ids and a current-week projection."""
    ros = fantasypros.fetch_ros()
    raw = http.get(f"{sleeper.API}/players/nfl", timeout=120).json()
    proj = sleeper.fetch_week_projections(season, week)
    by_key, kickers = {}, []
    for sid, p in raw.items():
        pos = p.get("position")
        if pos in ("QB", "RB", "WR", "TE", "K") and (p.get("team") or p.get("active")):
            key = f"{normalize_name(p.get('full_name') or '')}|{pos}"
            if key not in by_key or p.get("team"):
                by_key[key] = (sid, p)
            if pos == "K" and sid in proj:
                kickers.append((sid, p))

    pool, fake_id = [], 9_000_000
    for _, r in ros.iterrows():
        if r["pos"] not in ("QB", "RB", "WR", "TE"):
            continue
        sid, sp = by_key.get(r["key"], (None, {}))
        try:
            espn_id = int(sp.get("espn_id"))
        except (TypeError, ValueError):
            fake_id += 1
            espn_id = fake_id
        pts = proj.get(sid, (None, None, None))[1] if sid else None
        pool.append({"id": espn_id, "name": r["player"], "pos": r["pos"], "team": r["team"],
                     "rank": int(r["ros_rank"]), "proj": pts,
                     "injury": (sp.get("injury_status") or "ACTIVE").upper()})
    for i, (sid, sp) in enumerate(sorted(kickers, key=lambda k: -proj[k[0]][1])):
        try:
            espn_id = int(sp.get("espn_id"))
        except (TypeError, ValueError):
            fake_id += 1
            espn_id = fake_id
        pool.append({"id": espn_id, "name": sp.get("full_name", "Kicker"), "pos": "K",
                     "team": normalize_team(sp.get("team")), "rank": 400 + i,
                     "proj": proj[sid][1], "injury": "ACTIVE"})
    defenses = sorted(((sid, v) for sid, v in proj.items() if v[0] == "DST"),
                      key=lambda kv: -kv[1][1])
    for i, (sid, (_, pts, _)) in enumerate(defenses):
        team = normalize_team(sid)
        pool.append({"id": -16000 - PRO_ID.get(team, 0), "name": f"{team} D/ST", "pos": "DST",
                     "team": team, "rank": 450 + i, "proj": pts, "injury": "ACTIVE"})
    for p in pool:
        if p["proj"] is None:   # not in this week's feed: on bye or ruled out
            p["proj"] = 0.0
    return pool


def draft(pool: list[dict], rng: random.Random) -> dict[int, list[dict]]:
    """Twelve-team snake draft off a noisy board; K and DST go last."""
    board = sorted((p for p in pool if p["pos"] in CAPS),
                   key=lambda p: p["rank"] + rng.gauss(0, 4 + p["rank"] * 0.10))
    rosters = {tid: [] for tid in range(1, 13)}
    for rnd in range(ROUNDS - 2):
        order = range(1, 13) if rnd % 2 == 0 else range(12, 0, -1)
        for tid in order:
            have = {pos: sum(1 for p in rosters[tid] if p["pos"] == pos) for pos in CAPS}
            late = rnd >= ROUNDS - 5
            pick = next(p for p in board
                        if have[p["pos"]] < CAPS[p["pos"]]
                        and not (late and have["QB"] == 0 and p["pos"] != "QB")
                        and not (late and have["QB"] > 0 and have["TE"] == 0 and p["pos"] != "TE"))
            board.remove(pick)
            rosters[tid].append(pick)
    for pos in ("K", "DST"):
        best = sorted((p for p in pool if p["pos"] == pos), key=lambda p: p["rank"])
        for tid in rng.sample(range(1, 13), 12):
            rosters[tid].append(best.pop(0))
    return rosters


def set_lineup(roster: list[dict]) -> None:
    """Start the best player at each slot by rank; everyone else to the bench."""
    left = sorted(roster, key=lambda p: p["rank"])
    for p in roster:
        p["slot"] = "BN"
    for slot, n in LINEUP.items():
        eligible = ("RB", "WR", "TE") if slot == "FLEX" else (slot,)
        for p in [q for q in left if q["pos"] in eligible][:n]:
            p["slot"] = slot
            left.remove(p)


def entry(p: dict, week: int, rng: random.Random) -> dict:
    own = max(0.2, min(99.9, 100 - p["rank"] * 0.33 + rng.uniform(-4, 4)))
    return {"lineupSlotId": SLOT_ID[p.get("slot", "BN")], "playerPoolEntry": {"player": {
        "id": p["id"], "fullName": p["name"], "defaultPositionId": POS_ID[p["pos"]],
        "proTeamId": PRO_ID.get(p["team"], 0), "injuryStatus": p["injury"],
        "ownership": {"percentOwned": round(own, 1),
                      "percentChange": round(rng.uniform(-3, 9) if p["rank"] > 150 else
                                             rng.uniform(-1, 1), 1)},
        "stats": [{"scoringPeriodId": week, "statSourceId": 1, "appliedTotal": p["proj"]},
                  {"scoringPeriodId": 0, "statSourceId": 0,
                   "appliedTotal": round(p["proj"] * (week - 1) * rng.uniform(0.8, 1.1), 1)}],
    }}}


def schedule_and_records(rosters: dict, week: int, rng: random.Random):
    strength = {tid: sum(p["proj"] for p in r if p["slot"] != "BN") for tid, r in rosters.items()}
    ids = list(range(1, 13))
    rec = {tid: {"wins": 0, "losses": 0, "ties": 0, "pointsFor": 0.0, "pointsAgainst": 0.0}
           for tid in ids}
    sched = []
    for wk in range(1, REG_WEEKS + 1):
        k = (wk - 1) % 11
        rot = [ids[0]] + ids[1:][k:] + ids[1:][:k]
        for j in range(6):
            home, away = rot[j], rot[11 - j]
            m = {"matchupPeriodId": wk, "playoffTierType": "NONE", "winner": "UNDECIDED",
                 "home": {"teamId": home, "totalPoints": 0.0},
                 "away": {"teamId": away, "totalPoints": 0.0}}
            if wk < week:
                hp = round(rng.gauss(strength[home] * 0.97, 22), 2)
                ap = round(rng.gauss(strength[away] * 0.97, 22), 2)
                m["home"]["totalPoints"], m["away"]["totalPoints"] = hp, ap
                m["winner"] = "HOME" if hp > ap else "AWAY"
                for tid, mine, theirs in ((home, hp, ap), (away, ap, hp)):
                    rec[tid]["wins" if mine > theirs else "losses"] += 1
                    rec[tid]["pointsFor"] += mine
                    rec[tid]["pointsAgainst"] += theirs
            elif wk == week:  # Sunday afternoon: some players have played
                for side, tid in (("home", home), ("away", away)):
                    done = rng.uniform(0.25, 0.6)
                    live = round(strength[tid] * done * rng.uniform(0.8, 1.25), 2)
                    m[side]["totalPointsLive"] = live
                    m[side]["totalProjectedPointsLive"] = round(
                        live + strength[tid] * (1 - done), 2)
            sched.append(m)
    return sched, rec


def build(seed: int = 2026) -> tuple[dict, list]:
    state = sleeper.fetch_state()
    week, season = max(2, min(state["week"] or 10, REG_WEEKS)), state["season"] or 2026
    pool = player_pool(week, season)
    # Try seeds until the demo's own team sits on the playoff bubble, where
    # every page has something to say.
    for attempt in range(seed, seed + 200):
        rng = random.Random(attempt)
        rosters = draft([dict(p) for p in pool], rng)
        for roster in rosters.values():
            set_lineup(roster)
        sched, rec = schedule_and_records(rosters, week, rng)
        standing = sorted(rec, key=lambda t: (-rec[t]["wins"], -rec[t]["pointsFor"]))
        if 4 <= standing.index(MY_TEAM) <= 7:
            break
    # One rival carries an injured starter on IR, like a real league would.
    stash = next(p for p in rosters[3] if p["slot"] == "BN")
    stash["slot"], stash["injury"] = "IR", "INJURY_RESERVE"
    deadline = datetime(season, 11, 25, 17, 0, tzinfo=timezone.utc)
    league = {
        "scoringPeriodId": week,
        "status": {"currentMatchupPeriod": week, "finalScoringPeriod": 17},
        "settings": {
            "name": "Demo League",
            "acquisitionSettings": {"acquisitionBudget": BUDGET, "minimumBid": 0,
                                    "isUsingAcquisitionBudget": True},
            "rosterSettings": {"lineupSlotCounts": {
                **{str(SLOT_ID[s]): n for s, n in LINEUP.items()}, "20": 5, "21": 1}},
            "scheduleSettings": {"matchupPeriodCount": REG_WEEKS, "playoffTeamCount": 6,
                                 "playoffMatchupPeriodLength": 1,
                                 "playoffSeedingRule": "TOTAL_POINTS_SCORED"},
            "tradeSettings": {"deadlineDate": int(deadline.timestamp() * 1000)},
        },
        "teams": [{
            "id": tid, "name": TEAM_NAMES[tid - 1], "abbrev": TEAM_NAMES[tid - 1][:4].upper(),
            "owners": ["{DEMO-SWID}"] if tid == MY_TEAM else [f"{{OWNER-{tid}}}"],
            "divisionId": 0, "playoffSeed": standing.index(tid) + 1,
            "record": {"overall": {**rec[tid], "pointsFor": round(rec[tid]["pointsFor"], 2),
                                   "pointsAgainst": round(rec[tid]["pointsAgainst"], 2)}},
            "transactionCounter": {"acquisitionBudgetSpent": rng.randint(4, 82),
                                   "acquisitions": rng.randint(3, 24)},
            "roster": {"entries": [entry(p, week, rng) for p in rosters[tid]]},
        } for tid in range(1, 13)],
        "schedule": sched,
    }
    taken = {p["id"] for r in rosters.values() for p in r}
    wire = sorted((p for p in pool if p["id"] not in taken), key=lambda p: p["rank"])[:260]
    fas = [{"id": p["id"], "onTeamId": 0,
            "status": "WAIVERS" if rng.random() < 0.12 else "FREEAGENT",
            "player": entry(p, week, rng)["playerPoolEntry"]["player"]} for p in wire]
    return league, fas


if __name__ == "__main__":
    out = ROOT / ".fixture"
    out.mkdir(exist_ok=True)
    league, fas = build()
    (out / "league.json").write_text(json.dumps(league), encoding="utf-8")
    (out / "fas.json").write_text(json.dumps(fas), encoding="utf-8")
    print(f"Wrote {out}: week {league['scoringPeriodId']}, "
          f"{len(league['teams'])} teams, {len(fas)} free agents")
