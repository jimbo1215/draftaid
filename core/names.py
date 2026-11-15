"""Name and team normalization, so one player joins across every source."""

import re

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
FLEX_POSITIONS = ("RB", "WR", "TE")

# Team-code variants across sources, normalized to FantasyPros codes.
TEAM_ALIASES = {
    "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU", "LA": "LAR", "SD": "LAC", "OAK": "LV", "STL": "LAR",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name) -> str:
    """Lowercase alpha-only key with generational suffixes stripped
    ("Marvin Harrison Jr." -> "marvinharrison")."""
    if not isinstance(name, str):
        return ""
    tokens = re.sub(r"[^a-z\s]", "", name.lower()).split()
    return "".join(t for t in tokens if t not in _SUFFIXES)


def normalize_team(team) -> str:
    team = (team or "").upper().strip()
    return TEAM_ALIASES.get(team, team)


def player_key(name, pos: str, team) -> str:
    """Cross-source join key. Defenses are named inconsistently, so they key
    by team; everyone else keys by normalized name + position."""
    if pos == "DST":
        return f"dst{normalize_team(team).lower()}"
    return f"{normalize_name(name)}|{pos}"
