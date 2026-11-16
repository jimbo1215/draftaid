"""Composite blocks that more than one page shows."""

import pandas as pd

from ui import theme as t


def stash_table(stashes: pd.DataFrame) -> str:
    """Free agents who would raise the user's projected playoff lineups."""
    rows = [f"<tr><td>{t.who(p, small=True)}</td>"
            f"<td class='r'>{t.fmt_pts(p.get('po_pts'))}</td>"
            f"<td class='r'><span class='da-strong'>+{p['po_gain']:.1f}</span></td>"
            f"<td class='r hide-m da-dim'>{t.fmt_int(p.get('pct_owned'))}%</td></tr>"
            for _, p in stashes.iterrows()]
    return t.table([("Player", ""), (("Playoff points", "Points"), "r"),
                    (("Adds to your lineups", "Adds"), "r"), ("Rostered", "r hide-m")], rows)
