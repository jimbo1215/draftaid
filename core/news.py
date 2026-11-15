"""Player headlines from the Google News RSS feed."""

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
import streamlit as st

from core import http


def _age(pub: str | None) -> str:
    if not pub:
        return ""
    try:
        delta = pd.Timestamp.now(tz="UTC") - parsedate_to_datetime(pub)
    except (ValueError, TypeError):
        return ""
    return f"{delta.days}d ago" if delta.days >= 1 else f"{max(1, delta.seconds // 3600)}h ago"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_player_news(player_name: str, limit: int = 3) -> list[dict]:
    """Latest headlines as [{title, link, source, when}]; [] on any failure."""
    try:
        url = ("https://news.google.com/rss/search?q="
               + requests.utils.quote(f"{player_name} fantasy football")
               + "&hl=en-US&gl=US&ceid=US:en")
        root = ET.fromstring(http.get(url, timeout=8).content)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title, source = item.findtext("title") or "", ""
        if " - " in title:  # Google News appends the outlet after a dash
            title, source = title.rsplit(" - ", 1)
        items.append({"title": title, "link": item.findtext("link") or "",
                      "source": source, "when": _age(item.findtext("pubDate"))})
        if len(items) >= limit:
            break
    return items
