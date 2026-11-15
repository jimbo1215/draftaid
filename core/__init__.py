"""DraftAid core: data sources and fantasy logic, independent of the UI.

Data sources (fantasypros, sleeper, market, adp, news, espn) fetch and cache.
Logic modules (lineup, waivers, startsit, trades, outlook, draft) are pure
functions over DataFrames and dicts, so they can be unit-tested offline.
"""
