"""One place for outbound HTTP defaults."""

import requests

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def get(url: str, *, timeout: int = 30, headers: dict | None = None, **kwargs) -> requests.Response:
    """GET with a browser user agent and a timeout; raises on HTTP errors."""
    resp = requests.get(url, headers={**UA, **(headers or {})}, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp
