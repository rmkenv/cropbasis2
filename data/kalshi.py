"""
api/data/kalshi.py
Kalshi yield prediction market signal.
"""

import logging
import os
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

WASDE_YIELDS = {"Corn": 179.3, "Soybeans": 51.7, "Wheat": 51.2}
YIELD_STD    = {"Corn": 12.0,  "Soybeans": 4.5,  "Wheat": 6.0}

KALSHI_SEARCH = {
    "Corn":     ["corn yield", "USDA corn", "corn production"],
    "Soybeans": ["soybean yield", "USDA soybean"],
    "Wheat":    ["wheat yield", "USDA wheat"],
}


def fetch_kalshi_yield_signal(crop: str) -> dict:
    api_key  = os.environ.get("KALSHI_API_KEY", "")
    headers  = {
        "accept":        "application/json",
        "User-Agent":    "CropBasis/2.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    wasde = WASDE_YIELDS.get(crop, 0.0)
    std   = YIELD_STD.get(crop, 1.0)

    markets   = _search_markets(crop, headers)
    implied, summaries = _extract_implied(markets, wasde)
    pmd = abs(implied - wasde) / std if implied is not None else 0.0

    return {
        "kalshi_implied_yield": implied,
        "wasde_yield":          wasde,
        "pmd_zscore":           round(pmd, 3),
        "n_markets":            len(markets),
        "market_summaries":     summaries,
    }


def _search_markets(crop: str, headers: dict) -> list[dict]:
    found = []
    for term in KALSHI_SEARCH.get(crop, []):
        try:
            r = requests.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": 20, "status": "open", "search": term},
                headers=headers, timeout=15,
            )
            if r.status_code == 200:
                for m in r.json().get("markets", []):
                    title = m.get("title", "").lower()
                    if any(k in title for k in ("yield", "production", "usda", "bushel")):
                        if m not in found:
                            found.append(m)
        except Exception as exc:
            log.warning("Kalshi search '%s': %s", term, exc)
    return found


def _extract_implied(markets: list[dict], wasde: float):
    summaries    = []
    w_sum = w_tot = 0.0

    for m in markets:
        title   = m.get("title", "")
        yes_bid = m.get("yes_bid", 0) / 100
        yes_ask = m.get("yes_ask", 0) / 100
        mid     = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else None
        thresh  = _parse_threshold(title, wasde)

        summaries.append({
            "title":      title,
            "yes_mid":    round(mid * 100, 1) if mid else None,
            "threshold":  thresh,
            "ticker":     m.get("ticker", ""),
            "close_time": m.get("close_time", ""),
        })
        if mid and thresh:
            w_sum += mid * thresh
            w_tot += mid

    implied = round(w_sum / w_tot, 2) if w_tot > 0.1 else None
    return implied, summaries


def _parse_threshold(title: str, fallback: float) -> Optional[float]:
    for pat in [
        r"(?:above|over|>)\s*([\d]+\.?[\d]*)\s*bu",
        r"([\d]{2,3}\.?[\d]*)\s*bu",
    ]:
        m = re.search(pat, title, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def pmd_to_component(pmd_zscore: float) -> float:
    if pmd_zscore < 0.5:
        return 0.0
    return float(min(pmd_zscore / 3.0, 1.0))
