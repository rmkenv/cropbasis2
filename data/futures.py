"""
data/futures.py
CME futures via yfinance and USDA NASS QuickStats cash prices.

NASS data note:
  "PRICE RECEIVED" weekly series only exists Oct-Dec (post-harvest).
  "PRICES, CASH, BIDS" is the year-round weekly elevator cash price series.
  We try PRICES,CASH,BIDS first; fall back to PRICE RECEIVED for off-season.
  Both walk back up to 2 years if the current year has no data yet.
"""

import datetime as _dt
import logging
import os
import warnings
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

FUTURES_TICKERS = {"Corn": "ZC=F", "Soybeans": "ZS=F", "Wheat": "ZW=F"}
NASS_BASE_URL   = "https://quickstats.nass.usda.gov/api/api_GET/"
NASS_COMMODITY  = {"Corn": "CORN", "Soybeans": "SOYBEANS", "Wheat": "WHEAT"}

# NASS stat categories to try in order — cash bids are available year-round;
# price received is only published post-harvest
NASS_STAT_CATS = [
    "PRICES, CASH, BIDS",   # weekly elevator cash price — year-round
    "PRICE RECEIVED",       # post-harvest only, fallback
]


def fetch_futures_price(crop: str) -> float:
    sym = FUTURES_TICKERS[crop]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = yf.Ticker(sym).history(period="5d")
    if hist is None or hist.empty:
        raise RuntimeError(f"No futures data for {sym}. Check yfinance/network.")
    price = float(hist["Close"].iloc[-1])
    if price < 30:
        price *= 100  # normalise to cents/bu
    return round(price, 2)


def fetch_futures_history(crop: str, weeks: int = 104) -> pd.DataFrame:
    sym = FUTURES_TICKERS[crop]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = yf.Ticker(sym).history(period=f"{weeks}wk", interval="1wk")
    if hist is None or hist.empty:
        raise RuntimeError(f"No historical futures data for {sym}")
    df = hist[["Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = ["close_cents"]
    if df["close_cents"].median() < 30:
        df["close_cents"] *= 100
    return df.round(2)


def fetch_nass_cash_prices(crop: str, api_key: str,
                            year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch weekly cash prices from USDA NASS QuickStats.

    Tries PRICES,CASH,BIDS (year-round) before PRICE RECEIVED (harvest only).
    Walks back up to 2 years per category to find available data.
    """
    commodity = NASS_COMMODITY.get(crop, crop.upper())
    base_yr   = year or _dt.datetime.now().year

    for stat_cat in NASS_STAT_CATS:
        for yr in range(base_yr, base_yr - 3, -1):
            try:
                resp = requests.get(NASS_BASE_URL, params={
                    "key":              api_key,
                    "commodity_desc":   commodity,
                    "statisticcat_desc":stat_cat,
                    "year":             yr,
                    "freq_desc":        "WEEKLY",
                    "format":           "json",
                }, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("data"):
                    log.info("NASS cash prices: %s / %s / %d", commodity, stat_cat, yr)
                    return _parse_nass(payload)
                log.debug("NASS no data: %s / %s / %d", commodity, stat_cat, yr)
            except requests.HTTPError as e:
                log.warning("NASS HTTP error %s / %s / %d: %s", commodity, stat_cat, yr, e)
            except Exception as e:
                raise RuntimeError(f"NASS fetch error: {e}") from e

    raise RuntimeError(
        f"NASS returned no weekly cash price data for {commodity}. "
        f"Tried categories: {NASS_STAT_CATS}, years {base_yr}–{base_yr-2}. "
        "Upload a cash price CSV instead."
    )


def _parse_nass(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["data"])
    df["week_ending"] = pd.to_datetime(df.get("week_ending", ""), errors="coerce")
    df = df.dropna(subset=["week_ending"])
    df = df.sort_values("week_ending").groupby("state_alpha").last().reset_index()
    df["cash_price"] = (
        pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce") * 100
    )  # $/bu → ¢/bu
    df = df.dropna(subset=["cash_price"])
    return df[["state_alpha", "cash_price"]].rename(columns={"state_alpha": "state"})


def build_cash_prices_from_nass(nass_df: pd.DataFrame, crd_gdf) -> pd.DataFrame:
    state_map = nass_df.set_index("state")["cash_price"].to_dict()
    records   = [
        {"crd_id": str(r["crd_id"]), "cash_price": state_map[r["state"]]}
        for _, r in crd_gdf.iterrows()
        if r["state"] in state_map
    ]
    return pd.DataFrame(records)


def load_cash_prices_from_upload(csv_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    missing = {"crd_id", "cash_price"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["crd_id"]     = df["crd_id"].astype(str).str.strip()
    df["cash_price"] = pd.to_numeric(df["cash_price"], errors="coerce")
    df = df.dropna(subset=["cash_price"])
    if df.empty:
        raise ValueError("No valid rows in cash price CSV")
    return df[["crd_id", "cash_price"]]


def compute_basis(cash_df: pd.DataFrame, futures_price: float) -> pd.DataFrame:
    df = cash_df.copy()
    df["futures_price"] = futures_price
    df["basis"]         = (df["cash_price"] - futures_price).round(2)
    df["basis_pct"]     = ((df["basis"] / futures_price) * 100).round(3)
    return df
