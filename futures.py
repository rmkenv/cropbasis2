"""
api/data/futures.py
CME futures via yfinance and USDA NASS QuickStats cash prices.
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

FUTURES_TICKERS = {"Corn": "ZC=F", "Soybeans": "ZS=F", "Wheat": "ZW=F"}
NASS_BASE_URL   = "https://quickstats.nass.usda.gov/api/api_GET/"
NASS_COMMODITY  = {"Corn": "CORN", "Soybeans": "SOYBEANS", "Wheat": "WHEAT"}


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
    commodity = NASS_COMMODITY.get(crop, crop.upper())
    yr        = year or __import__("datetime").datetime.now().year
    resp = requests.get(NASS_BASE_URL, params={
        "key": api_key, "commodity_desc": commodity,
        "statisticcat_desc": "PRICE RECEIVED",
        "year": yr, "freq_desc": "WEEKLY", "format": "json",
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or not data["data"]:
        raise RuntimeError(f"NASS returned no price data for {commodity} {yr}")

    df = pd.DataFrame(data["data"])
    df["week_ending"] = pd.to_datetime(df.get("week_ending", ""), errors="coerce")
    df = df.dropna(subset=["week_ending"])
    df = df.sort_values("week_ending").groupby("state_alpha").last().reset_index()
    df["cash_price"] = pd.to_numeric(
        df["Value"].str.replace(",", ""), errors="coerce") * 100  # $/bu → ¢/bu
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
