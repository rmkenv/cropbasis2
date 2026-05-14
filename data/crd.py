"""
data/crd.py
USDA Crop Reporting District (CRD) boundary loader.

Fetches from USDA NASS ArcGIS REST service in paginated batches of 1000
(the service hard-caps resultRecordCount at 1000; requesting more returns 400).
Falls back to an alternative USDA endpoint if the primary is unavailable.
Results cached to /tmp for the lifetime of the Streamlit Cloud container.
"""

import io
import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Primary: USDA NASS ArcGIS REST — paginated, max 1000 records per request
_NASS_BASE = (
    "https://services.arcgis.com/jsIt88o7Q1aNtdNe/arcgis/rest/services/"
    "CropReportingDistricts_2023/FeatureServer/0/query"
)

# Fallback: USDA ERS GeoJSON (full file, single request)
_ERS_FALLBACK_URL = (
    "https://services.arcgis.com/jsIt88o7Q1aNtdNe/arcgis/rest/services/"
    "CropReportingDistricts_2023/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&outSR=4326"  # no resultRecordCount
)

_CACHE = Path("/tmp/crd_boundaries.geojson")

CROP_STATES: dict[str, set[str]] = {
    "Corn":     {"IA", "IL", "IN", "OH", "MN", "WI", "MO", "KS", "NE", "SD", "ND", "MI", "KY", "PA"},
    "Soybeans": {"IA", "IL", "IN", "OH", "MN", "MO", "KS", "NE", "ND", "SD", "MI", "AR", "MS"},
    "Wheat":    {"KS", "OK", "TX", "CO", "NE", "SD", "ND", "MT", "WA", "OR", "ID", "WY", "MN"},
}


def load_crd_boundaries(crop: str = "Corn",
                         state_filter: list[str] | None = None) -> gpd.GeoDataFrame:
    gdf    = _load_or_fetch()
    states = set(state_filter) if state_filter else CROP_STATES.get(crop, set())
    if states:
        gdf = gdf[gdf["state"].isin(states)].copy()
    if gdf.empty:
        raise RuntimeError(
            f"No CRD boundaries found for states={states}. "
            "Check state abbreviations or widen the filter."
        )
    return gdf.reset_index(drop=True)


def _load_or_fetch() -> gpd.GeoDataFrame:
    if _CACHE.exists():
        log.info("Loading CRD boundaries from /tmp cache")
        try:
            return _normalise(gpd.read_file(_CACHE))
        except Exception as e:
            log.warning("Cache read failed (%s), re-fetching", e)
            _CACHE.unlink(missing_ok=True)

    gdf = _fetch_paginated()
    if gdf is None or gdf.empty:
        log.warning("Paginated fetch failed, trying fallback URL")
        gdf = _fetch_fallback()

    if gdf is None or gdf.empty:
        raise RuntimeError(
            "Could not fetch CRD boundaries from USDA NASS. "
            "Check network connectivity."
        )

    try:
        gdf.to_file(_CACHE, driver="GeoJSON")
        log.info("Cached %d CRDs to /tmp", len(gdf))
    except Exception as e:
        log.warning("Cache write failed: %s", e)

    return gdf


def _fetch_paginated() -> gpd.GeoDataFrame | None:
    """
    Fetch all CRDs in batches of 1000 using resultOffset pagination.
    The NASS service hard-caps resultRecordCount at 1000; requesting
    more returns a 400 error.
    """
    all_features = []
    offset       = 0
    batch_size   = 1000
    crs          = None

    log.info("Fetching CRD boundaries from USDA NASS (paginated)…")

    while True:
        params = {
            "where":             "1=1",
            "outFields":         "*",
            "f":                 "geojson",
            "outSR":             "4326",
            "resultRecordCount": batch_size,
            "resultOffset":      offset,
        }
        try:
            resp = requests.get(_NASS_BASE, params=params, timeout=60)
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error("NASS paginated fetch HTTP error at offset %d: %s", offset, e)
            return None
        except requests.RequestException as e:
            log.error("NASS paginated fetch network error: %s", e)
            return None

        try:
            data = resp.json()
        except Exception as e:
            log.error("NASS response not valid JSON: %s", e)
            return None

        features = data.get("features", [])
        if not features:
            break   # no more records

        all_features.extend(features)
        log.info("  fetched %d records (offset %d)", len(features), offset)

        if len(features) < batch_size:
            break   # last page
        offset += batch_size

    if not all_features:
        return None

    geojson = {
        "type":     "FeatureCollection",
        "features": all_features,
    }
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    return _normalise(gdf)


def _fetch_fallback() -> gpd.GeoDataFrame | None:
    """
    Fallback: single GeoJSON request without resultRecordCount.
    Some ArcGIS services honour this; others return their default page size.
    """
    log.info("Trying fallback CRD fetch (no pagination params)…")
    try:
        resp = requests.get(_ERS_FALLBACK_URL, timeout=90)
        resp.raise_for_status()
        gdf = gpd.read_file(io.BytesIO(resp.content))
        return _normalise(gdf)
    except Exception as e:
        log.error("Fallback CRD fetch failed: %s", e)
        return None


def _normalise(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rename = {}
    cols   = set(gdf.columns)
    for c in ("CRD_CD", "GEOID", "OBJECTID"):
        if c in cols and "crd_id" not in rename.values():
            rename[c] = "crd_id"
    for c in ("NAME", "CRD_NAME", "DISTNAME"):
        if c in cols and "crd_name" not in rename.values():
            rename[c] = "crd_name"
    for c in ("STATE_ABBR", "STUSPS", "State"):
        if c in cols and "state" not in rename.values():
            rename[c] = "state"
    gdf = gdf.rename(columns=rename)
    if "crd_id"   not in gdf.columns: gdf["crd_id"]   = gdf.index.astype(str)
    if "crd_name" not in gdf.columns: gdf["crd_name"] = gdf["crd_id"]
    if "state"    not in gdf.columns: gdf["state"]    = "US"
    gdf["crd_id"] = gdf["crd_id"].astype(str).str.strip()
    return gdf[["crd_id", "crd_name", "state", "geometry"]].to_crs("EPSG:4326")


def join_data_to_crds(crd_gdf: gpd.GeoDataFrame,
                       ndvi_df: pd.DataFrame,
                       basis_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = crd_gdf.merge(ndvi_df,  on="crd_id", how="left")
    merged = merged.merge(basis_df,  on="crd_id", how="left")
    fill = {
        "ndvi_current":  0.5,
        "ndvi_baseline": 0.5,
        "ndvi_std":      0.05,
        "ndvi_zscore":   0.0,
        "cash_price":    0.0,
        "futures_price": 0.0,
        "basis":         0.0,
        "basis_pct":     0.0,
    }
    for col, val in fill.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)
    return merged
