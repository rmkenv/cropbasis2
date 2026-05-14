"""
api/data/crd.py
USDA Crop Reporting District boundary loader.
Used by serverless functions — reads from /tmp cache on Vercel.
"""

import io
import logging
import os
from pathlib import Path

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

NASS_CRD_URL = (
    "https://services.arcgis.com/jsIt88o7Q1aNtdNe/arcgis/rest/services/"
    "CropReportingDistricts_2023/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&outSR=4326&resultRecordCount=2000"
)

# Vercel functions have a writable /tmp directory
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
        raise RuntimeError(f"No CRD boundaries found for states={states}")
    return gdf.reset_index(drop=True)


def _load_or_fetch() -> gpd.GeoDataFrame:
    if _CACHE.exists():
        return _normalise(gpd.read_file(_CACHE))
    resp = requests.get(NASS_CRD_URL, timeout=60)
    resp.raise_for_status()
    gdf = _normalise(gpd.read_file(io.BytesIO(resp.content)))
    gdf.to_file(_CACHE, driver="GeoJSON")
    return gdf


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
                       ndvi_df, basis_df) -> gpd.GeoDataFrame:
    import pandas as pd
    merged = crd_gdf.merge(ndvi_df,  on="crd_id", how="left")
    merged = merged.merge(basis_df,  on="crd_id", how="left")
    fill = {
        "ndvi_current": 0.5, "ndvi_baseline": 0.5, "ndvi_std": 0.05,
        "ndvi_zscore": 0.0, "cash_price": 0.0, "futures_price": 0.0,
        "basis": 0.0, "basis_pct": 0.0,
    }
    for col, val in fill.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)
    return merged
