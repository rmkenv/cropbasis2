"""
data/crd.py
USDA Crop Reporting District (CRD) boundary loader.

Loads from a bundled GeoJSON file (data/crd_boundaries.geojson) that ships
with the repo — no network call required. The boundaries are constructed from
the USDA NASS 3x3 geographic grid structure for each major crop-producing state.

CRD IDs follow USDA convention: {STATE}{DISTRICT_NUMBER}
e.g. IA10 = Iowa Northwest, IA50 = Iowa Central, KS90 = Kansas Southeast
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

# Bundled boundaries — relative to this file's location
_BUNDLED = Path(__file__).parent / "crd_boundaries.geojson"

CROP_STATES: dict[str, set[str]] = {
    "Corn":     {"IA", "IL", "IN", "OH", "MN", "WI", "MO", "KS", "NE", "SD", "ND", "MI", "KY", "PA"},
    "Soybeans": {"IA", "IL", "IN", "OH", "MN", "MO", "KS", "NE", "ND", "SD", "MI", "AR", "MS"},
    "Wheat":    {"KS", "OK", "TX", "CO", "NE", "SD", "ND", "MT", "WA", "OR", "ID", "WY", "MN"},
}


def load_crd_boundaries(crop: str = "Corn",
                         state_filter: list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Load CRD boundaries for the given crop's key states from the bundled file.

    Parameters
    ----------
    crop         : "Corn" | "Soybeans" | "Wheat" — selects default state set
    state_filter : optional explicit list e.g. ["IA", "IL"] — overrides crop default
    """
    if not _BUNDLED.exists():
        raise FileNotFoundError(
            f"Bundled CRD file not found at {_BUNDLED}. "
            "Ensure data/crd_boundaries.geojson is committed to the repo."
        )

    gdf    = gpd.read_file(_BUNDLED)
    states = set(state_filter) if state_filter else CROP_STATES.get(crop, set())

    if states:
        gdf = gdf[gdf["state"].isin(states)].copy()

    if gdf.empty:
        raise RuntimeError(
            f"No CRD boundaries found for states={states}. "
            f"Available states: {sorted(gpd.read_file(_BUNDLED)['state'].unique())}"
        )

    log.info("Loaded %d CRDs for %s (%s)", len(gdf), crop, states)
    return gdf.reset_index(drop=True)


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
