"""Unit tests for BRI scoring — no network required.
Run with: python -m pytest tests/ -v
"""
import sys
import os

# Add repo root to path so 'data' and 'utils' packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from utils.scoring import compute_bri, top_risk_table
from data.kalshi   import pmd_to_component, _parse_threshold
from data.forecast import forecast_basis, build_district_basis_history


def _make_gdf(n=20, include_crop_pixels=True):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        row = {
            "crd_id":        f"CRD_{i:03d}",
            "crd_name":      f"District {i}",
            "state":         "IA",
            "ndvi_zscore":   float(rng.normal(-0.5, 1.5)),
            "ndvi_current":  0.55,
            "ndvi_baseline": 0.60,
            "ndvi_std":      0.05,
            "cash_price":    float(480 + rng.normal(0, 20)),
            "futures_price": 480.0,
            "basis":         float(rng.normal(-15, 20)),
            "basis_pct":     -3.1,
            "geometry":      box(-93 - i * 0.1, 41, -92.9 - i * 0.1, 41.1),
        }
        if include_crop_pixels:
            row["n_crop_pixels"] = int(rng.integers(15, 800))
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_bri_range():
    gdf = compute_bri(_make_gdf(), ndvi_threshold=-1.5, pmd_zscore=0.0)
    assert gdf["bri"].min() >= 0.0
    assert gdf["bri"].max() <= 1.0


def test_bri_risk_labels():
    gdf = compute_bri(_make_gdf())
    assert set(gdf["risk_label"]).issubset({"Low", "Moderate", "High", "Severe"})


def test_n_crop_pixels_preserved():
    gdf  = compute_bri(_make_gdf(include_crop_pixels=True))
    assert "n_crop_pixels" in gdf.columns
    rows = top_risk_table(gdf, n=10)
    assert all("n_crop_pixels" in r for r in rows)


def test_n_crop_pixels_missing_gets_zero():
    gdf = _make_gdf(include_crop_pixels=False)
    assert "n_crop_pixels" not in gdf.columns
    result = compute_bri(gdf)
    assert "n_crop_pixels" in result.columns
    assert (result["n_crop_pixels"] == 0).all()


def test_pmd_component():
    assert pmd_to_component(0.3) == 0.0
    assert 0.0 < pmd_to_component(1.5) <= 1.0
    assert pmd_to_component(10.0) == 1.0


def test_bri_with_kalshi():
    gdf      = _make_gdf()
    no_pmd   = compute_bri(gdf, pmd_zscore=0.0)["bri"].mean()
    with_pmd = compute_bri(gdf, pmd_zscore=2.0)["bri"].mean()
    assert with_pmd != no_pmd


def test_top_risk_table_sorted():
    gdf  = compute_bri(_make_gdf())
    rows = top_risk_table(gdf, n=5)
    assert len(rows) == 5
    bri_vals = [r["bri"] for r in rows]
    assert bri_vals == sorted(bri_vals, reverse=True)


def test_kalshi_threshold_parse():
    assert _parse_threshold("Will USDA corn yield exceed 180 bu/acre?", 179.0) == 180.0
    assert _parse_threshold("Yield above 51.5 bu/acre", 50.0) == 51.5
    assert _parse_threshold("No number here at all", 179.0) is None


def test_forecast_basis():
    import warnings
    warnings.filterwarnings("ignore")
    rng    = np.random.default_rng(7)
    dates  = pd.date_range("2023-01-06", periods=104, freq="W")
    series = pd.Series(-15 + np.cumsum(rng.normal(0, 2, 104)), index=dates)
    ndvi   = pd.DataFrame({"ndvi_zscore": rng.normal(-0.3, 0.8, 104)}, index=dates)
    fc = forecast_basis(series, ndvi, crop="Corn")
    assert "forecast_df" in fc
    assert fc["exog_used"] is True
    assert len(fc["forecast_df"]["date"]) == 4


def test_build_district_basis_history():
    dates = pd.date_range("2023-01-06", periods=52, freq="W")
    hist  = pd.DataFrame({"close_cents": np.linspace(450, 490, 52)}, index=dates)
    basis = build_district_basis_history(hist, cash_price=470, futures_price=480)
    assert len(basis) == 52
    assert abs(float(basis.iloc[-1]) - (470 - 480)) < 0.1
