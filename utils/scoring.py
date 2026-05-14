"""
api/utils/scoring.py
3-component Basis Risk Index: NDVI z-score + basis deviation + Kalshi PMD.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

WEIGHTS_KALSHI    = {"ndvi": 0.50, "basis": 0.35, "pmd": 0.15}
WEIGHTS_NO_KALSHI = {"ndvi": 0.60, "basis": 0.40, "pmd": 0.00}

# Columns always passed through to top_risk_table and GeoJSON properties.
# n_crop_pixels is the CDL validation signal: how many confirmed crop pixels
# were used in the NDVI calculation.  Low counts (< 50) are flagged in the UI.
_EXPORT_COLS = [
    "crd_id", "crd_name", "state",
    "ndvi_zscore", "ndvi_current", "ndvi_baseline", "ndvi_stress_flag",
    "n_crop_pixels",        # ← CDL pixel count — key validity signal
    "cash_price", "futures_price", "basis", "basis_pct",
    "ndvi_component", "basis_component", "pmd_component",
    "bri", "risk_label", "bri_color",
]


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_bri(gdf: gpd.GeoDataFrame,
                ndvi_threshold: float = -1.5,
                pmd_zscore: float = 0.0) -> gpd.GeoDataFrame:
    out = gdf.copy()

    # Ensure n_crop_pixels column always exists (may be absent if NDVI fetch
    # returned no records for some CRDs)
    if "n_crop_pixels" not in out.columns:
        out["n_crop_pixels"] = 0
    out["n_crop_pixels"] = out["n_crop_pixels"].fillna(0).astype(int)

    out["ndvi_stress_flag"] = out["ndvi_zscore"] < ndvi_threshold
    out["ndvi_component"]   = _minmax(out["ndvi_zscore"].abs())
    out["basis_component"]  = _minmax((out["basis"] - out["basis"].median()).abs())

    pmd_norm = float(min(max(pmd_zscore - 0.5, 0) / 3.0, 1.0)) if pmd_zscore >= 0.5 else 0.0
    out["pmd_component"] = pmd_norm

    w = WEIGHTS_KALSHI if pmd_zscore > 0 else WEIGHTS_NO_KALSHI
    out["bri"] = (
        w["ndvi"]  * out["ndvi_component"] +
        w["basis"] * out["basis_component"] +
        w["pmd"]   * out["pmd_component"]
    ).round(4)

    out["risk_label"] = pd.cut(
        out["bri"], bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Moderate", "High", "Severe"], right=False,
    ).astype(str)

    out["bri_color"] = out["bri"].apply(_hex)
    return out


def _hex(v: float) -> str:
    v = max(0.0, min(1.0, float(v)))
    if v <= 0.5:
        t = v * 2;  r, g, b = int(t * 255), 200, 50
    else:
        t = (v - 0.5) * 2;  r, g, b = 255, int((1 - t) * 200), 50
    return f"#{r:02x}{g:02x}{b:02x}"


def top_risk_table(gdf: gpd.GeoDataFrame, n: int = 10) -> list[dict]:
    """
    Return the top-n highest-BRI districts as a list of dicts.
    Includes n_crop_pixels so the frontend can flag low-confidence NDVI values.
    """
    cols = [c for c in _EXPORT_COLS if c in gdf.columns]
    top  = gdf[cols].sort_values("bri", ascending=False).head(n)
    return top.to_dict(orient="records")
