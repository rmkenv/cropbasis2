"""
api/data/ndvi.py
Sentinel-2 L2A NDVI masked to USDA Cropland Data Layer (CDL) crop pixels.

What changed vs. the naive centroid approach:
  - CDL fetched for each CRD bounding box via Planetary Computer usda-cdl COG
    (falls back to USDA CropScape WCS if PC CDL unavailable)
  - Only pixels classified as the target crop in CDL are included in NDVI
  - S2 10m pixels are aligned to the 30m CDL grid via nearest-neighbour resize
  - Minimum 10 crop pixels required; returns None if CRD has insufficient coverage
  - CDL mask cached in /tmp keyed by (crd_id, year, crop) — annual data, safe to cache

CDL crop class codes (USDA NASS, stable across years):
  Corn:     1, +doubles 225,226,237,241
  Soybeans: 5, +doubles 26,205,226,239,254
  Wheat:    22,23,24,26,51,52, +doubles 225,236
"""

import io
import logging
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.ndimage import zoom

log = logging.getLogger(__name__)

import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

# ── Constants ─────────────────────────────────────────────────────────────────

PC_STAC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
CDL_WCS_URL  = "https://nassgeodata.gmu.edu/CropScape/wms_dls/"
MAX_CLOUD    = 25.0
WINDOW_DEG   = 0.15    # widened from 0.10 to ensure enough crop pixels
MAX_WORKERS  = 4
MIN_CROP_PIX = 10      # minimum CDL crop pixels to compute a valid NDVI
CDL_RES_DEG  = 0.00027 # ~30m in degrees at 40°N

# USDA CDL class codes for each crop (primary + double-crop variants)
CDL_CLASSES: dict[str, set[int]] = {
    "Corn":     {1, 225, 226, 237, 241},
    "Soybeans": {5, 26, 205, 226, 239, 254},
    "Wheat":    {22, 23, 24, 26, 51, 52, 225, 236},
}

# /tmp cache dir — survives Vercel warm lambda reuse
_CDL_CACHE_DIR = Path("/tmp/cdl_masks")
_CDL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_ndvi_for_crds(crd_gdf: gpd.GeoDataFrame,
                         target_date: datetime,
                         crop: str = "Corn") -> pd.DataFrame:
    """
    For every CRD, fetch Sentinel-2 NDVI restricted to CDL-confirmed crop pixels.

    Parameters
    ----------
    crd_gdf     : GeoDataFrame with crd_id + geometry columns
    target_date : centre date for the analysis window
    crop        : "Corn" | "Soybeans" | "Wheat" — drives CDL class filter

    Returns
    -------
    DataFrame: crd_id, ndvi_current, ndvi_baseline, ndvi_std, ndvi_zscore,
               n_crop_pixels (how many CDL pixels matched the crop class)
    """
    if crop not in CDL_CLASSES:
        raise ValueError(f"Unknown crop '{crop}'. Expected: {list(CDL_CLASSES)}")

    # app.py injects secrets as PC_SUBSCRIPTION_KEY into os.environ
    pc_key = os.environ.get("PC_SUBSCRIPTION_KEY") or os.environ.get("PC_SDK_SUBSCRIPTION_KEY")
    if pc_key:
        planetary_computer.settings.set_subscription_key(pc_key)

    records, failed = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_ndvi_for_row, row, target_date, crop): row["crd_id"]
            for _, row in crd_gdf.iterrows()
        }
        for fut in as_completed(futures):
            crd_id = futures[fut]
            try:
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                else:
                    failed.append(crd_id)
            except Exception as exc:
                log.warning("NDVI failed %s: %s", crd_id, exc)
                failed.append(crd_id)

    if failed:
        pct = len(failed) / len(crd_gdf) * 100
        log.warning(
            "%d/%d CRDs (%.0f%%) had no usable %s CDL pixels or Sentinel-2 scenes: %s",
            len(failed), len(crd_gdf), pct, crop, failed[:6],
        )

    return pd.DataFrame(records)


# ── Per-CRD worker ────────────────────────────────────────────────────────────

def _ndvi_for_row(row, target_date: datetime, crop: str) -> Optional[dict]:
    crd_id = str(row["crd_id"])
    bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)

    # Expand bounds to WINDOW_DEG minimum width/height
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    half = WINDOW_DEG / 2
    aoi_bounds = (
        min(bounds[0], cx - half),
        min(bounds[1], cy - half),
        max(bounds[2], cx + half),
        max(bounds[3], cy + half),
    )
    aoi = box(*aoi_bounds)

    # ── CDL mask ──────────────────────────────────────────────────────────────
    crop_mask, cdl_shape = _get_cdl_mask(crd_id, aoi_bounds, target_date.year, crop)
    if crop_mask is None or crop_mask.sum() < MIN_CROP_PIX:
        log.debug("CRD %s: insufficient %s CDL pixels (%s)",
                  crd_id, crop, crop_mask.sum() if crop_mask is not None else 0)
        return None

    # ── Current NDVI over crop pixels ────────────────────────────────────────
    current, n_pix = _fetch_ndvi_masked(aoi, aoi_bounds, crop_mask, cdl_shape,
        (target_date - timedelta(days=10)).strftime("%Y-%m-%d"),
        (target_date + timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    if current is None:
        return None

    # ── 5-year same-week baseline ─────────────────────────────────────────────
    baseline_vals = []
    for yr_off in range(1, 6):
        hist = target_date.replace(year=target_date.year - yr_off)
        # Re-fetch CDL mask for the historical year (different crop patterns)
        hist_mask, hist_shape = _get_cdl_mask(
            crd_id, aoi_bounds, hist.year, crop
        )
        if hist_mask is None or hist_mask.sum() < MIN_CROP_PIX:
            # Use current-year mask as proxy if historical CDL unavailable
            hist_mask, hist_shape = crop_mask, cdl_shape

        v, _ = _fetch_ndvi_masked(aoi, aoi_bounds, hist_mask, hist_shape,
            (hist - timedelta(days=10)).strftime("%Y-%m-%d"),
            (hist + timedelta(days=10)).strftime("%Y-%m-%d"),
        )
        if v is not None:
            baseline_vals.append(v)

    if len(baseline_vals) < 2:
        log.debug("CRD %s: only %d baseline years with data", crd_id, len(baseline_vals))
        return None

    baseline = float(np.mean(baseline_vals))
    std      = float(np.std(baseline_vals, ddof=1))
    std_safe = max(std, 0.04)
    zscore   = (current - baseline) / std_safe

    return {
        "crd_id":         crd_id,
        "ndvi_current":   round(current,  4),
        "ndvi_baseline":  round(baseline, 4),
        "ndvi_std":       round(std,      4),
        "ndvi_zscore":    round(zscore,   3),
        "n_crop_pixels":  int(n_pix),
    }


# ── CDL mask fetch + cache ────────────────────────────────────────────────────

def _get_cdl_mask(crd_id: str, bounds: tuple, year: int,
                   crop: str) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Return (bool_mask, (rows, cols)) where True = target crop pixel.
    Tries PC STAC usda-cdl first, falls back to CropScape WCS.
    Caches result to /tmp.
    """
    cache_key = f"{crd_id}_{year}_{crop}"
    cache_path = _CDL_CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache_path.unlink(missing_ok=True)

    result = _fetch_cdl_pc(bounds, year, crop)
    if result[0] is None:
        result = _fetch_cdl_wcs(bounds, year, crop)

    # Cache even None results (avoids re-fetching known-empty areas)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def _fetch_cdl_pc(bounds: tuple, year: int,
                   crop: str) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Fetch CDL from Planetary Computer usda-cdl COG collection.
    PC hosts annual CONUS CDL as a cloud-optimised GeoTIFF.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = pystac_client.Client.open(
                PC_STAC_URL, modifier=planetary_computer.sign_inplace
            )

        search = client.search(
            collections = ["usda-cdl"],
            bbox        = list(bounds),          # [minx, miny, maxx, maxy]
            datetime    = f"{year}-01-01/{year}-12-31",
            max_items   = 1,
        )
        items = list(search.items())
        if not items:
            return None, None

        href = planetary_computer.sign(items[0].assets["data"].href)
        return _read_cdl_mask(href, bounds, crop)

    except Exception as exc:
        log.debug("PC CDL fetch failed (%s), trying WCS fallback", exc)
        return None, None


def _fetch_cdl_wcs(bounds: tuple, year: int,
                    crop: str) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Fetch CDL from USDA CropScape WCS endpoint (fallback).
    Returns a GeoTIFF as bytes, read via rasterio MemoryFile.
    """
    minx, miny, maxx, maxy = bounds
    params = {
        "SERVICE":  "WCS",
        "VERSION":  "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": f"CDL{year}",
        "CRS":      "EPSG:4326",
        "BBOX":     f"{minx},{miny},{maxx},{maxy}",
        "RESX":     str(CDL_RES_DEG),
        "RESY":     str(CDL_RES_DEG),
        "FORMAT":   "GTiff",
    }
    try:
        resp = requests.get(CDL_WCS_URL, params=params, timeout=30)
        resp.raise_for_status()
        if b"GeoTIFF" not in resp.content[:100] and b"II" not in resp.content[:4] \
                and b"MM" not in resp.content[:4]:
            log.warning("CDL WCS returned non-TIFF content for year %d", year)
            return None, None

        with rasterio.MemoryFile(io.BytesIO(resp.content)) as memfile:
            with memfile.open() as src:
                data = src.read(1)

        return _classify_cdl(data, crop)

    except Exception as exc:
        log.warning("CDL WCS fetch failed for %d: %s", year, exc)
        return None, None


def _read_cdl_mask(href: str, bounds: tuple,
                    crop: str) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """Read a windowed region from a CDL COG and return the crop mask."""
    try:
        with rasterio.open(href) as src:
            win  = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=win)
        return _classify_cdl(data, crop)
    except Exception as exc:
        log.debug("CDL COG read failed: %s", exc)
        return None, None


def _classify_cdl(data: np.ndarray,
                   crop: str) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """Convert raw CDL pixel values to a boolean mask for the target crop."""
    if data.size == 0:
        return None, None
    classes = CDL_CLASSES[crop]
    mask    = np.isin(data, list(classes))
    return mask, data.shape


# ── Sentinel-2 NDVI with CDL mask applied ────────────────────────────────────

@lru_cache(maxsize=512)
def _fetch_s2_items(date_start: str, date_end: str,
                     aoi_wkt: str) -> tuple:
    """Cached STAC search — returns tuple of item IDs + hrefs."""
    from shapely import wkt
    aoi = wkt.loads(aoi_wkt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
    search = client.search(
        collections = ["sentinel-2-l2a"],
        intersects  = mapping(aoi),
        datetime    = f"{date_start}/{date_end}",
        query       = {"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby      = "eo:cloud_cover",
        max_items   = 4,
    )
    items = list(search.items())
    # Return hashable structure for lru_cache
    return tuple(
        (item.id,
         planetary_computer.sign(item.assets["B04"].href),
         planetary_computer.sign(item.assets["B08"].href))
        for item in items
    )


def _fetch_ndvi_masked(aoi, aoi_bounds: tuple,
                        crop_mask: np.ndarray, cdl_shape: tuple,
                        date_start: str, date_end: str) -> tuple[Optional[float], int]:
    """
    Fetch Sentinel-2 scenes, align CDL mask to S2 pixel grid,
    compute NDVI only over confirmed crop pixels.

    Returns (median_ndvi, n_crop_pixels_used).
    """
    aoi_wkt = aoi.wkt
    try:
        item_refs = _fetch_s2_items(date_start, date_end, aoi_wkt)
    except Exception as exc:
        log.debug("S2 STAC search failed: %s", exc)
        return None, 0

    if not item_refs:
        return None, 0

    all_ndvi = []
    n_crop   = 0

    for item_id, red_href, nir_href in item_refs:
        try:
            ndvi_vals, n = _compute_masked_ndvi(
                red_href, nir_href, aoi_bounds, crop_mask, cdl_shape
            )
            if ndvi_vals is not None:
                all_ndvi.extend(ndvi_vals)
                n_crop = max(n_crop, n)
        except Exception as exc:
            log.debug("Scene %s failed: %s", item_id, exc)
            continue

    if not all_ndvi:
        return None, 0

    return float(np.median(all_ndvi)), n_crop


def _compute_masked_ndvi(red_href: str, nir_href: str,
                          bounds: tuple, crop_mask: np.ndarray,
                          cdl_shape: tuple) -> tuple[Optional[list], int]:
    """
    Read S2 B04/B08 for the bounding box, resize CDL mask to match S2 grid,
    return NDVI values for crop-classified pixels only.
    """
    def _read(href: str) -> np.ndarray:
        with rasterio.open(href) as src:
            win = from_bounds(*bounds, transform=src.transform)
            return src.read(1, window=win).astype(np.float32)

    red = _read(red_href)
    nir = _read(nir_href)

    if red.size == 0 or nir.size == 0:
        return None, 0

    # ── Align CDL mask to S2 pixel dimensions ─────────────────────────────────
    # S2 B04/B08 are 10m; CDL is 30m → S2 grid is ~3x finer
    # Use nearest-neighbour zoom so crop boundaries aren't blurred
    if red.shape != cdl_shape:
        zoom_factors = (red.shape[0] / cdl_shape[0],
                        red.shape[1] / cdl_shape[1])
        aligned_mask = zoom(crop_mask.astype(np.uint8),
                            zoom_factors,
                            order=0).astype(bool)   # order=0 = nearest-neighbour
    else:
        aligned_mask = crop_mask

    # ── Quality filter + crop mask ─────────────────────────────────────────────
    s2_valid  = (red > 0) & (nir > 0) & (red < 10000) & (nir < 10000)
    final_mask = s2_valid & aligned_mask

    n_crop = int(final_mask.sum())
    if n_crop < MIN_CROP_PIX:
        return None, n_crop

    ndvi = (nir[final_mask] - red[final_mask]) / \
           (nir[final_mask] + red[final_mask])

    return ndvi.tolist(), n_crop
