"""
data/ndvi.py
Sentinel-2 L2A NDVI via Planetary Computer STAC.

CDL masking strategy (in priority order):
  1. Planetary Computer usda-cdl COG collection (preferred)
  2. USDA CropScape WCS endpoint (fallback)
  3. Unmasked centroid NDVI (degraded fallback when both CDL sources unreachable)
     — logged as a warning; n_crop_pixels = 0 flags these in the UI

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
WINDOW_DEG   = 0.15
MAX_WORKERS  = 4
MIN_CROP_PIX = 10
CDL_RES_DEG  = 0.00027  # ~30m in degrees at 40°N
CDL_TIMEOUT  = 20       # seconds — fail fast so we fall through to unmasked

CDL_CLASSES: dict[str, set[int]] = {
    "Corn":     {1, 225, 226, 237, 241},
    "Soybeans": {5, 26, 205, 226, 239, 254},
    "Wheat":    {22, 23, 24, 26, 51, 52, 225, 236},
}

_CDL_CACHE_DIR = Path("/tmp/cdl_masks")
_CDL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Module-level flag: set True after first CDL failure to skip retries this session
_cdl_unavailable: bool = False


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_ndvi_for_crds(crd_gdf: gpd.GeoDataFrame,
                         target_date: datetime,
                         crop: str = "Corn") -> pd.DataFrame:
    """
    For every CRD fetch Sentinel-2 NDVI.
    Uses CDL crop-pixel masking when available; falls back to unmasked centroid
    NDVI when CDL sources are unreachable (n_crop_pixels = 0 in that case).
    """
    global _cdl_unavailable

    if crop not in CDL_CLASSES:
        raise ValueError(f"Unknown crop '{crop}'. Expected: {list(CDL_CLASSES)}")

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
        log.warning("%d/%d CRDs had no usable Sentinel-2 data", len(failed), len(crd_gdf))

    return pd.DataFrame(records)


# ── Per-CRD worker ────────────────────────────────────────────────────────────

def _ndvi_for_row(row, target_date: datetime, crop: str) -> Optional[dict]:
    global _cdl_unavailable

    crd_id = str(row["crd_id"])
    bounds = row.geometry.bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    half = WINDOW_DEG / 2
    aoi_bounds = (
        min(bounds[0], cx - half), min(bounds[1], cy - half),
        max(bounds[2], cx + half), max(bounds[3], cy + half),
    )
    aoi = box(*aoi_bounds)

    # ── Try CDL-masked NDVI first ─────────────────────────────────────────────
    if not _cdl_unavailable:
        result = _ndvi_with_cdl(crd_id, aoi, aoi_bounds, target_date, crop)
        if result is not None:
            return result
        # If CDL fetch returned None, check if it was a connectivity failure
        # by probing the cache — if nothing was cached, CDL is unreachable
        cache_probe = _CDL_CACHE_DIR / f"{crd_id}_{target_date.year}_{crop}.pkl"
        if not cache_probe.exists():
            log.warning("CDL unreachable for %s — switching to unmasked NDVI fallback", crd_id)
            _cdl_unavailable = True

    # ── Unmasked centroid NDVI fallback ───────────────────────────────────────
    # Used when both PC CDL and CropScape WCS are unavailable.
    # n_crop_pixels = 0 flags these rows in the UI as unverified.
    return _ndvi_unmasked(crd_id, aoi, aoi_bounds, target_date)


def _ndvi_with_cdl(crd_id, aoi, aoi_bounds, target_date, crop) -> Optional[dict]:
    """NDVI restricted to CDL-confirmed crop pixels."""
    crop_mask, cdl_shape = _get_cdl_mask(crd_id, aoi_bounds, target_date.year, crop)
    if crop_mask is None or crop_mask.sum() < MIN_CROP_PIX:
        return None

    current, n_pix = _fetch_ndvi_masked(
        aoi, aoi_bounds, crop_mask, cdl_shape,
        (target_date - timedelta(days=10)).strftime("%Y-%m-%d"),
        (target_date + timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    if current is None:
        return None

    baseline_vals = []
    for yr_off in range(1, 6):
        hist = target_date.replace(year=target_date.year - yr_off)
        h_mask, h_shape = _get_cdl_mask(crd_id, aoi_bounds, hist.year, crop)
        if h_mask is None or h_mask.sum() < MIN_CROP_PIX:
            h_mask, h_shape = crop_mask, cdl_shape
        v, _ = _fetch_ndvi_masked(
            aoi, aoi_bounds, h_mask, h_shape,
            (hist - timedelta(days=10)).strftime("%Y-%m-%d"),
            (hist + timedelta(days=10)).strftime("%Y-%m-%d"),
        )
        if v is not None:
            baseline_vals.append(v)

    if len(baseline_vals) < 2:
        return None

    return _build_record(crd_id, current, baseline_vals, n_pix)


def _ndvi_unmasked(crd_id, aoi, aoi_bounds, target_date) -> Optional[dict]:
    """
    NDVI over all valid pixels in the AOI — no crop masking.
    Used as degraded fallback when CDL is unavailable.
    n_crop_pixels = 0 signals unmasked in the UI.
    """
    current, _ = _fetch_ndvi_raw(
        aoi, aoi_bounds,
        (target_date - timedelta(days=10)).strftime("%Y-%m-%d"),
        (target_date + timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    if current is None:
        return None

    baseline_vals = []
    for yr_off in range(1, 6):
        hist = target_date.replace(year=target_date.year - yr_off)
        v, _ = _fetch_ndvi_raw(
            aoi, aoi_bounds,
            (hist - timedelta(days=10)).strftime("%Y-%m-%d"),
            (hist + timedelta(days=10)).strftime("%Y-%m-%d"),
        )
        if v is not None:
            baseline_vals.append(v)

    if len(baseline_vals) < 2:
        return None

    return _build_record(crd_id, current, baseline_vals, n_crop_pixels=0)


def _build_record(crd_id, current, baseline_vals, n_crop_pixels) -> dict:
    baseline = float(np.mean(baseline_vals))
    std      = float(np.std(baseline_vals, ddof=1))
    zscore   = (current - baseline) / max(std, 0.04)
    return {
        "crd_id":        crd_id,
        "ndvi_current":  round(current,  4),
        "ndvi_baseline": round(baseline, 4),
        "ndvi_std":      round(std,      4),
        "ndvi_zscore":   round(zscore,   3),
        "n_crop_pixels": int(n_crop_pixels),
    }


# ── CDL mask fetch + cache ────────────────────────────────────────────────────

def _get_cdl_mask(crd_id, bounds, year, crop):
    cache_path = _CDL_CACHE_DIR / f"{crd_id}_{year}_{crop}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache_path.unlink(missing_ok=True)

    result = _fetch_cdl_pc(bounds, year, crop)
    if result[0] is None:
        result = _fetch_cdl_wcs(bounds, year, crop)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def _fetch_cdl_pc(bounds, year, crop):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = pystac_client.Client.open(
                PC_STAC_URL, modifier=planetary_computer.sign_inplace
            )
        search = client.search(
            collections=["usda-cdl"],
            bbox=list(bounds),
            datetime=f"{year}-01-01/{year}-12-31",
            max_items=1,
        )
        items = list(search.items())
        if not items:
            return None, None
        href = planetary_computer.sign(items[0].assets["data"].href)
        return _read_cdl_mask(href, bounds, crop)
    except Exception as exc:
        log.debug("PC CDL fetch failed: %s", exc)
        return None, None


def _fetch_cdl_wcs(bounds, year, crop):
    minx, miny, maxx, maxy = bounds
    # CDL only exists through the prior year — cap at current year - 1
    import datetime as _dt
    cdl_year = min(year, _dt.datetime.now().year - 1)
    params = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": f"CDL{cdl_year}", "CRS": "EPSG:4326",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "RESX": str(CDL_RES_DEG), "RESY": str(CDL_RES_DEG), "FORMAT": "GTiff",
    }
    try:
        resp = requests.get(CDL_WCS_URL, params=params, timeout=CDL_TIMEOUT)
        resp.raise_for_status()
        if b"II" not in resp.content[:4] and b"MM" not in resp.content[:4]:
            log.warning("CDL WCS returned non-TIFF for %d", cdl_year)
            return None, None
        with rasterio.MemoryFile(io.BytesIO(resp.content)) as memfile:
            with memfile.open() as src:
                data = src.read(1)
        return _classify_cdl(data, crop)
    except Exception as exc:
        log.warning("CDL WCS fetch failed for %d: %s", cdl_year, exc)
        return None, None


def _read_cdl_mask(href, bounds, crop):
    try:
        with rasterio.open(href) as src:
            win  = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=win)
        return _classify_cdl(data, crop)
    except Exception as exc:
        log.debug("CDL COG read failed: %s", exc)
        return None, None


def _classify_cdl(data, crop):
    if data.size == 0:
        return None, None
    mask = np.isin(data, list(CDL_CLASSES[crop]))
    return mask, data.shape


# ── Sentinel-2 NDVI ───────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _fetch_s2_items(date_start, date_end, aoi_wkt):
    from shapely import wkt as shapely_wkt
    aoi = shapely_wkt.loads(aoi_wkt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(aoi),
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby="eo:cloud_cover",
        max_items=4,
    )
    items = list(search.items())
    return tuple(
        (item.id,
         planetary_computer.sign(item.assets["B04"].href),
         planetary_computer.sign(item.assets["B08"].href))
        for item in items
    )


def _fetch_ndvi_masked(aoi, aoi_bounds, crop_mask, cdl_shape, date_start, date_end):
    """NDVI over CDL-masked crop pixels."""
    try:
        item_refs = _fetch_s2_items(date_start, date_end, aoi.wkt)
    except Exception as exc:
        log.debug("S2 STAC search failed: %s", exc)
        return None, 0

    if not item_refs:
        return None, 0

    all_ndvi, n_crop = [], 0
    for item_id, red_href, nir_href in item_refs:
        try:
            vals, n = _compute_masked_ndvi(red_href, nir_href, aoi_bounds, crop_mask, cdl_shape)
            if vals is not None:
                all_ndvi.extend(vals)
                n_crop = max(n_crop, n)
        except Exception as exc:
            log.debug("Scene %s failed: %s", item_id, exc)

    return (float(np.median(all_ndvi)), n_crop) if all_ndvi else (None, 0)


def _fetch_ndvi_raw(aoi, aoi_bounds, date_start, date_end):
    """Unmasked NDVI over all valid pixels — CDL fallback."""
    try:
        item_refs = _fetch_s2_items(date_start, date_end, aoi.wkt)
    except Exception as exc:
        log.debug("S2 STAC search failed: %s", exc)
        return None, 0

    if not item_refs:
        return None, 0

    all_ndvi = []
    for item_id, red_href, nir_href in item_refs:
        try:
            def _read(href):
                with rasterio.open(href) as src:
                    win = from_bounds(*aoi_bounds, transform=src.transform)
                    return src.read(1, window=win).astype(np.float32)
            red, nir = _read(red_href), _read(nir_href)
            valid = (red > 0) & (nir > 0) & (red < 10000) & (nir < 10000)
            if valid.sum() >= MIN_CROP_PIX:
                ndvi = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
                all_ndvi.extend(ndvi.tolist())
        except Exception as exc:
            log.debug("Unmasked scene %s failed: %s", item_id, exc)

    return (float(np.median(all_ndvi)), 0) if all_ndvi else (None, 0)


def _compute_masked_ndvi(red_href, nir_href, bounds, crop_mask, cdl_shape):
    def _read(href):
        with rasterio.open(href) as src:
            win = from_bounds(*bounds, transform=src.transform)
            return src.read(1, window=win).astype(np.float32)

    red, nir = _read(red_href), _read(nir_href)
    if red.size == 0 or nir.size == 0:
        return None, 0

    if red.shape != cdl_shape:
        zf = (red.shape[0] / cdl_shape[0], red.shape[1] / cdl_shape[1])
        aligned_mask = zoom(crop_mask.astype(np.uint8), zf, order=0).astype(bool)
    else:
        aligned_mask = crop_mask

    valid = (red > 0) & (nir > 0) & (red < 10000) & (nir < 10000)
    final = valid & aligned_mask
    n     = int(final.sum())
    if n < MIN_CROP_PIX:
        return None, n

    ndvi = (nir[final] - red[final]) / (nir[final] + red[final])
    return ndvi.tolist(), n
