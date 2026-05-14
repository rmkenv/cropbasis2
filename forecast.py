"""
api/data/forecast.py
ARIMAX basis forecasting — SARIMAX(1,0,1)(1,0,1)52 with NDVI lag-2 exog.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

log = logging.getLogger(__name__)

ARIMA_ORDER    = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 52)
FORECAST_WEEKS = 4
MIN_OBS        = 26


def forecast_basis(basis_series: pd.Series,
                   ndvi_history: Optional[pd.DataFrame] = None,
                   crop: str = "Corn") -> dict:
    series = basis_series.dropna().sort_index()
    if len(series) < MIN_OBS:
        raise ValueError(f"Need ≥{MIN_OBS} weekly obs; got {len(series)}")

    exog_train, exog_fore, exog_used = _build_exog(series, ndvi_history)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            series, exog=exog_train,
            order=ARIMA_ORDER, seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)

    fc      = fit.get_forecast(steps=FORECAST_WEEKS, exog=exog_fore)
    summary = fc.summary_frame(alpha=0.20)

    last_date  = series.index[-1]
    fore_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=FORECAST_WEEKS, freq="W"
    )

    return {
        "forecast_df": pd.DataFrame({
            "date":     fore_dates.strftime("%Y-%m-%d").tolist(),
            "forecast": summary["mean"].round(2).tolist(),
            "lower_80": summary["mean_ci_lower"].round(2).tolist(),
            "upper_80": summary["mean_ci_upper"].round(2).tolist(),
        }),
        "current_basis": round(float(series.iloc[-1]), 2),
        "forecast_4wk":  round(float(summary["mean"].iloc[-1]), 2),
        "forecast_bias": round(float(summary["mean"].iloc[-1]) - float(series.iloc[-1]), 2),
        "model_aic":     round(float(fit.aic), 1),
        "exog_used":     exog_used,
    }


def _build_exog(series, ndvi_history):
    if ndvi_history is None or ndvi_history.empty:
        return None, None, False
    try:
        ndvi = (
            ndvi_history["ndvi_zscore"]
            .resample("W").mean()
            .rolling(3, min_periods=1).mean()
            .shift(2)
        )
        aligned    = ndvi.reindex(series.index).fillna(0.0)
        last_val   = float(aligned.dropna().iloc[-1]) if not aligned.dropna().empty else 0.0
        return (
            aligned.values.reshape(-1, 1),
            np.full((FORECAST_WEEKS, 1), last_val),
            True,
        )
    except Exception as exc:
        log.warning("NDVI exog failed: %s", exc)
        return None, None, False


def build_district_basis_history(futures_hist: pd.DataFrame,
                                  cash_price: float,
                                  futures_price: float) -> pd.Series:
    if futures_hist is None or futures_hist.empty:
        raise ValueError("futures_hist is empty")
    current_basis = cash_price - futures_price
    latest        = float(futures_hist["close_cents"].iloc[-1])
    delta         = futures_hist["close_cents"] - latest
    return (current_basis + delta).rename("basis_cents")
