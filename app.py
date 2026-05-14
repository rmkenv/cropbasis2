"""
app.py — CropBasis
Streamlit Cloud entry point.

Secrets (set in Streamlit Cloud dashboard or .streamlit/secrets.toml locally):
    NASS_API_KEY          — USDA NASS QuickStats (required unless uploading CSV)
    PC_SDK_SUBSCRIPTION_KEY — Planetary Computer (optional, raises rate limits)
    KALSHI_API_KEY        — Kalshi (optional)

Run locally:
    streamlit run app.py
"""

import io
import json
import logging
import os
import warnings
from datetime import date, datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from data.crd      import load_crd_boundaries, join_data_to_crds
from data.ndvi     import fetch_ndvi_for_crds
from data.futures  import (fetch_futures_price, fetch_futures_history,
                            load_cash_prices_from_upload,
                            fetch_nass_cash_prices,
                            build_cash_prices_from_nass, compute_basis)
from data.kalshi   import fetch_kalshi_yield_signal
from data.forecast import forecast_basis, build_district_basis_history
from utils.scoring import compute_bri, top_risk_table

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# ── Inject Streamlit secrets into os.environ so data modules can read them ────
# ndvi.py reads PC_SUBSCRIPTION_KEY; secrets.toml may use PC_SDK_SUBSCRIPTION_KEY
# — support both names so either works.
for _k in ("NASS_API_KEY", "PC_SUBSCRIPTION_KEY", "PC_SDK_SUBSCRIPTION_KEY", "KALSHI_API_KEY"):
    if _k in st.secrets and _k not in os.environ:
        os.environ[_k] = st.secrets[_k]
# Alias so ndvi.py always finds it regardless of which name was set in secrets
if "PC_SDK_SUBSCRIPTION_KEY" in os.environ and "PC_SUBSCRIPTION_KEY" not in os.environ:
    os.environ["PC_SUBSCRIPTION_KEY"] = os.environ["PC_SDK_SUBSCRIPTION_KEY"]


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropBasis | Basis Risk Intelligence",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; background:#0d1117; color:#c9d1d9; }
.main .block-container { padding-top:1.2rem; max-width:1400px; }
section[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #30363d; }
.cb-header { font-family:'DM Serif Display',serif; font-size:2.2rem; color:#f0b429; letter-spacing:-0.02em; }
.cb-sub    { font-size:0.72rem; color:#8b949e; letter-spacing:0.08em; text-transform:uppercase; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.9rem 1.1rem; }
.metric-label { font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; }
.metric-value { font-size:1.5rem; font-weight:600; color:#f0b429; }
.metric-sub   { font-size:0.68rem; color:#8b949e; margin-top:0.1rem; }
hr { border-color:#30363d; }
.stButton>button { background:#f0b429; color:#0d1117; border:none; border-radius:4px;
  font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:0.78rem; padding:0.4rem 1.1rem; }
.stButton>button:hover { background:#ffc94a; }
.stSelectbox label,.stSlider label,.stDateInput label,.stTextInput label {
  font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; }
div[data-testid="stDataFrame"] { border:1px solid #30363d; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _card(col, label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _choropleth(gdf: gpd.GeoDataFrame):
    import plotly.express as px
    gdf2     = gdf.reset_index(drop=True)
    geojson  = json.loads(gdf2.to_json())
    for i, f in enumerate(geojson["features"]):
        f["id"] = str(i)

    hover = {c: True for c in ["state","ndvi_zscore","basis","pmd_component","risk_label"]
             if c in gdf2.columns}

    fig = px.choropleth_mapbox(
        gdf2, geojson=geojson, locations=gdf2.index, color="bri",
        hover_name="crd_name", hover_data=hover,
        color_continuous_scale=[[0,"#22c55e"],[0.25,"#a3e635"],
                                 [0.5,"#facc15"],[0.75,"#f97316"],[1,"#ef4444"]],
        range_color=(0,1), mapbox_style="carto-darkmatter",
        zoom=3.8, center={"lat":41.0,"lon":-95.0}, opacity=0.78,
        labels={"bri":"BRI","ndvi_zscore":"NDVI Z","basis":"Basis ¢/bu",
                "pmd_component":"PMD","risk_label":"Risk","state":"State"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#c9d1d9", font_family="IBM Plex Mono",
        height=560, margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(
            title="BRI", titlefont=dict(color="#c9d1d9"),
            tickfont=dict(color="#c9d1d9"), bgcolor="#161b22",
            bordercolor="#30363d", borderwidth=1, thickness=13, len=0.55,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def _risk_table(gdf: gpd.GeoDataFrame):
    rows = top_risk_table(gdf, n=10)
    df   = pd.DataFrame(rows)

    # Rename for display
    rename = {
        "crd_name":"District","state":"State","ndvi_zscore":"NDVI Z-Score",
        "n_crop_pixels":"CDL Pixels","basis":"Basis ¢/bu",
        "pmd_component":"Kalshi PMD","bri":"BRI Score","risk_label":"Risk",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    def _color_risk(val):
        return {"Severe":"color:#f85149;font-weight:600","High":"color:#f0883e;font-weight:600",
                "Moderate":"color:#d29922","Low":"color:#3fb950"}.get(str(val),"")

    styled = df.style
    if "Risk" in df.columns:
        styled = styled.map(_color_risk, subset=["Risk"])
    if "CDL Pixels" in df.columns:
        styled = styled.map(
            lambda v: "color:#d29922" if isinstance(v, (int,float)) and 0 < v < 50 else "",
            subset=["CDL Pixels"],
        )

    st.markdown("#### Top 10 Highest-Risk Districts")
    st.caption("CDL Pixels = USDA Cropland Data Layer–confirmed crop pixels used in NDVI. "
               "Values < 50 (amber) indicate sparse crop coverage — treat NDVI with caution.")
    st.dataframe(styled, use_container_width=True, height=400)

    # BRI histogram
    import plotly.express as px
    fig = px.histogram(gdf, x="bri", nbins=20, color_discrete_sequence=["#f0b429"],
                       labels={"bri":"BRI Score"}, title="BRI Distribution — All Districts")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", font_family="IBM Plex Mono",
                      height=240, margin=dict(l=10,r=10,t=36,b=10), bargap=0.08)
    fig.update_xaxes(gridcolor="#30363d")
    fig.update_yaxes(gridcolor="#30363d", title="Districts")
    st.plotly_chart(fig, use_container_width=True)


def _forecast_chart(fc: dict, crop: str):
    import plotly.graph_objects as go
    df  = pd.DataFrame(fc["forecast_df"])
    cur = fc["current_basis"]
    f4  = fc["forecast_4wk"]
    aic = fc["model_aic"]
    exog= fc["exog_used"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["date"]) + list(df["date"])[::-1],
        y=list(df["upper_80"]) + list(df["lower_80"])[::-1],
        fill="toself", fillcolor="rgba(240,180,41,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="80% PI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["forecast"],
        mode="lines+markers",
        line=dict(color="#f0b429", width=2.5), marker=dict(size=7),
        name="Forecast",
    ))
    fig.add_hline(y=0,   line_color="#30363d", line_dash="dot")
    fig.add_hline(y=cur, line_color="#8b949e", line_dash="dash",
                  annotation_text=f"Current: {cur:.1f}¢",
                  annotation_font_color="#8b949e")

    model_note = f"ARIMAX(1,0,1)(1,0,1)₅₂{' + NDVI lag-2' if exog else ''}  |  AIC {aic:.0f}"
    fig.update_layout(
        title=dict(text=f"{crop} Basis Forecast — 4 Weeks  |  {model_note}",
                   font=dict(size=13, color="#c9d1d9")),
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#c9d1d9", font_family="IBM Plex Mono",
        height=320, margin=dict(l=10,r=10,t=50,b=10),
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", title="Basis (¢/bu)"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    _card(c1, "Current Basis",   f"{cur:.1f}¢")
    bias = f4 - cur
    _card(c2, "4-Wk Forecast",  f"{f4:.1f}¢",
          sub=f"{'▲' if bias>=0 else '▼'} {abs(bias):.1f}¢ vs current")
    _card(c3, "Model AIC",       f"{aic:.0f}",
          sub="NDVI exog used" if exog else "AR-only")


def _kalshi_panel(k: dict, crop: str):
    pmd  = k["pmd_zscore"]
    imp  = k["kalshi_implied_yield"]
    wde  = k["wasde_yield"]
    n    = k["n_markets"]
    mkts = k["market_summaries"]

    c1,c2,c3,c4 = st.columns(4)
    _card(c1, "Kalshi Markets",        str(n))
    _card(c2, "PMD Z-Score",           f"{pmd:.2f}σ", sub="≥1.5σ = significant")
    _card(c3, f"{crop} Implied Yield", f"{imp:.1f} bu/ac" if imp else "—",
          sub="probability-weighted")
    _card(c4, "WASDE Consensus",       f"{wde:.1f} bu/ac", sub="Dec 2024 baseline")

    if mkts:
        df = pd.DataFrame(mkts)
        st.markdown("#### Active Kalshi Yield Markets")
        st.dataframe(df, use_container_width=True, height=260)
    else:
        st.info(f"No open Kalshi yield markets for {crop}. "
                "Off-season is normal — PMD component set to 0.")


def _export_panel(gdf: gpd.GeoDataFrame, fc: dict | None, crop: str):
    export_cols = [c for c in [
        "crd_id","crd_name","state","ndvi_zscore","ndvi_current","ndvi_baseline",
        "n_crop_pixels","cash_price","futures_price","basis","basis_pct",
        "ndvi_component","basis_component","pmd_component","bri","risk_label","ndvi_stress_flag",
    ] if c in gdf.columns]

    c1, c2, c3 = st.columns(3)

    csv_buf = io.StringIO()
    gdf[export_cols].to_csv(csv_buf, index=False)
    c1.download_button("⬇ BRI CSV", csv_buf.getvalue(),
                       f"cropbasis_{crop.lower()}_{date.today()}.csv",
                       "text/csv", use_container_width=True)

    c2.download_button("⬇ BRI GeoJSON", gdf.to_json(),
                       f"cropbasis_{crop.lower()}_{date.today()}.geojson",
                       "application/json", use_container_width=True)

    if fc:
        fc_buf = io.StringIO()
        pd.DataFrame(fc["forecast_df"]).to_csv(fc_buf, index=False)
        c3.download_button("⬇ Forecast CSV", fc_buf.getvalue(),
                           f"forecast_{crop.lower()}_{date.today()}.csv",
                           "text/csv", use_container_width=True)

    st.caption(f"{len(gdf)} CRDs · {crop} · {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _run_pipeline(crop, end_date_str, state_filter_json, nass_key):
    """Fetch CRDs, NDVI, futures. Cached 1 hr."""
    target_date  = datetime.strptime(end_date_str, "%Y-%m-%d")
    state_filter = json.loads(state_filter_json)
    crd_gdf      = load_crd_boundaries(crop=crop, state_filter=state_filter)
    ndvi_df      = fetch_ndvi_for_crds(crd_gdf, target_date, crop=crop)
    futures_px   = fetch_futures_price(crop)
    futures_hist = fetch_futures_history(crop, weeks=104)
    cash_df_nass = None
    if nass_key:
        try:
            nass_state   = fetch_nass_cash_prices(crop, nass_key)
            cash_df_nass = build_cash_prices_from_nass(nass_state, crd_gdf)
        except Exception as e:
            logging.warning("NASS QuickStats fetch failed: %s", e)
    return crd_gdf, ndvi_df, futures_px, futures_hist, cash_df_nass


@st.cache_data(ttl=3600, show_spinner=False)
def _run_kalshi(crop):
    return fetch_kalshi_yield_signal(crop)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    st.markdown("---")

    crop = st.selectbox("Crop", ["Corn", "Soybeans", "Wheat"])

    end_date = st.date_input("Analysis Date", value=date.today())

    ndvi_threshold = st.slider(
        "NDVI Stress Threshold (z-score)",
        min_value=-3.0, max_value=0.0, value=-1.5, step=0.1,
        help="Districts below this z-score are flagged as crop-stressed.",
    )

    st.markdown("---")
    st.markdown("### 📥 Cash Prices")

    cash_source = st.radio("Source", ["NASS QuickStats API", "Upload CSV"], horizontal=True)

    nass_key  = ""
    cash_file = None

    if cash_source == "NASS QuickStats API":
        # Prefer secret, allow manual override
        nass_key = os.environ.get("NASS_API_KEY", "")
        if not nass_key:
            nass_key = st.text_input("NASS API Key", type="password",
                                     help="Free at quickstats.nass.usda.gov/api")
        else:
            st.caption("✓ NASS_API_KEY loaded from secrets")
    else:
        cash_file = st.file_uploader("CSV (crd_id, cash_price in ¢/bu)", type=["csv"])

    st.markdown("---")
    state_raw = st.text_input("State Filter (optional)", placeholder="e.g. IA,IL,IN",
                              help="Blank = default states for selected crop")
    state_filter = (
        [s.strip().upper() for s in state_raw.split(",") if s.strip()]
        if state_raw.strip() else None
    )

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="cb-header">🌽 CropBasis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cb-sub">Regional Commodity Basis Risk Intelligence — '
    'Sentinel-2 NDVI × CME Futures × Kalshi Prediction Markets</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

for key in ("result_gdf","futures_px","futures_hist","forecast_result","kalshi","crop"):
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if not nass_key and cash_file is None:
        st.error("Provide a NASS API key or upload a cash price CSV.")
        st.stop()

    with st.spinner("Fetching CRD boundaries, Sentinel-2 NDVI, and futures…"):
        try:
            crd_gdf, ndvi_df, futures_px, futures_hist, cash_df_nass = _run_pipeline(
                crop             = crop,
                end_date_str     = end_date.strftime("%Y-%m-%d"),
                state_filter_json= json.dumps(state_filter),
                nass_key         = nass_key,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    with st.spinner("Computing basis…"):
        if cash_file is not None:
            try:
                cash_df = load_cash_prices_from_upload(cash_file.read())
            except ValueError as e:
                st.error(str(e)); st.stop()
        elif cash_df_nass is not None:
            cash_df = cash_df_nass
        else:
            st.error("No cash prices available. Check your NASS key or upload a CSV.")
            st.stop()

        basis_df = compute_basis(cash_df, futures_px)
        joined   = join_data_to_crds(crd_gdf, ndvi_df, basis_df)

    with st.spinner("Fetching Kalshi signal…"):
        try:
            kalshi = _run_kalshi(crop)
            pmd_z  = kalshi["pmd_zscore"]
        except Exception as e:
            st.warning(f"Kalshi unavailable: {e}")
            kalshi = {"pmd_zscore":0.0,"n_markets":0,"kalshi_implied_yield":None,
                      "wasde_yield":0.0,"market_summaries":[]}
            pmd_z  = 0.0

    result = compute_bri(joined, ndvi_threshold=ndvi_threshold, pmd_zscore=pmd_z)

    with st.spinner("Fitting ARIMAX basis forecast…"):
        try:
            median_basis = float(result["basis"].median())
            rep          = result.iloc[(result["basis"] - median_basis).abs().argsort()[:1]]
            rep_cash     = float(rep["cash_price"].iloc[0])
            basis_hist   = build_district_basis_history(futures_hist, rep_cash, futures_px)
            ndvi_hist_df = pd.DataFrame(
                {"ndvi_zscore": [float(result["ndvi_zscore"].mean())] * len(futures_hist)},
                index=futures_hist.index,
            )
            forecast_result = forecast_basis(basis_hist, ndvi_hist_df, crop=crop)
        except Exception as e:
            st.warning(f"Basis forecast failed: {e}")
            forecast_result = None

    st.session_state.result_gdf      = result
    st.session_state.futures_px      = futures_px
    st.session_state.forecast_result = forecast_result
    st.session_state.kalshi          = kalshi
    st.session_state.crop            = crop

elif st.session_state.result_gdf is not None:
    # Re-score on threshold slider change without re-fetching
    prev = st.session_state.result_gdf
    drop = [c for c in ["ndvi_component","basis_component","pmd_component",
                         "bri","risk_label","bri_color","ndvi_stress_flag","weights_used"]
            if c in prev.columns]
    k    = st.session_state.kalshi
    pmd_z= k["pmd_zscore"] if k else 0.0
    st.session_state.result_gdf = compute_bri(
        prev.drop(columns=drop), ndvi_threshold=ndvi_threshold, pmd_zscore=pmd_z
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.result_gdf is not None:
    gdf      = st.session_state.result_gdf
    fps      = st.session_state.futures_px
    fc       = st.session_state.forecast_result
    kalshi   = st.session_state.kalshi
    sel_crop = st.session_state.crop

    # Metric row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    _card(c1, "Avg BRI",          f"{gdf['bri'].mean():.3f}")
    _card(c2, "Severe CRDs",      str(int((gdf["risk_label"]=="Severe").sum())))
    _card(c3, "High CRDs",        str(int((gdf["risk_label"]=="High").sum())))
    _card(c4, f"{sel_crop} Futures", f"{fps:.1f}¢", sub="¢/bu front-month")
    _card(c5, "Stressed Districts",
          str(int(gdf["ndvi_stress_flag"].sum())),
          sub=f"NDVI < {ndvi_threshold:.1f}σ")
    _card(c6, "Kalshi PMD",
          f"{kalshi['pmd_zscore']:.2f}σ" if kalshi else "—",
          sub="0 = no divergence")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_map, tab_tbl, tab_fc, tab_kal, tab_exp = st.tabs([
        "🗺 BRI Map", "📊 Risk Table", "📈 Basis Forecast", "🎲 Kalshi", "📤 Export"
    ])
    with tab_map: _choropleth(gdf)
    with tab_tbl: _risk_table(gdf)
    with tab_fc:
        if fc: _forecast_chart(fc, sel_crop)
        else:  st.info("Basis forecast unavailable for this run.")
    with tab_kal:
        if kalshi: _kalshi_panel(kalshi, sel_crop)
        else:      st.info("No Kalshi data available.")
    with tab_exp: _export_panel(gdf, fc, sel_crop)

else:
    st.info(
        "👈 Configure parameters in the sidebar and click **▶ Run Analysis**.\n\n"
        "Cash price source required: NASS QuickStats API key or uploaded CSV.",
        icon="ℹ️",
    )
    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:1.4rem 1.8rem;font-family:'IBM Plex Mono',monospace;
                font-size:0.76rem;color:#8b949e;line-height:2;margin-top:0.8rem;">
    <div style="color:#f0b429;font-size:0.86rem;margin-bottom:0.5rem;">── Pipeline ──</div>
    Sentinel-2 L2A (Planetary Computer) → NDVI z-score vs 5-yr baseline, CDL crop-masked<br>
    CME Futures via yfinance → ZC=F / ZS=F / ZW=F<br>
    USDA NASS QuickStats or CSV → basis = cash − futures per CRD<br>
    Kalshi → PMD z-score vs WASDE<br>
    ARIMAX(1,0,1)(1,0,1)₅₂ + NDVI lag-2 → 4-wk basis forecast<br>
    BRI = 0.50×NDVI + 0.35×basis + 0.15×PMD
    </div>
    """, unsafe_allow_html=True)
