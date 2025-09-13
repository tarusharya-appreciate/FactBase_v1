# app.py
from __future__ import annotations
from datetime import datetime, date, timedelta
import io
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# =================== PAGE & THEME ===================
st.set_page_config(page_title="Live Positions Dashboard+", layout="wide", page_icon="📊")

# Minimal custom CSS for clean visuals
st.markdown("""
<style>
:root {
  --card-bg: #ffffff10;
  --bdr: 12px;
}
[data-testid="stMetric"] {
  background: var(--card-bg);
  border-radius: var(--bdr);
  padding: 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,.08);
}
.block-container { padding-top: 1.2rem; }
hr { margin: 0.6rem 0 1.2rem 0; }
.kpi {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
  border: 1px solid #e5e7eb;
  border-radius: var(--bdr);
  padding: 14px 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,.05);
}
.section-title {
  font-size: 1.2rem; font-weight: 700; margin: 4px 0 6px 0;
}
.small-muted { color: #6b7280; font-size: .88rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #f6f7fb; border-radius: 999px; padding: 8px 14px;
  box-shadow: inset 0 0 0 1px #e5e7eb;
}
.stTabs [aria-selected="true"] { background: #e5edff; box-shadow: inset 0 0 0 2px #c7d2fe; }
.ag-theme-balham .ag-header-cell-label { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Live Positions Dashboard+")
st.caption("Filters · grouping · pivoting · charts · export · auto-refresh")
st.divider()

# ============== ALTair defaults for readable charts ==============
alt.themes.enable("none")
alt.data_transformers.disable_max_rows()
BASE_FONT = "Inter, Segoe UI, system-ui, sans-serif"

def style_chart(ch: alt.Chart, height=380):
    return (
        ch.properties(height=height)
        .configure_axis(labelFont=BASE_FONT, titleFont=BASE_FONT, grid=True, gridOpacity=0.12)
        .configure_legend(labelFont=BASE_FONT, titleFont=BASE_FONT, orient="top")
        .configure_view(strokeOpacity=0)
        .configure_title(font=BASE_FONT, fontSize=14, anchor="start")
    )

# =================== DATA SOURCE ===================
S3_PREFIX = "https://analytic-purposes.s3.ap-south-1.amazonaws.com/fact_base_alpaca/v1/live_update_file/"

def csv_url_for(d: date) -> str:
    return f"{S3_PREFIX}{d.strftime('%Y-%m-%d')}.csv"

def check_exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=8)
        return r.ok
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=300)
def read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

@st.cache_data(show_spinner=False, ttl=300)
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            # try datetime
            try:
                parsed = pd.to_datetime(out[c], errors="raise", infer_datetime_format=True)
                out[c] = parsed
                continue
            except Exception:
                pass
            # try numeric
            try:
                cleaned = out[c].astype(str).str.replace(",", "", regex=False)
                num = pd.to_numeric(cleaned, errors="coerce")
                if num.notna().mean() > 0.7:
                    out[c] = num
            except Exception:
                pass
    return out

# ============== Sidebar: Source + refresh ============
with st.sidebar:
    st.header("📦 Data Source")

    pick_date = st.date_input("Target date", value=datetime.now().date(), format="YYYY-MM-DD")
    use_latest_available = st.toggle("Use most recent available (past 14 days)", value=True)
    resolved_url = csv_url_for(pick_date)
    if use_latest_available:
        found = False
        for d in range(0, 15):
            candidate = csv_url_for(pick_date - timedelta(days=d))
            if check_exists(candidate):
                resolved_url = candidate
                found = True
                break
        if not found:
            st.warning("No file found in the last 14 days. Using selected date URL.")
    st.code(resolved_url, language="text")

    uploaded = st.file_uploader("Or upload a CSV (overrides URL)", type=["csv"])
    st.markdown("---")
    st.header("♻️ Auto Refresh")
    auto_refresh = st.toggle("Enable auto refresh", value=False)
    refresh_every = st.number_input("Refresh interval (seconds)", min_value=120, max_value=600, value=200, step=5)

    load_btn = st.button("📥 Load / Reload Data", use_container_width=True)

# =================== LOAD DATA ===================
if load_btn or not st.session_state.get("df_loaded"):
    try:
        with st.spinner("Loading data…"):
            if uploaded is not None:
                df = read_csv_bytes(uploaded.read())
                st.session_state["source"] = "uploaded"
            else:
                df = read_csv_url(resolved_url)
                st.session_state["source"] = resolved_url
            df = coerce_types(df)
            st.session_state["df"] = df
            st.session_state["df_loaded"] = True
            st.session_state["df_view"] = df.copy()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df: pd.DataFrame | None = st.session_state.get("df")
if df is None or df.empty:
    st.warning("No data loaded yet.")
    st.stop()

st.success(f"Loaded {len(df):,} rows from: {st.session_state.get('source')}")

# =================== KPI STRIP ===================
k1, k2, k3, k4 = st.columns([1,1,1,1])
with k1:
    st.metric("Rows", f"{len(df):,}")
with k2:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.metric("Numeric Columns", f"{len(num_cols)}")
with k3:
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    st.metric("Datetime Columns", f"{len(dt_cols)}")
with k4:
    st.metric("Columns", f"{len(df.columns)}")

st.divider()

# =================== TABS ===================
tab_data, tab_pivot, tab_charts, tab_summary = st.tabs([
    "🔎 Data & Filters",
    "📐 Pivot Lab",
    "📈 Graph Builder",
    "🧪 Summary & Export"
])

# -------------------- TAB 1: DATA & FILTERS --------------------
with tab_data:
    st.markdown('<div class="section-title">Interactive filters</div>', unsafe_allow_html=True)
    with st.container():
        cols = df.columns.tolist()
        df_view = df.copy()

        co1, co2 = st.columns([1,2])
        with co1:
            col_to_filter = st.selectbox("Column to filter", options=["(none)"] + cols, index=0)
        with co2:
            q = st.text_input("Free text search (across all columns)")

        if col_to_filter != "(none)":
            sample_vals = sorted(df_view[col_to_filter].dropna().astype(str).unique().tolist())[:2000]
            selected_vals = st.multiselect(f"Filter values in `{col_to_filter}`", options=sample_vals)
            if selected_vals:
                df_view = df_view[df_view[col_to_filter].astype(str).isin(selected_vals)]

        if q:
            mask = df_view.apply(lambda s: s.astype(str).str.contains(q, case=False, na=False))
            df_view = df_view[mask.any(axis=1)]

        # Save for other tabs
        st.session_state["df_view"] = df_view

    st.markdown('<div class="section-title">Interactive table</div>', unsafe_allow_html=True)

    def build_grid(dfi: pd.DataFrame):
        gb = GridOptionsBuilder.from_dataframe(dfi)
        gb.configure_default_column(
            filter=True, sortable=True, resizable=True, editable=False,
            enablePivot=True, enableRowGroup=True, enableValue=True,
            floatingFilter=True
        )
        gb.configure_side_bar()
        gb.configure_selection("multiple", use_checkbox=True)
        gb.configure_grid_options(
            rowGroupPanelShow="always",
            pivotPanelShow="always",
            animateRows=True,
            suppressMenuHide=False,
            groupSelectsChildren=True,
            domLayout="normal",
            rowHeight=28
        )
        # Nice number formatting for numeric columns
        for c in dfi.select_dtypes(include=[np.number]).columns:
            gb.configure_column(c, type=["numericColumn", "numberColumnFilter"])

        grid_options = gb.build()
        grid = AgGrid(
            dfi,
            gridOptions=grid_options,
            theme="balham",                # visually nicer grid theme
            height=620,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            enable_enterprise_modules=True,
            allow_unsafe_jscode=True
        )
        return grid

    grid = build_grid(st.session_state["df_view"])

    with st.expander("ℹ️ Grid Summary", expanded=False):
        st.write("Filtered & sorted rows visible:", len(grid["data"]))

# -------------------- TAB 2: PIVOT LAB --------------------
with tab_pivot:
    st.markdown('<div class="section-title">Build your pivot</div>', unsafe_allow_html=True)
    dfv = st.session_state["df_view"]
    cols_all = dfv.columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        idx_cols = st.multiselect("Rows (index)", options=cols_all, help="Fields to appear as rows")
    with c2:
        col_cols = st.multiselect("Columns", options=[c for c in cols_all if c not in idx_cols], help="Pivot into columns")
    with c3:
        val_cols = st.multiselect("Values", options=[c for c in cols_all if c not in idx_cols + col_cols], help="Numeric fields to aggregate")

    c4, c5, c6, c7 = st.columns([1,1,1,1])
    with c4:
        agg_names = st.multiselect("Aggregations", options=["sum", "mean", "count", "min", "max", "median", "nunique"], default=["sum"])
    with c5:
        margins = st.checkbox("Grand totals (margins)", value=False)
    with c6:
        fill_zeros = st.checkbox("Fill NaN with 0", value=True)
    with c7:
        heatmap_toggle = st.checkbox("Heatmap", value=False)

    c8, c9 = st.columns([2,1])
    with c8:
        topn_sort_col = st.selectbox("Top-N by (optional)", options=["(none)"] + (val_cols if val_cols else []), index=0)
    with c9:
        topn_k = st.number_input("Top-N rows", min_value=1, value=20)

    def _aggfunc_from_name(name: str):
        return {
            "sum": "sum",
            "mean": "mean",
            "count": "count",
            "min": "min",
            "max": "max",
            "median": "median",
            "nunique": pd.Series.nunique,
        }[name]

    def compute_percent_of_total(pivot_df: pd.DataFrame) -> pd.DataFrame:
        base = pivot_df.copy()
        numeric_cols = base.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return base
        totals = base[numeric_cols].sum()
        pct = base.copy()
        pct[numeric_cols] = (base[numeric_cols] / totals.replace({0: np.nan})) * 100.0
        return pct

    if idx_cols and val_cols and agg_names:
        try:
            aggfuncs = [_aggfunc_from_name(a) for a in agg_names]
            pivot = pd.pivot_table(
                dfv,
                index=idx_cols,
                columns=col_cols if col_cols else None,
                values=val_cols,
                aggfunc=aggfuncs if len(aggfuncs) > 1 else aggfuncs[0],
                margins=margins,
                dropna=False
            )
            if isinstance(pivot.columns, pd.MultiIndex):
                pivot.columns = [" | ".join(map(str, tup)) for tup in pivot.columns.to_flat_index()]
            pivot = pivot.reset_index()
            if fill_zeros:
                pivot = pivot.fillna(0)

            # Top-N sort
            if topn_sort_col != "(none)":
                candidate_cols = [c for c in pivot.columns if topn_sort_col in c]
                if candidate_cols:
                    pivot = pivot.sort_values(candidate_cols[0], ascending=False).head(topn_k)

            st.markdown('<div class="section-title">Pivot result</div>', unsafe_allow_html=True)
            st.dataframe(pivot, use_container_width=True)
            st.session_state["pivot_df"] = pivot

            # Optional heatmap
            if heatmap_toggle:
                numeric_cols = pivot.select_dtypes(include=[np.number]).columns
                if len(numeric_cols):
                    melt = pivot[idx_cols + list(numeric_cols)].melt(id_vars=idx_cols, var_name="Metric", value_name="Value")
                    melt["RowKey"] = melt[idx_cols].astype(str).agg(" | ".join, axis=1)
                    chart = alt.Chart(melt).mark_rect().encode(
                        x=alt.X("Metric:N", sort=None, title="Metric"),
                        y=alt.Y("RowKey:N", sort=None, title="Row"),
                        color=alt.Color("Value:Q", scale=alt.Scale(scheme="blues")),
                        tooltip=list(melt.columns),
                    )
                    st.altair_chart(style_chart(chart), use_container_width=True)
                else:
                    st.info("No numeric cells for heatmap.")
            # Download
            csv_bytes = pivot.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download pivot as CSV", data=csv_bytes, file_name="pivot.csv", mime="text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Pivot error: {e}")
    else:
        st.info("Choose at least **Rows** and **Values** to build a pivot.")

# -------------------- TAB 3: CHARTS --------------------
with tab_charts:
    st.markdown('<div class="section-title">Graph Builder</div>', unsafe_allow_html=True)
    dfv = st.session_state["df_view"]
    cols_all = dfv.columns.tolist()
    dt_cols = dfv.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    num_cols = dfv.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols_all if c not in num_cols]

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox("Type", ["line", "bar", "area", "scatter"])
    with c2:
        x_col = st.selectbox("X axis", options=cols_all, index=(cols_all.index(dt_cols[0]) if dt_cols else 0))
    with c3:
        group_col = st.selectbox("Group (optional)", options=["(none)"] + cat_cols, index=0)

    y_cols = st.multiselect("Y axis (one or more)", options=num_cols, default=(num_cols[:1] if num_cols else []))
    agg_name = st.selectbox("Aggregation for grouped series", options=["sum", "mean", "count", "min", "max", "median"], index=0)

    c4, c5 = st.columns(2)
    with c4:
        rolling = st.number_input("Rolling window (0=off)", min_value=0, value=0)
    with c5:
        limit_rows = st.number_input("Sample first N rows (0=all)", min_value=0, value=0)

    df_plot = dfv.copy()
    if limit_rows > 0:
        df_plot = df_plot.head(limit_rows)

    # Rolling metrics
    if rolling and y_cols:
        for c in list(y_cols):
            if pd.api.types.is_numeric_dtype(df_plot[c]):
                df_plot[f"{c}_roll{rolling}"] = df_plot[c].rolling(rolling).mean()
                y_cols.append(f"{c}_roll{rolling}")

    if chart_type in ("line", "bar", "area") and y_cols:
        long_df = df_plot.melt(
            id_vars=[x_col] + ([group_col] if group_col != "(none)" else []),
            value_vars=y_cols, var_name="Metric", value_name="Value"
        )
        if group_col != "(none)":
            long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

            long_df = (
                long_df
                .dropna(subset=["Value"])
                .groupby([x_col, group_col, "Metric"], as_index=False)
                .agg(Value=("Value", agg_name))   # <- named aggregation requires (col, func)
            )

        x_enc = alt.X(f"{x_col}:T" if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]) else f"{x_col}:N", title=x_col)
        base = alt.Chart(long_df).encode(x=x_enc, y=alt.Y("Value:Q"))
        tooltip = list(long_df.columns)

        if chart_type == "line":
            chart = base.mark_line().encode(color="Metric:N", tooltip=tooltip)
        elif chart_type == "area":
            chart = base.mark_area(opacity=0.7).encode(color="Metric:N", tooltip=tooltip)
        else:
            chart = base.mark_bar().encode(color="Metric:N", tooltip=tooltip)

        if group_col != "(none)":
            chart = chart.encode(detail=f"{group_col}:N", tooltip=tooltip + [f"{group_col}:N"])

        st.altair_chart(style_chart(chart), use_container_width=True)

    elif chart_type == "scatter" and y_cols:
        y_sc = y_cols[0]
        x_type = "Q" if pd.api.types.is_numeric_dtype(df_plot[x_col]) else ("T" if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]) else "N")
        base = (
            alt.Chart(df_plot).mark_circle(size=60)
            .encode(
                x=alt.X(f"{x_col}:{x_type}", title=x_col),
                y=alt.Y(f"{y_sc}:Q", title=y_sc),
                color=None if group_col == "(none)" else alt.Color(f"{group_col}:N"),
                tooltip=list(df_plot.columns),
            )
        )
        st.altair_chart(style_chart(base), use_container_width=True)
    else:
        st.info("Select at least one numeric Y column.")

# -------------------- TAB 4: SUMMARY & EXPORT --------------------
with tab_summary:
    st.markdown('<div class="section-title">Quick stats</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        try:
            st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)
        except Exception:
            st.info("No numeric columns to describe.")
    with c2:
        st.write("**Columns**")
        st.write(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}))

    st.markdown('<div class="section-title">Download current filtered view</div>', unsafe_allow_html=True)
    csv_bytes = st.session_state["df_view"].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="filtered_view.csv", mime="text/csv", use_container_width=True)

# =================== AUTO REFRESH ===================
if auto_refresh:
    st.toast(f"Auto refresh in {int(refresh_every)}s…", icon="♻️")
    st.rerun()
