# app.py
from datetime import datetime
import io
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(page_title="Live Positions Dashboard", layout="wide")
st.title("📊 Live Positions Dashboard")
st.caption("Filters, grouping, pivoting, sorting, export — powered by AgGrid")

# ---- Hardcoded public URL (no AWS creds needed) ----
CSV_URL = f"https://analytic-purposes.s3.ap-south-1.amazonaws.com/fact_base_alpaca/v1/live_update_file/{datetime.now().strftime('%Y-%m-%d')}.csv"

st.sidebar.header("Data Source (fixed)")
st.sidebar.code(CSV_URL)
path_to_read = CSV_URL

# ---- Cache CSV load ----
@st.cache_data(show_spinner=False, ttl=300)
def read_csv_url(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning("Couldn’t load today’s CSV. Try reloading or pick another date.")
        raise

def build_grid(df: pd.DataFrame):
    gb = GridOptionsBuilder.from_dataframe(df)
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
    grid_options = gb.build()
    grid = AgGrid(
        df,
        gridOptions=grid_options,
        height=650,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True
    )
    return grid

# ---- Auto refresh controls ----
st.sidebar.header("Auto Refresh")
auto_refresh = st.sidebar.toggle("Enable auto refresh", value=False)
refresh_every = st.sidebar.number_input("Refresh interval (seconds)", min_value=120, max_value=600, value=200, step=5)
load_btn = st.sidebar.button("📥 Load / Reload Data")

# ---- Load data ----
if load_btn or not st.session_state.get("df_loaded"):
    try:
        with st.spinner(f"Reading {path_to_read} ..."):
            df = read_csv_url(path_to_read)
            st.session_state["df"] = df
            st.session_state["df_loaded"] = True
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("DataFrame is empty.")
    st.stop()

st.success(f"Loaded {len(df):,} rows from: {path_to_read}")

# ---- Optional quick filters ----
with st.expander("🔎 Quick column filters (optional)"):
    cols = df.columns.tolist()
    col_to_filter = st.selectbox("Select a column to filter", options=["(none)"] + cols, index=0)
    df_view = df.copy()
    if col_to_filter != "(none)":
        sample_vals = sorted(df_view[col_to_filter].dropna().astype(str).unique().tolist())[:2000]
        selected_vals = st.multiselect(f"Filter values in `{col_to_filter}`", options=sample_vals)
        if selected_vals:
            df_view = df_view[df_view[col_to_filter].astype(str).isin(selected_vals)]

    q = st.text_input("Free text search (across all columns)")
    if q:
        mask = df_view.apply(lambda s: s.astype(str).str.contains(q, case=False, na=False))
        df_view = df_view[mask.any(axis=1)]

# ---- Pivot Maker (Excel-like) ----
with st.expander("📐 Pivot maker (Excel-like)"):
    cols_all = df_view.columns.tolist()

    # Choose rows (index), columns, values, and aggfunc
    idx_cols = st.multiselect("Rows (index)", options=cols_all, help="Fields to appear as rows")
    col_cols = st.multiselect("Columns", options=[c for c in cols_all if c not in idx_cols], help="Fields to pivot into columns")
    val_cols = st.multiselect("Values", options=[c for c in cols_all if c not in idx_cols + col_cols], help="Numeric fields to aggregate")

    agg_name = st.selectbox("Aggregation",
                            options=["sum", "mean", "count", "min", "max", "median", "nunique"],
                            index=0)
    margins = st.checkbox("Show grand totals (margins)", value=False)
    fill_zeros = st.checkbox("Fill NaN with 0", value=True)

    # Map agg name -> function
    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "min": "min",
        "max": "max",
        "median": "median",
        "nunique": pd.Series.nunique,
    }
    aggfunc = agg_map[agg_name]

    if idx_cols and val_cols:
        try:
            pivot = pd.pivot_table(
                df_view,
                index=idx_cols,
                columns=col_cols if col_cols else None,
                values=val_cols,
                aggfunc=aggfunc,
                margins=margins,
                dropna=False
            )

            # Flatten MultiIndex columns for display
            if isinstance(pivot.columns, pd.MultiIndex):
                pivot.columns = [" | ".join(map(str, tup)) for tup in pivot.columns.to_flat_index()]
            pivot = pivot.reset_index()

            if fill_zeros:
                pivot = pivot.fillna(0)

            st.subheader("Pivot result")
            st.dataframe(pivot, use_container_width=True)

            # Download button
            csv_bytes = pivot.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download pivot as CSV", data=csv_bytes, file_name="pivot.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Pivot error: {e}")
    else:
        st.info("Choose at least **Rows** and **Values** to build a pivot.")

# ---- Grid (original interactive table on filtered data) ----
grid = build_grid(df_view)

with st.expander("ℹ️ Summary"):
    st.write("Filtered & sorted rows currently visible in grid:", len(grid["data"]))

# ---- Auto refresh loop ----
if auto_refresh:
    st.toast(f"Auto refresh in {refresh_every}s...", icon="♻️")
    st.rerun()
