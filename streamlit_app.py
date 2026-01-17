"""
Minimal Streamlit app for viewing port simulation databases.

UX goals:
- Only two pages: Explore / Raw data
- One simple control card
- One plot card
- Opinionated defaults
"""

from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------
# Config
# -----------------------

DATA_TABLES = ("measurements", "forecast", "scheduling")

st.set_page_config(
    page_title="Port Simulation Viewer",
    page_icon="⚓",
    layout="wide",
)

# -----------------------
# Data helpers
# -----------------------


def connect(db_path: str):
    return sqlite3.connect(db_path)


@st.cache_data
def get_db_files():
    base = Path(__file__).parent
    return sorted([str(p.name) for p in base.glob("*.db")])


@st.cache_data(ttl=5)
def get_sources(db_path: str) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT source_id, source_name FROM source ORDER BY source_name",
            conn,
        )


@st.cache_data(ttl=5)
def get_metrics(db_path: str) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT metric_id, metric_name, unit FROM metric ORDER BY metric_name",
            conn,
        )


@st.cache_data(ttl=5)
def load_data(
    db_path: str,
    table: str,
    source_ids: list[int] | None,
    metric_ids: list[int] | None,
    start_time: str | None,
    end_time: str | None,
) -> pd.DataFrame:

    with connect(db_path) as conn:
        q = f"""
            SELECT
                d.timestamp,
                s.source_name,
                m.metric_name,
                m.unit,
                d.value
            FROM {table} d
            JOIN source s ON d.source_id = s.source_id
            JOIN metric m ON d.metric_id = m.metric_id
            WHERE 1=1
        """
        params: list = []

        if source_ids:
            q += f" AND d.source_id IN ({','.join('?' * len(source_ids))})"
            params.extend(source_ids)

        if metric_ids:
            q += f" AND d.metric_id IN ({','.join('?' * len(metric_ids))})"
            params.extend(metric_ids)

        if start_time:
            q += " AND d.timestamp >= ?"
            params.append(start_time)

        if end_time:
            q += " AND d.timestamp <= ?"
            params.append(end_time)

        q += " ORDER BY d.timestamp"

        return pd.read_sql_query(q, conn, params=params)


# -----------------------
# Plot
# -----------------------


def make_plot(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    fig = go.Figure()

    for (source, metric), g in df.groupby(["source_name", "metric_name"]):
        unit = g["unit"].iloc[0]
        label = f"{source} • {metric}"
        if unit:
            label += f" ({unit})"

        fig.add_trace(
            go.Scatter(
                x=g["timestamp"],
                y=g["value"],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# -----------------------
# App
# -----------------------


def main():
    st.title("⚓ Port Simulation Viewer")

    # -----------------------
    # Sidebar
    # -----------------------

    db_files = get_db_files()
    if not db_files:
        st.error("No .db files found")
        return

    selected_db = st.sidebar.selectbox("Database", db_files)
    page = st.sidebar.radio("Page", ["Explore", "Raw data"])

    sources = get_sources(selected_db)
    metrics = get_metrics(selected_db)

    # -----------------------
    # EXPLORE PAGE
    # -----------------------

    if page == "Explore":
        st.subheader("Explore")

        # ---- Controls card ----
        with st.container(border=True):
            st.markdown("### Filters")

            col1, col2, col3 = st.columns(3)

            with col1:
                table = st.selectbox("Table", DATA_TABLES, format_func=str.capitalize)

            with col2:
                selected_sources = st.multiselect(
                    "Sources",
                    options=sources["source_name"].tolist(),
                    default=sources["source_name"].head(1).tolist(),
                )

            with col3:
                metric_labels = [
                    f"{r.metric_name} ({r.unit})" if r.unit else r.metric_name
                    for r in metrics.itertuples()
                ]
                label_to_name = dict(zip(metric_labels, metrics["metric_name"]))

                selected_metric_labels = st.multiselect(
                    "Metrics",
                    options=metric_labels,
                    default=[],
                )
                selected_metrics = [label_to_name[l] for l in selected_metric_labels]

            col4, col5 = st.columns(2)
            with col4:
                start_date = st.date_input("Start date", value=None)
            with col5:
                end_date = st.date_input("End date", value=None)

        # ---- Plot card ----
        with st.container(border=True):
            st.markdown("### Time series")

            if not selected_sources and not selected_metrics:
                st.info("Select a source or a metric to begin")
                return

            source_ids = (
                sources.loc[
                    sources["source_name"].isin(selected_sources), "source_id"
                ].tolist()
                if selected_sources
                else None
            )

            metric_ids = (
                metrics.loc[
                    metrics["metric_name"].isin(selected_metrics), "metric_id"
                ].tolist()
                if selected_metrics
                else None
            )

            start_time = (
                datetime.combine(start_date, datetime.min.time()).isoformat()
                if start_date
                else None
            )
            end_time = (
                datetime.combine(end_date, datetime.max.time()).isoformat()
                if end_date
                else None
            )

            df = load_data(
                selected_db,
                table,
                source_ids,
                metric_ids,
                start_time,
                end_time,
            )

            fig = make_plot(df)

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for the selected filters")

    # -----------------------
    # RAW DATA PAGE
    # -----------------------

    else:
        st.subheader("Raw data")

        with st.container(border=True):
            table = st.selectbox("Table", DATA_TABLES, format_func=str.capitalize)
            limit = st.slider("Rows", 100, 5000, 1000, step=100)

        with connect(selected_db) as conn:
            df = pd.read_sql_query(
                f"""
                SELECT
                    d.timestamp,
                    s.source_name,
                    m.metric_name,
                    m.unit,
                    d.value
                FROM {table} d
                JOIN source s ON d.source_id = s.source_id
                JOIN metric m ON d.metric_id = m.metric_id
                ORDER BY d.timestamp DESC
                LIMIT ?
                """,
                conn,
                params=[limit],
            )

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name=f"{table}_raw.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
