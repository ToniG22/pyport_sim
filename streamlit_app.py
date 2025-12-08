"""Streamlit app for viewing port simulation database."""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data tables available for querying
DATA_TABLES = ("measurements", "forecast", "scheduling")

# Set page config
st.set_page_config(
    page_title="Port Simulation Database Viewer", page_icon="⚓", layout="wide"
)


@st.cache_data
def get_database_files():
    """Find all .db files in the project directory."""
    project_dir = Path(__file__).parent
    db_files = list(project_dir.glob("*.db"))
    return [str(db.relative_to(project_dir)) for db in db_files]


@st.cache_data(ttl=5)
def get_tables(db_path: str):
    """Get list of tables in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return []


@st.cache_data(ttl=5)
def get_sources(db_path: str):
    """Get all sources from the database."""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT source_id, source_name, source_type FROM source ORDER BY source_type, source_name"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading sources: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def get_metrics(db_path: str):
    """Get all metrics from the database."""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT metric_id, metric_name, unit, data_type FROM metric ORDER BY metric_name"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_data(
    db_path: str,
    table: str,
    source_ids: list = None,
    metric_ids: list = None,
    start_time: str = None,
    end_time: str = None,
):
    """Load data from database with JOINs to get source and metric names."""
    if table not in DATA_TABLES:
        st.error(f"Invalid table: {table}")
        return None

    try:
        conn = sqlite3.connect(db_path)

        # Build query with JOINs
        query = f"""
            SELECT 
                d.measurement_id,
                d.timestamp,
                s.source_name,
                s.source_type,
                m.metric_name,
                m.unit,
                d.value
            FROM {table} d
            JOIN source s ON d.source_id = s.source_id
            JOIN metric m ON d.metric_id = m.metric_id
            WHERE 1=1
        """
        params = []

        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            query += f" AND d.source_id IN ({placeholders})"
            # Convert numpy.int64 to Python int for SQLite compatibility
            params.extend([int(x) for x in source_ids])

        if metric_ids:
            placeholders = ",".join("?" * len(metric_ids))
            query += f" AND d.metric_id IN ({placeholders})"
            # Convert numpy.int64 to Python int for SQLite compatibility
            params.extend([int(x) for x in metric_ids])

        if start_time:
            query += " AND d.timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND d.timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY d.timestamp"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=5)
def get_table_counts(db_path: str):
    """Get record counts for each data table."""
    counts = {}
    try:
        conn = sqlite3.connect(db_path)
        for table in DATA_TABLES:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        st.error(f"Error getting counts: {e}")
    return counts


@st.cache_data(ttl=5)
def get_time_range(db_path: str, table: str):
    """Get min and max timestamps from a table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
        result = cursor.fetchone()
        conn.close()
        return result[0], result[1]
    except Exception as e:
        return None, None


def create_time_series_plot(df: pd.DataFrame, group_by: str = "auto"):
    """Create a time series plot from the dataframe."""
    if df.empty:
        return None

    # Convert value to numeric
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Determine grouping
    unique_sources = df["source_name"].nunique()
    unique_metrics = df["metric_name"].nunique()

    if group_by == "auto":
        if unique_metrics == 1:
            group_by = "source"
        elif unique_sources == 1:
            group_by = "metric"
        else:
            group_by = "both"

    fig = go.Figure()

    if group_by == "source":
        for source in df["source_name"].unique():
            source_df = df[df["source_name"] == source]
            fig.add_trace(
                go.Scatter(
                    x=source_df["timestamp"],
                    y=source_df["value"],
                    name=source,
                    mode="lines",
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                    + "Value: %{y:.2f}<extra></extra>",
                )
            )
        y_title = df["metric_name"].iloc[0]
        unit = df["unit"].iloc[0]
        if unit:
            y_title += f" ({unit})"

    elif group_by == "metric":
        for metric in df["metric_name"].unique():
            metric_df = df[df["metric_name"] == metric]
            unit = metric_df["unit"].iloc[0]
            label = f"{metric} ({unit})" if unit else metric
            fig.add_trace(
                go.Scatter(
                    x=metric_df["timestamp"],
                    y=metric_df["value"],
                    name=label,
                    mode="lines",
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                    + "Value: %{y:.2f}<extra></extra>",
                )
            )
        y_title = "Value"

    else:  # both
        for source in df["source_name"].unique():
            for metric in df["metric_name"].unique():
                mask = (df["source_name"] == source) & (df["metric_name"] == metric)
                subset = df[mask]
                if not subset.empty:
                    unit = subset["unit"].iloc[0]
                    label = f"{source} - {metric}"
                    if unit:
                        label += f" ({unit})"
                    fig.add_trace(
                        go.Scatter(
                            x=subset["timestamp"],
                            y=subset["value"],
                            name=label,
                            mode="lines",
                            hovertemplate="<b>%{fullData.name}</b><br>"
                            + "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                            + "Value: %{y:.2f}<extra></extra>",
                        )
                    )
        y_title = "Value"

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title=y_title,
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    st.title("⚓ Port Simulation Database Viewer")

    # Sidebar - Database selection
    st.sidebar.header("🗄️ Database Selection")

    db_files = get_database_files()
    if not db_files:
        st.error("No database files found in the project directory!")
        st.info("Run the simulation first with `python main.py`")
        return

    selected_db = st.sidebar.selectbox("Select Database", db_files, key="db_selector")

    # Check if required tables exist
    tables = get_tables(selected_db)
    required_tables = {"source", "metric", "measurements", "forecast", "scheduling"}
    missing_tables = required_tables - set(tables)

    if missing_tables:
        st.error(f"Database missing required tables: {missing_tables}")
        st.info("The database schema may not be initialized. Run the simulation first.")
        return

    # Show database overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Overview")

    # Table counts
    counts = get_table_counts(selected_db)
    for table, count in counts.items():
        st.sidebar.metric(f"{table.capitalize()}", count)

    # Source and metric info
    sources_df = get_sources(selected_db)
    metrics_df = get_metrics(selected_db)

    st.sidebar.metric("Sources", len(sources_df))
    st.sidebar.metric("Metrics", len(metrics_df))

    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Data Explorer", "📋 Sources & Metrics", "📊 Compare Tables", "🔍 Raw Data"]
    )

    # Tab 1: Data Explorer
    with tab1:
        st.header("Data Explorer")

        if sources_df.empty or metrics_df.empty:
            st.warning("No sources or metrics found in database.")
        else:
            # Controls
            col1, col2 = st.columns(2)

            with col1:
                selected_table = st.selectbox(
                    "📁 Data Table",
                    DATA_TABLES,
                    format_func=lambda x: x.capitalize(),
                    help="Select which data table to query",
                    key="explorer_table_selector",
                )

            with col2:
                # Get time range for selected table
                min_time, max_time = get_time_range(selected_db, selected_table)
                if min_time and max_time:
                    try:
                        min_dt = datetime.fromisoformat(min_time.replace("Z", "+00:00"))
                        max_dt = datetime.fromisoformat(max_time.replace("Z", "+00:00"))
                        st.info(
                            f"Time range: {min_dt.strftime('%Y-%m-%d %H:%M')} to {max_dt.strftime('%Y-%m-%d %H:%M')}"
                        )
                    except:
                        pass

            # Source selection with type grouping
            st.subheader("🔌 Select Sources")
            source_types = sources_df["source_type"].unique().tolist()

            source_cols = st.columns(len(source_types)) if source_types else [st]
            selected_source_ids = []

            for i, source_type in enumerate(source_types):
                with source_cols[i]:
                    type_sources = sources_df[sources_df["source_type"] == source_type]
                    st.markdown(f"**{source_type.capitalize()}**")
                    for _, row in type_sources.iterrows():
                        if st.checkbox(
                            row["source_name"], key=f"src_{row['source_id']}", value=False
                        ):
                            selected_source_ids.append(row["source_id"])

            # Metric selection
            st.subheader("📏 Select Metrics")

            # Group metrics by prefix for better organization
            metric_groups = {}
            for _, row in metrics_df.iterrows():
                prefix = row["metric_name"].split("_")[0]
                if prefix not in metric_groups:
                    metric_groups[prefix] = []
                metric_groups[prefix].append(row)

            metric_cols = st.columns(min(4, len(metric_groups)))
            selected_metric_ids = []

            for i, (group_name, metrics) in enumerate(metric_groups.items()):
                with metric_cols[i % len(metric_cols)]:
                    st.markdown(f"**{group_name.capitalize()}**")
                    for metric in metrics:
                        unit_str = f" ({metric['unit']})" if metric["unit"] else ""
                        if st.checkbox(
                            f"{metric['metric_name']}{unit_str}",
                            key=f"metric_{metric['metric_id']}",
                            value=False,
                        ):
                            selected_metric_ids.append(metric["metric_id"])

            # Load and display data
            st.markdown("---")

            if not selected_source_ids and not selected_metric_ids:
                st.info("👆 Select at least one source or metric to view data")
            else:
                # Load data
                df = load_data(
                    selected_db,
                    selected_table,
                    source_ids=selected_source_ids if selected_source_ids else None,
                    metric_ids=selected_metric_ids if selected_metric_ids else None,
                )

                if df is None or df.empty:
                    st.warning("No data matches the selected filters")
                else:
                    st.success(f"📊 Loaded **{len(df)}** records")

                    # Plot
                    fig = create_time_series_plot(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Statistics
                    with st.expander("📈 Statistics", expanded=False):
                        df_numeric = df.copy()
                        df_numeric["value"] = pd.to_numeric(
                            df_numeric["value"], errors="coerce"
                        )

                        stats_by_source_metric = df_numeric.groupby(
                            ["source_name", "metric_name"]
                        )["value"].agg(["count", "mean", "std", "min", "max"])
                        st.dataframe(stats_by_source_metric, use_container_width=True)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"{selected_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    # Tab 2: Sources & Metrics
    with tab2:
        st.header("Sources & Metrics Registry")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔌 Sources")
            if not sources_df.empty:
                # Show sources grouped by type
                for source_type in sources_df["source_type"].unique():
                    type_df = sources_df[sources_df["source_type"] == source_type]
                    with st.expander(
                        f"{source_type.capitalize()} ({len(type_df)})", expanded=True
                    ):
                        st.dataframe(
                            type_df[["source_id", "source_name"]].reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.info("No sources registered")

        with col2:
            st.subheader("📏 Metrics")
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("No metrics registered")

    # Tab 3: Compare Tables
    with tab3:
        st.header("Compare Data Tables")
        st.info(
            "Compare the same source/metric across measurements, forecasts, and scheduling"
        )

        col1, col2 = st.columns(2)

        with col1:
            compare_source = st.selectbox(
                "Select Source",
                options=sources_df["source_name"].tolist() if not sources_df.empty else [],
                key="compare_source",
            )

        with col2:
            compare_metric = st.selectbox(
                "Select Metric",
                options=metrics_df["metric_name"].tolist() if not metrics_df.empty else [],
                key="compare_metric",
            )

        if compare_source and compare_metric:
            # Get IDs
            source_id = sources_df[sources_df["source_name"] == compare_source][
                "source_id"
            ].iloc[0]
            metric_id = metrics_df[metrics_df["metric_name"] == compare_metric][
                "metric_id"
            ].iloc[0]

            # Load from all tables
            fig = go.Figure()

            colors = {
                "measurements": "#636EFA",
                "forecast": "#EF553B",
                "scheduling": "#00CC96",
            }

            has_data = False
            for table in DATA_TABLES:
                df = load_data(
                    selected_db, table, source_ids=[source_id], metric_ids=[metric_id]
                )
                if df is not None and not df.empty:
                    has_data = True
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["value"],
                            name=table.capitalize(),
                            mode="lines",
                            line=dict(color=colors[table]),
                            hovertemplate=f"<b>{table.capitalize()}</b><br>"
                            + "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                            + "Value: %{y:.2f}<extra></extra>",
                        )
                    )

            if has_data:
                unit = metrics_df[metrics_df["metric_name"] == compare_metric][
                    "unit"
                ].iloc[0]
                y_title = f"{compare_metric} ({unit})" if unit else compare_metric

                fig.update_layout(
                    title=f"{compare_source} - {compare_metric}",
                    xaxis_title="Timestamp",
                    yaxis_title=y_title,
                    hovermode="x unified",
                    height=500,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    f"No data found for {compare_source}/{compare_metric} in any table"
                )

    # Tab 4: Raw Data
    with tab4:
        st.header("Raw Data View")

        raw_table = st.selectbox(
            "Select Table",
            DATA_TABLES,
            format_func=lambda x: x.capitalize(),
            key="raw_table",
        )

        # Load all data from table (limited)
        try:
            conn = sqlite3.connect(selected_db)
            query = f"""
                SELECT 
                    d.measurement_id,
                    d.timestamp,
                    s.source_name,
                    s.source_type,
                    m.metric_name,
                    m.unit,
                    d.value
                FROM {raw_table} d
                JOIN source s ON d.source_id = s.source_id
                JOIN metric m ON d.metric_id = m.metric_id
                ORDER BY d.timestamp DESC
                LIMIT 1000
            """
            raw_df = pd.read_sql_query(query, conn)
            conn.close()

            st.info(f"Showing latest 1000 records from {raw_table}")
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading raw data: {e}")


if __name__ == "__main__":
    main()
