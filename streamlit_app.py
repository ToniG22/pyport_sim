"""Streamlit app for viewing port simulation database."""

import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

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
def load_data(db_path: str, table: str = "measurements"):
    """Load data from database."""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


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


def main():
    st.title("⚓ Port Simulation Database Viewer")

    # Sidebar - Database selection
    st.sidebar.header("Database Selection")

    db_files = get_database_files()
    if not db_files:
        st.error("No database files found in the project directory!")
        return

    selected_db = st.sidebar.selectbox("Select Database", db_files)

    # Get available tables
    tables = get_tables(selected_db)
    if not tables:
        st.error(f"No tables found in {selected_db}")
        return

    selected_table = st.sidebar.selectbox(
        "Select Table", tables, index=0 if "measurements" in tables else 0
    )

    # Load data
    df = load_data(selected_db, selected_table)

    if df is None or df.empty:
        st.warning(f"No data found in {selected_table} table")
        return

    # Parse timestamps
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except:
            st.error("Could not parse timestamp column")
            return
    else:
        st.error("No timestamp column found")
        return

    # Show basic info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Database Info")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Unique Timestamps", len(df["timestamp"].unique()))

    # Get unique sources and metrics
    sources = sorted(df["source"].unique().tolist())
    metrics = sorted(df["metric"].unique().tolist())

    # Main controls in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_sources = st.multiselect(
            "Sources",
            sources,
            help="Select one or more sources to display"
        )

    with col2:
        selected_metrics = st.multiselect(
            "Metrics",
            metrics,
            help="Select one or more metrics to display"
        )

    with col3:
        # Date range
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Filter data
    filtered_df = df.copy()

    if selected_sources:
        filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

    if selected_metrics:
        filtered_df = filtered_df[filtered_df["metric"].isin(selected_metrics)]

    # Apply date range filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_date)
            & (filtered_df["timestamp"].dt.date <= end_date)
        ]

    # Show filtered data info
    st.markdown("---")
    st.info(f"📊 Showing **{len(filtered_df)}** records")

    # Plot
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
    else:
        # Pivot data for plotting - prioritize showing sources as lines
        if len(selected_metrics) == 1:
            # Single metric - show sources as separate lines
            pivot_df = filtered_df.pivot_table(
                index="timestamp", columns="source", values="value", aggfunc="first"
            )
        elif len(selected_sources) == 1:
            # Single source - show metrics as separate lines
            pivot_df = filtered_df.pivot_table(
                index="timestamp", columns="metric", values="value", aggfunc="first"
            )
        else:
            # Multiple sources and metrics - create combined labels
            filtered_df["label"] = filtered_df["source"] + " - " + filtered_df["metric"]
            pivot_df = filtered_df.pivot_table(
                index="timestamp", columns="label", values="value", aggfunc="first"
            )

        st.line_chart(pivot_df, use_container_width=True, height=500)

        # Statistics
        st.markdown("---")
        st.subheader("Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(filtered_df[["value"]].describe(), use_container_width=True)

        with col2:
            # Show raw data sample
            st.subheader("Data Sample")
            st.dataframe(filtered_df.head(10), use_container_width=True)

        # Download button
        st.markdown("---")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
