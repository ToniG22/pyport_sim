"""
Scenario comparison script for port electrification studies.

This script compares, for fleets of 5, 10, and 20 vessels:

    - No optimization, no DER    (no_opt_no_der)
    - Optimization, no DER       (opt_no_der)
    - Optimization with DER      (opt_der)

using the databases:

    5_vessels_no_opt_no_der.db
    5_vessels_opt_no_der.db
    5_vessels_opt_der.db
    10_vessels_no_opt_no_der.db
    10_vessels_opt_no_der.db
    10_vessels_opt_der.db
    20_vessels_no_opt_no_der.db
    20_vessels_opt_no_der.db
    20_vessels_opt_der.db

For each fleet size, it will:
    - Compare grid import, PV production and total charger load over time
    - Compute reliability (on‑time, delayed, cancelled trips)
    - Summarise peak power, energy from grid and PV, and reliability metrics

Outputs:
    tests/output/5/
    tests/output/10/
    tests/output/20/

Each folder contains:
    - power_timeseries_<n>_vessels.png
    - energy_breakdown_<n>_vessels.png
    - reliability_<n>_vessels.png
    - summary_<n>_vessels.csv
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

SCENARIOS: Dict[str, str] = {
    "no_opt_no_der": "No Optimization, No DER",
    "opt_no_der": "Optimization, No DER",
    "opt_der": "Optimization + DER",
}

VESSEL_COUNTS: List[int] = [5, 10, 20]

# Trip schedule times (for reference / annotations if needed)
TRIP_DEPARTURE_HOURS: List[int] = [9, 14]

# Colors per scenario (consistent across plots)
COLORS: Dict[str, str] = {
    "no_opt_no_der": "#e74c3c",  # Red
    "opt_no_der": "#3498db",     # Blue
    "opt_der": "#2ecc71",        # Green
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def project_root() -> Path:
    """Return project root assuming this file lives in tests/port_eletrification_studies/."""
    return Path(__file__).parent.parent.parent


def get_db_path(vessels: int, scenario: str) -> Path:
    """
    Get database path for a given fleet size and scenario.

    Expected pattern (in project root):
        <vessels>_vessels_<scenario>.db
    """
    return project_root() / f"{vessels}_vessels_{scenario}.db"


def ensure_output_dir(vessels: int) -> Path:
    """
    Ensure output directory tests/output/<vessels> exists and return it.
    """
    base = Path(__file__).parent.parent / "output" / str(vessels)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_dt_hours(index: pd.Series) -> float:
    """Infer timestep in hours from a datetime index; fall back to 0.25 h."""
    if len(index) < 2:
        return 0.25
    delta = (index.iloc[1] - index.iloc[0]).total_seconds() / 3600.0
    return float(delta) if delta > 0 else 0.25


# --------------------------------------------------------------------------- #
#  Data loading helpers
# --------------------------------------------------------------------------- #

def load_port_timeseries(db_path: Path) -> pd.DataFrame:
    """
    Load port‑level power time‑series from a database.

    Returns a DataFrame with:
        timestamp (datetime64[ns])
        consumption      (power_active_consumption, kW)
        production       (power_active_production,  kW)  [PV / renewables]
        import_power     (power_active_import,      kW)
        export_power     (power_active_export,      kW)
        contracted_power (contracted_power,         kW)
        bess_discharge   (bess_discharge,           kW)
        bess_charge      (bess_charge,              kW)
    """
    if not db_path.exists():
        print(f"  [WARN] Database not found: {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))

    # Get metric IDs of interest
    metric_names = [
        "power_active_consumption",
        "power_active_production",
        "power_active_import",
        "power_active_export",
        "contracted_power",
        "bess_discharge",
        "bess_charge",
    ]
    metrics = pd.read_sql_query(
        f"""
        SELECT metric_id, metric_name
        FROM metric
        WHERE metric_name IN ({",".join([repr(m) for m in metric_names])})
        """,
        conn,
    )
    if metrics.empty:
        conn.close()
        return pd.DataFrame()

    metric_id_by_name = dict(zip(metrics["metric_name"], metrics["metric_id"]))

    # Get port source id
    port_source = pd.read_sql_query(
        "SELECT source_id FROM source WHERE source_type = 'port' LIMIT 1",
        conn,
    )
    if port_source.empty:
        conn.close()
        return pd.DataFrame()
    port_id = int(port_source.iloc[0, 0])

    metric_ids = tuple(metric_id_by_name.values())
    df = pd.read_sql_query(
        f"""
        SELECT timestamp, metric_id, CAST(value AS FLOAT) AS value
        FROM measurements
        WHERE source_id = {port_id}
          AND metric_id IN ({",".join(str(m) for m in metric_ids)})
        ORDER BY timestamp
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Pivot into wide format
    pivot = df.pivot_table(
        index="timestamp",
        columns="metric_id",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.sort_index()

    # Map metric_id columns back to semantic names
    col_map = {
        metric_id_by_name["power_active_consumption"]: "consumption",
        metric_id_by_name["power_active_production"]: "production",
        metric_id_by_name["power_active_import"]: "import_power",
        metric_id_by_name["power_active_export"]: "export_power",
        metric_id_by_name["contracted_power"]: "contracted_power",
        metric_id_by_name["bess_discharge"]: "bess_discharge",
        metric_id_by_name["bess_charge"]: "bess_charge",
    }
    pivot = pivot.rename(columns=col_map)

    # Ensure all expected columns exist
    for col in [
        "consumption",
        "production",
        "import_power",
        "export_power",
        "contracted_power",
        "bess_discharge",
        "bess_charge",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot.reset_index(inplace=True)
    pivot["timestamp"] = pd.to_datetime(pivot["timestamp"])
    return pivot


def load_boat_state_data(db_path: Path) -> pd.DataFrame:
    """
    Load boat state data (sailing = 1.0, not sailing = 0.0) for reliability.
    """
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))

    # Get 'state' metric id
    state_result = pd.read_sql_query(
        "SELECT metric_id FROM metric WHERE metric_name = 'state'", conn
    )
    if state_result.empty:
        conn.close()
        return pd.DataFrame()
    state_id = int(state_result.iloc[0, 0])

    # Boats
    boat_sources = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'boat'",
        conn,
    )

    all_data: List[pd.DataFrame] = []
    for _, row in boat_sources.iterrows():
        df = pd.read_sql_query(
            f"""
            SELECT timestamp,
                   CAST(value AS FLOAT) AS state,
                   '{row["source_name"]}' AS boat
            FROM measurements
            WHERE source_id = {int(row["source_id"])}
              AND metric_id = {state_id}
            ORDER BY timestamp
            """,
            conn,
        )
        all_data.append(df)

    conn.close()

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result


# --------------------------------------------------------------------------- #
#  Reliability analysis
# --------------------------------------------------------------------------- #

@dataclass
class ReliabilityStats:
    total_trips: int
    on_time_trips: int
    delayed_trips: int
    cancelled_trips: int

    @property
    def delay_rate(self) -> float:
        return (self.delayed_trips / self.total_trips * 100) if self.total_trips else 0.0

    @property
    def cancel_rate(self) -> float:
        return (self.cancelled_trips / self.total_trips * 100) if self.total_trips else 0.0


def analyze_reliability(db_path: Path, num_vessels: int) -> ReliabilityStats:
    """
    Analyse trip reliability based on boat state changes.

    Heuristic:
        - Trip = transition from state 0 -> 1 (boat starts sailing)
        - Morning slot: around 09:00
        - Afternoon slot: around 14:00
        - On‑time if departure within first 15 minutes of slot
        - Delayed if departure later in the day
        - Cancelled if no departure detected for that slot
    """
    total_trips = num_vessels * 2  # two trips per boat (morning, afternoon)

    if not db_path.exists():
        return ReliabilityStats(
            total_trips=total_trips,
            on_time_trips=0,
            delayed_trips=0,
            cancelled_trips=total_trips,
        )

    state_data = load_boat_state_data(db_path)
    if state_data.empty:
        return ReliabilityStats(
            total_trips=total_trips,
            on_time_trips=0,
            delayed_trips=0,
            cancelled_trips=total_trips,
        )

    boats = state_data["boat"].unique()

    on_time = 0
    delayed = 0
    cancelled = 0

    for boat in boats:
        boat_df = state_data[state_data["boat"] == boat].sort_values("timestamp")
        vals = boat_df["state"].values
        ts = boat_df["timestamp"].values

        # Find trip start times (0 -> 1)
        starts: List[datetime] = []
        for i in range(1, len(vals)):
            if vals[i] == 1.0 and vals[i - 1] == 0.0:
                starts.append(pd.to_datetime(ts[i]))

        morning_found = False
        afternoon_found = False

        for dep in starts:
            h, m = dep.hour, dep.minute

            # Morning (slot at 09:00)
            if not morning_found:
                if h == 9 and m < 15:
                    on_time += 1
                    morning_found = True
                    continue
                if 9 <= h < 14:
                    delayed += 1
                    morning_found = True
                    continue

            # Afternoon (slot at 14:00)
            if not afternoon_found:
                if h == 14 and m < 15:
                    on_time += 1
                    afternoon_found = True
                    continue
                if 14 <= h < 18:
                    delayed += 1
                    afternoon_found = True
                    continue

        if not morning_found:
            cancelled += 1
        if not afternoon_found:
            cancelled += 1

    return ReliabilityStats(
        total_trips=total_trips,
        on_time_trips=on_time,
        delayed_trips=delayed,
        cancelled_trips=cancelled,
    )


# --------------------------------------------------------------------------- #
#  Plotting helpers
# --------------------------------------------------------------------------- #

def plot_power_timeseries(
    vessels: int,
    scenario_ts: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Plot charger consumption vs contracted power for all three scenarios
    on a single set of axes for a given fleet size.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for scenario, df in scenario_ts.items():
        if df.empty:
            continue
        label = SCENARIOS.get(scenario, scenario)
        color = COLORS.get(scenario, None)

        ax.plot(
            df["timestamp"],
            df["consumption"],
            label=f"{label} – load",
            color=color,
            linewidth=1.5,
        )

    # Contracted power (assume same across scenarios – use first non‑empty)
    contracted = None
    for df in scenario_ts.values():
        if not df.empty and "contracted_power" in df.columns:
            contracted = float(df["contracted_power"].iloc[0])
            break

    if contracted is not None:
        ax.axhline(
            y=contracted,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Contracted power ({contracted:.0f} kW)",
        )

    ax.set_title(f"Port load vs contracted power – {vessels} vessels")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = output_dir / f"power_timeseries_{vessels}_vessels.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_energy_breakdown(
    vessels: int,
    scenario_ts: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    For each scenario, compute total daily energy from:
        - grid import
        - PV / renewables
        - BESS discharge

    and display as grouped bars (kWh). This highlights the effect of
    optimization + DER on grid dependence and renewable utilisation.
    """
    scenario_names: List[str] = []
    grid_energy: List[float] = []
    pv_energy: List[float] = []
    bess_discharge_energy: List[float] = []

    for scenario, df in scenario_ts.items():
        if df.empty:
            continue
        dt_h = _safe_dt_hours(df["timestamp"])

        e_grid = float((df["import_power"] * dt_h).sum())
        e_pv = float((df["production"] * dt_h).sum())
        e_bess_dis = float((df["bess_discharge"] * dt_h).sum())

        scenario_names.append(SCENARIOS.get(scenario, scenario))
        grid_energy.append(e_grid)
        pv_energy.append(e_pv)
        bess_discharge_energy.append(e_bess_dis)

    if not scenario_names:
        return

    x = np.arange(len(scenario_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, grid_energy, width, label="Grid import (kWh)", color="#e74c3c")
    ax.bar(x, pv_energy, width, label="PV production (kWh)", color="#3498db")
    ax.bar(
        x + width,
        bess_discharge_energy,
        width,
        label="BESS discharge (kWh)",
        color="#9b59b6",
    )

    ax.set_title(f"Energy breakdown by scenario – {vessels} vessels")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Energy (kWh)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = output_dir / f"energy_breakdown_{vessels}_vessels.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_reliability(
    vessels: int,
    reliability_by_scenario: Dict[str, ReliabilityStats],
    output_dir: Path,
) -> None:
    """
    Plot stacked bars of on‑time / delayed / cancelled trips for each scenario.
    """
    scenario_keys = [s for s in SCENARIOS.keys() if s in reliability_by_scenario]
    if not scenario_keys:
        return

    labels = [SCENARIOS[s].replace(", ", "\n") for s in scenario_keys]
    on_time = [reliability_by_scenario[s].on_time_trips for s in scenario_keys]
    delayed = [reliability_by_scenario[s].delayed_trips for s in scenario_keys]
    cancelled = [reliability_by_scenario[s].cancelled_trips for s in scenario_keys]

    x = np.arange(len(scenario_keys))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_on = ax.bar(x, on_time, width, label="On‑time", color="#2ecc71")
    bars_del = ax.bar(x, delayed, width, bottom=on_time, label="Delayed", color="#f1c40f")
    bottom_cancel = [on_time[i] + delayed[i] for i in range(len(on_time))]
    bars_can = ax.bar(
        x,
        cancelled,
        width,
        bottom=bottom_cancel,
        label="Cancelled",
        color="#e74c3c",
    )

    ax.set_title(f"Trip reliability by scenario – {vessels} vessels")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Number of trips")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bars in [bars_on, bars_del, bars_can]:
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out_path = output_dir / f"reliability_{vessels}_vessels.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
#  Summary table
# --------------------------------------------------------------------------- #

def build_summary_row(
    vessels: int,
    scenario: str,
    ts: pd.DataFrame,
    rel: ReliabilityStats,
) -> Dict[str, object]:
    """
    Build a single summary row with key metrics for CSV.
    """
    if ts.empty:
        return {
            "Vessels": vessels,
            "Scenario": SCENARIOS.get(scenario, scenario),
            "Total trips": rel.total_trips,
            "On‑time trips": rel.on_time_trips,
            "Delayed trips": rel.delayed_trips,
            "Cancelled trips": rel.cancelled_trips,
            "Delay rate (%)": rel.delay_rate,
            "Cancel rate (%)": rel.cancel_rate,
            "Peak charger load (kW)": 0.0,
            "Peak grid import (kW)": 0.0,
            "Energy from grid (kWh)": 0.0,
            "Energy from PV (kWh)": 0.0,
            "Energy from BESS discharge (kWh)": 0.0,
            "Approx. renewable share (%)": 0.0,
        }

    dt_h = _safe_dt_hours(ts["timestamp"])
    peak_load = float(ts["consumption"].max())
    peak_import = float(ts["import_power"].max())
    e_load = float((ts["consumption"] * dt_h).sum())
    e_grid = float((ts["import_power"] * dt_h).sum())
    e_pv = float((ts["production"] * dt_h).sum())
    e_bess_dis = float((ts["bess_discharge"] * dt_h).sum())

    if e_load > 0:
        # Everything that did not come from grid is counted as "non‑grid" energy
        renewable_share = max(0.0, 1.0 - e_grid / e_load) * 100.0
    else:
        renewable_share = 0.0

    return {
        "Vessels": vessels,
        "Scenario": SCENARIOS.get(scenario, scenario),
        "Total trips": rel.total_trips,
        "On‑time trips": rel.on_time_trips,
        "Delayed trips": rel.delayed_trips,
        "Cancelled trips": rel.cancelled_trips,
        "Delay rate (%)": round(rel.delay_rate, 1),
        "Cancel rate (%)": round(rel.cancel_rate, 1),
        "Peak charger load (kW)": round(peak_load, 1),
        "Peak grid import (kW)": round(peak_import, 1),
        "Energy from grid (kWh)": round(e_grid, 1),
        "Energy from PV (kWh)": round(e_pv, 1),
        "Energy from BESS discharge (kWh)": round(e_bess_dis, 1),
        "Approx. renewable share (%)": round(renewable_share, 1),
    }


def save_summary_table(
    vessels: int,
    scenario_ts: Dict[str, pd.DataFrame],
    reliability_by_scenario: Dict[str, ReliabilityStats],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Build and save the per‑fleet summary CSV.
    """
    rows: List[Dict[str, object]] = []
    for scenario in SCENARIOS.keys():
        ts = scenario_ts.get(scenario, pd.DataFrame())
        rel = reliability_by_scenario.get(
            scenario, ReliabilityStats(0, 0, 0, 0)
        )
        rows.append(build_summary_row(vessels, scenario, ts, rel))

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"summary_{vessels}_vessels.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return df


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("Port electrification – scenario comparison")
    print("=" * 70)
    print(f"Project root: {project_root()}")

    for vessels in VESSEL_COUNTS:
        print(f"\n--- {vessels} vessels: comparing scenarios ---")
        output_dir = ensure_output_dir(vessels)
        print(f"  Output directory: {output_dir}")

        scenario_ts: Dict[str, pd.DataFrame] = {}
        reliability_by_scenario: Dict[str, ReliabilityStats] = {}

        for scenario in SCENARIOS.keys():
            db_path = get_db_path(vessels, scenario)
            print(f"  Loading {db_path.name} ({scenario}) ...")

            ts = load_port_timeseries(db_path)
            rel = analyze_reliability(db_path, vessels)

            if ts.empty:
                print("    [WARN] No port timeseries data found.")
            else:
                print(
                    f"    Loaded {len(ts)} timesteps. "
                    f"Peak load = {ts['consumption'].max():.1f} kW, "
                    f"Peak grid import = {ts['import_power'].max():.1f} kW"
                )

            print(
                f"    Trips – on‑time: {rel.on_time_trips}, "
                f"delayed: {rel.delayed_trips}, "
                f"cancelled: {rel.cancelled_trips}"
            )

            scenario_ts[scenario] = ts
            reliability_by_scenario[scenario] = rel

        # Plots for this fleet size
        print("  Creating plots...")
        plot_power_timeseries(vessels, scenario_ts, output_dir)
        plot_energy_breakdown(vessels, scenario_ts, output_dir)
        plot_reliability(vessels, reliability_by_scenario, output_dir)

        # Summary CSV
        print("  Creating summary table...")
        save_summary_table(vessels, scenario_ts, reliability_by_scenario, output_dir)

    print("\nAll comparisons completed.")
    print("See per‑fleet results under tests/output/<vessels>/")


if __name__ == "__main__":
    main()

