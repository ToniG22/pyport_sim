"""System Integration Validation Scenario.

Scenario:
    - One electric vessel connected to a dedicated charger
    - Active PV generation with BESS
    - Fixed grid connection with contracted power limit

Objective:
    Verify correct energy flow consistency, module interaction, and global
    power balance across a complete 24-hour simulation cycle.

Validation criteria:
    - Power balance at each timestep:
          P_grid + P_PV + P_BESS = P_load
      where:
          P_grid = P_import - P_export  (port metrics)
          P_PV   = power_active_production (port metric)
          P_BESS = bess_discharge - bess_charge (port metrics)
          P_load = power_active_consumption (port metric)
    - Compliance with component constraints:
          - Grid import ≤ contracted power
          - SOC bounds respected for vessel and BESS
          - BESS power limited by its rated charge/discharge power

This script both RUNS the simulation (writing an SQLite DB) and then
QUERIES the DB to perform the above validations and generate plots/CSVs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models import Port, Boat, Charger, PV, BESS, BESSControlStrategy
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "system_integration_validation.db"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "sim"

# ---------------------------------------------------------------------------
# Scenario configuration helpers
# ---------------------------------------------------------------------------


def build_port_scenario() -> Port:
    """Create port + components for the integration scenario."""
    port = Port(
        name="Funchal_System_Test",
        contracted_power=80,  # kW grid limit
        lat=32.64542,
        lon=-16.90841,
        tariff_path="assets/tariff/default_tariff.json",
    )

    # One vessel
    boat = Boat(
        name="Boat",
        motor_power=120,
        weight=3500,
        length=9.0,
        battery_capacity=120.0,
        range_speed=18.0,
        soc=0.20,
    )

    # Dedicated charger
    charger = Charger(name="Charger", max_power=22.0, efficiency=0.95)

    # PV + BESS
    pv_system = PV(
        name="PV",
        capacity=20.0,  # kW DC
        tilt=30.0,
        azimuth=180.0,
        latitude=port.lat,
        longitude=port.lon,
    )

    bess = BESS(
        name="BESS",
        capacity=100.0,  # kWh
        max_charge_power=30.0,  # kW
        max_discharge_power=30.0,  # kW
        efficiency=0.90,
        soc_min=0.10,
        soc_max=0.90,
        initial_soc=0.5,
        control_strategy=BESSControlStrategy.DEFAULT,
    )

    port.add_boat(boat)
    port.add_charger(charger)
    port.add_pv(pv_system)
    port.add_bess(bess)

    return port


def run_simulation() -> None:
    """Run a 24h batch simulation and populate the SQLite DB."""
    settings = Settings(
        timestep=900,  # 15 minutes
        mode=SimulationMode.BATCH,
        db_path=str(DB_PATH),
        use_optimizer=False,
        power_limit_mode=True,
    )

    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()
    db_manager.initialize_default_metrics()

    port = build_port_scenario()
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date="2025-06-20",  # clear-sky summer day
        days=1,
    )
    sim.run()


# ---------------------------------------------------------------------------
# DB-based validation & plotting
# ---------------------------------------------------------------------------


def _load_port_timeseries(db_path: Path) -> pd.DataFrame:
    """Load port-level power metrics and derive P_grid, P_PV, P_BESS, P_load."""
    conn = sqlite3.connect(db_path)

    # Port source
    port_src = pd.read_sql_query(
        "SELECT source_id FROM source WHERE source_type = 'port'", conn
    )
    if port_src.empty:
        conn.close()
        raise RuntimeError("No port source found in DB.")
    port_id = int(port_src.iloc[0, 0])

    # Metric mapping
    metrics = pd.read_sql_query(
        "SELECT metric_id, metric_name FROM metric", conn
    )
    metric_map = dict(zip(metrics["metric_name"], metrics["metric_id"]))

    def fetch_metric(name: str) -> pd.DataFrame:
        mid = metric_map.get(name)
        if mid is None:
            return pd.DataFrame(columns=["timestamp", name])
        q = f"""
            SELECT timestamp, CAST(value AS FLOAT) AS {name}
            FROM measurements
            WHERE source_id = {port_id} AND metric_id = {mid}
            ORDER BY timestamp
        """
        return pd.read_sql_query(q, conn)

    df_load = fetch_metric("power_active_consumption")
    df_pv = fetch_metric("power_active_production")
    df_bess_dis = fetch_metric("bess_discharge")
    df_bess_ch = fetch_metric("bess_charge")
    df_imp = fetch_metric("power_active_import")
    df_exp = fetch_metric("power_active_export")

    conn.close()

    # Merge on timestamp
    dfs = [df_load, df_pv, df_bess_dis, df_bess_ch, df_imp, df_exp]
    ts = dfs[0]
    for extra in dfs[1:]:
        ts = ts.merge(extra, on="timestamp", how="outer")

    ts = ts.sort_values("timestamp").reset_index(drop=True)
    ts = ts.fillna(0.0)

    # Derived quantities
    ts["P_load"] = ts["power_active_consumption"]
    ts["P_PV"] = ts["power_active_production"]
    ts["P_BESS"] = ts["bess_discharge"] - ts["bess_charge"]
    ts["P_grid"] = ts["power_active_import"] - ts["power_active_export"]

    return ts


def _load_soc_timeseries(db_path: Path) -> pd.DataFrame:
    """Load SOC and BESS power for constraint checks."""
    conn = sqlite3.connect(db_path)

    metrics = pd.read_sql_query(
        "SELECT metric_id, metric_name FROM metric", conn
    )
    metric_map = dict(zip(metrics["metric_name"], metrics["metric_id"]))
    soc_id = metric_map.get("soc")
    power_id = metric_map.get("power_active")

    boats = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'boat'", conn
    )
    besses = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'bess'", conn
    )

    series = []

    def fetch_soc_power(source_row, src_type: str):
        sid = int(source_row["source_id"])
        sname = source_row["source_name"]
        q_soc = f"""
            SELECT timestamp, CAST(value AS FLOAT) AS soc_pct
            FROM measurements
            WHERE source_id = {sid} AND metric_id = {soc_id}
            ORDER BY timestamp
        """
        df_soc = pd.read_sql_query(q_soc, conn)
        df_soc["source_name"] = sname
        df_soc["source_type"] = src_type
        if power_id is not None and src_type == "bess":
            q_p = f"""
                SELECT timestamp, CAST(value AS FLOAT) AS power_kw
                FROM measurements
                WHERE source_id = {sid} AND metric_id = {power_id}
                ORDER BY timestamp
            """
            df_p = pd.read_sql_query(q_p, conn)
            df = df_soc.merge(df_p, on="timestamp", how="left")
        else:
            df = df_soc
        series.append(df)

    for _, row in boats.iterrows():
        fetch_soc_power(row, "boat")
    for _, row in besses.iterrows():
        fetch_soc_power(row, "bess")

    conn.close()
    if not series:
        return pd.DataFrame()
    return pd.concat(series, ignore_index=True)


def validate_and_plot_from_db() -> None:
    """Perform DB-based validations and generate plots/CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = _load_port_timeseries(DB_PATH)
    soc_df = _load_soc_timeseries(DB_PATH)

    # ------------------------------------------------------------------
    # Validation 1: Power balance P_grid + P_PV + P_BESS = P_load
    # ------------------------------------------------------------------
    ts["balance_error"] = ts["P_grid"] + ts["P_PV"] + ts["P_BESS"] - ts["P_load"]
    max_abs_err = ts["balance_error"].abs().max()
    print(f"Max absolute power balance error: {max_abs_err:.6e} kW")
    assert max_abs_err < 1e-6 or max_abs_err < 1e-3, "Power balance error too large."

    # ------------------------------------------------------------------
    # Validation 2: Grid import within contracted power
    # ------------------------------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    cp_df = pd.read_sql_query(
        "SELECT DISTINCT value FROM measurements "
        "JOIN metric USING(metric_id) "
        "JOIN source USING(source_id) "
        "WHERE metric.metric_name = 'contracted_power' "
        "AND source.source_type = 'port'",
        conn,
    )
    conn.close()
    contracted = float(cp_df.iloc[0, 0]) if not cp_df.empty else 80.0
    max_import = ts["power_active_import"].max()
    print(f"Max grid import: {max_import:.3f} kW (contracted {contracted:.3f} kW)")
    assert max_import <= contracted + 1e-6

    # ------------------------------------------------------------------
    # Validation 3: SOC limits & BESS power rating
    # ------------------------------------------------------------------
    if not soc_df.empty:
        bess_soc = soc_df[soc_df["source_type"] == "bess"]
        boat_soc = soc_df[soc_df["source_type"] == "boat"]

        if not bess_soc.empty:
            min_bess_soc = bess_soc["soc_pct"].min()
            max_bess_soc = bess_soc["soc_pct"].max()
            print(
                f"BESS SOC range: {min_bess_soc:.2f}% – {max_bess_soc:.2f}% "
                "(expected within 10–90%)"
            )
            assert 9.9 <= min_bess_soc <= 90.1
            assert 9.9 <= max_bess_soc <= 90.1

            if "power_kw" in bess_soc.columns:
                max_bess_p = bess_soc["power_kw"].abs().max()
                print(f"Max |BESS power|: {max_bess_p:.3f} kW (limit 30 kW)")
                assert max_bess_p <= 30.0 + 1e-6

        if not boat_soc.empty:
            min_boat_soc = boat_soc["soc_pct"].min()
            max_boat_soc = boat_soc["soc_pct"].max()
            print(f"Boat SOC range: {min_boat_soc:.2f}% – {max_boat_soc:.2f}%")
            assert 0.0 <= min_boat_soc <= 100.0
            assert 0.0 <= max_boat_soc <= 100.0

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "system_profile_db.csv"
    ts.to_csv(csv_path, index=False)
    print(f"✓ DB-based system profile CSV saved to: {csv_path}")

    # ------------------------------------------------------------------
    # Plots (power flows + SOC)
    # ------------------------------------------------------------------
    times = pd.to_datetime(ts["timestamp"])
    hours = times.dt.hour + times.dt.minute / 60.0

    fig, (ax_p, ax_soc) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_p.plot(hours, ts["P_load"], "k-", linewidth=2, label="Load (P_load)")
    ax_p.plot(hours, ts["P_PV"], "gold", linewidth=2, label="P_PV")
    ax_p.plot(hours, ts["P_BESS"], "purple", linewidth=1.5, label="P_BESS (discharge−charge)")
    ax_p.plot(hours, ts["P_grid"], "r--", linewidth=1.5, label="P_grid (import−export)")
    ax_p.set_ylabel("Power (kW)")
    ax_p.set_title("System power balance from DB")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper right", fontsize=9)

    if not soc_df.empty:
        soc_df_local = soc_df.copy()
        soc_df_local["hour"] = pd.to_datetime(soc_df_local["timestamp"]).dt.hour + (
            pd.to_datetime(soc_df_local["timestamp"]).dt.minute / 60.0
        )
        for sname, grp in soc_df_local.groupby("source_name"):
            style = "-" if grp["source_type"].iloc[0] == "boat" else "--"
            ax_soc.plot(
                grp["hour"],
                grp["soc_pct"],
                style,
                linewidth=2,
                label=f"{sname} SOC",
            )

    ax_soc.set_xlabel("Hour of day")
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_ylim(0, 100)
    ax_soc.grid(True, alpha=0.3)
    ax_soc.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "system_profile_db.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ DB-based system profile plot saved to: {plot_path}")


def main():
    print("=" * 80)
    print("SYSTEM INTEGRATION VALIDATION - DB-BASED CHECK")
    print("=" * 80)

    run_simulation()
    print(f"\nSimulation completed. Results stored in: {DB_PATH}")

    validate_and_plot_from_db()

    print("\n" + "=" * 80)
    print("All DB-based validations completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()

