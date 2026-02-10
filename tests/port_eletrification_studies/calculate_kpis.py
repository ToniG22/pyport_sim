"""
Calculate KPIs for Port Electrification Study

KPI 1 – Total Cost of Operation (€)
KPI 2 – Cost per Successful Trip (€/trip)
KPI 3 – Peak Power Demand (kW)
KPI 4 – Self-Consumption Rate (% of PV consumed locally)
KPI 5 – Self-Sufficiency Rate (% of demand from PV)
KPI 6 – Trip Reliability (% of trips completed)
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
from tabulate import tabulate

# ============================================================================
# CONFIGURATION
# ============================================================================

# Electricity tariff (€/kWh) - simplified time-of-use
TARIFF_PEAK = 0.25  # Peak hours (8:00-20:00)
TARIFF_OFF_PEAK = 0.12  # Off-peak hours (20:00-8:00)

# Contracted power for baseline comparison
CONTRACTED_POWER = 80  # kW

# Scenarios
SCENARIOS = [
    ("no_opt_no_der", "No Optimization, No DER", False),
    ("opt_no_der", "Optimization, No DER", False),
    ("opt_der", "Optimization + DER", True),
]

VESSEL_COUNTS = [5, 10, 20]


def get_db_path(vessels: int, scenario: str) -> str:
    """Get database path for a scenario."""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / f"{vessels}_vessels_{scenario}.db")


def get_electricity_cost(timestamp_str: str) -> float:
    """Get electricity cost based on time of day."""
    hour = int(timestamp_str.split(" ")[1].split(":")[0])
    if 8 <= hour < 20:
        return TARIFF_PEAK
    return TARIFF_OFF_PEAK


def analyze_trip_reliability(db_path: str, num_vessels: int) -> dict:
    """Analyze trip reliability from boat state data."""
    total_trips = num_vessels * 2  # 2 trips per vessel per day
    
    if not os.path.exists(db_path):
        return {'on_time': 0, 'delayed': 0, 'cancelled': total_trips, 'total': total_trips}
    
    conn = sqlite3.connect(db_path)
    
    # Get state metric
    state_result = pd.read_sql_query(
        "SELECT metric_id FROM metric WHERE metric_name = 'state'", conn
    )
    if state_result.empty:
        conn.close()
        return {'on_time': 0, 'delayed': 0, 'cancelled': total_trips, 'total': total_trips}
    
    state_id = state_result.iloc[0, 0]
    
    # Get boat sources
    boats = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'boat'", conn
    )
    
    on_time = 0
    delayed = 0
    cancelled = 0
    
    for _, boat in boats.iterrows():
        query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as state
            FROM measurements
            WHERE source_id = {boat['source_id']} AND metric_id = {state_id}
            ORDER BY timestamp
        """
        states = pd.read_sql_query(query, conn)
        
        if states.empty:
            cancelled += 2
            continue
        
        states['timestamp'] = pd.to_datetime(states['timestamp'])
        
        # Find trip starts (state changes to 1.0 = sailing)
        trip_starts = []
        prev_state = 0
        for _, row in states.iterrows():
            if row['state'] == 1.0 and prev_state == 0.0:
                trip_starts.append(row['timestamp'])
            prev_state = row['state']
        
        # Analyze each trip slot
        morning_found = False
        afternoon_found = False
        
        for ts in trip_starts:
            hour = ts.hour
            minute = ts.minute
            
            if 9 <= hour < 12:
                if hour == 9 and minute < 15:
                    on_time += 1
                else:
                    delayed += 1
                morning_found = True
            elif 14 <= hour < 18:
                if hour == 14 and minute < 15:
                    on_time += 1
                else:
                    delayed += 1
                afternoon_found = True
        
        if not morning_found:
            cancelled += 1
        if not afternoon_found:
            cancelled += 1
    
    conn.close()
    
    return {
        'on_time': on_time,
        'delayed': delayed,
        'cancelled': cancelled,
        'total': total_trips,
        'reliability': ((on_time + delayed) / total_trips * 100) if total_trips > 0 else 0
    }


def calculate_kpis(db_path: str, has_der: bool = False) -> dict:
    """Calculate all KPIs for a scenario."""
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    
    # Get port source
    port_source = pd.read_sql_query(
        "SELECT source_id FROM source WHERE source_type = 'port'", conn
    )
    if port_source.empty:
        conn.close()
        return None
    port_id = port_source.iloc[0, 0]
    
    # Get metrics
    metrics = pd.read_sql_query("SELECT metric_id, metric_name FROM metric", conn)
    metric_map = dict(zip(metrics['metric_name'], metrics['metric_id']))
    
    timestep_hours = 0.25  # 15 minutes
    
    # =========================================================================
    # Get consumption and import data
    # =========================================================================
    
    consumption_query = f"""
        SELECT timestamp, CAST(value AS FLOAT) as power_kw
        FROM measurements
        WHERE source_id = {port_id} 
        AND metric_id = {metric_map.get('power_active_consumption', -1)}
        ORDER BY timestamp
    """
    consumption_df = pd.read_sql_query(consumption_query, conn)
    
    if consumption_df.empty:
        conn.close()
        return None
    
    consumption_df['energy_kwh'] = consumption_df['power_kw'] * timestep_hours
    total_consumption_kwh = consumption_df['energy_kwh'].sum()
    peak_consumption = consumption_df['power_kw'].max()
    
    # Grid import
    import_query = f"""
        SELECT timestamp, CAST(value AS FLOAT) as power_kw
        FROM measurements
        WHERE source_id = {port_id} 
        AND metric_id = {metric_map.get('power_active_import', -1)}
        ORDER BY timestamp
    """
    import_df = pd.read_sql_query(import_query, conn)
    
    if not import_df.empty:
        import_df['energy_kwh'] = import_df['power_kw'] * timestep_hours
        import_df['cost_eur'] = import_df.apply(
            lambda row: row['energy_kwh'] * get_electricity_cost(row['timestamp']),
            axis=1
        )
        total_grid_import_kwh = import_df['energy_kwh'].sum()
        total_cost_eur = import_df['cost_eur'].sum()
        peak_import = import_df['power_kw'].max()
    else:
        total_grid_import_kwh = total_consumption_kwh
        consumption_df['cost_eur'] = consumption_df.apply(
            lambda row: row['energy_kwh'] * get_electricity_cost(row['timestamp']),
            axis=1
        )
        total_cost_eur = consumption_df['cost_eur'].sum()
        peak_import = peak_consumption
    
    # =========================================================================
    # PV data
    # =========================================================================
    
    pv_generated_kwh = 0
    pv_consumed_kwh = 0
    self_consumption_rate = 0
    self_sufficiency_rate = 0
    
    if has_der:
        # Get PV production from port metrics
        pv_query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as power_kw
            FROM measurements
            WHERE source_id = {port_id} 
            AND metric_id = {metric_map.get('power_active_production', -1)}
            ORDER BY timestamp
        """
        pv_df = pd.read_sql_query(pv_query, conn)
        
        if not pv_df.empty:
            pv_df['energy_kwh'] = pv_df['power_kw'] * timestep_hours
            pv_generated_kwh = pv_df['energy_kwh'].sum()
        
        # Self-consumption = consumption - grid import (what came from PV)
        pv_consumed_kwh = max(0, total_consumption_kwh - total_grid_import_kwh)
        
        if pv_generated_kwh > 0:
            self_consumption_rate = min(100, (pv_consumed_kwh / pv_generated_kwh) * 100)
        
        if total_consumption_kwh > 0:
            self_sufficiency_rate = (pv_consumed_kwh / total_consumption_kwh) * 100
    
    conn.close()
    
    return {
        'total_consumption_kwh': total_consumption_kwh,
        'total_grid_import_kwh': total_grid_import_kwh,
        'total_cost_eur': total_cost_eur,
        'peak_consumption_kw': peak_consumption,
        'peak_import_kw': peak_import,
        'pv_generated_kwh': pv_generated_kwh,
        'pv_consumed_kwh': pv_consumed_kwh,
        'self_consumption_rate': self_consumption_rate,
        'self_sufficiency_rate': self_sufficiency_rate,
    }


def main():
    print("=" * 100)
    print("PORT ELECTRIFICATION STUDY - KPI ANALYSIS")
    print("=" * 100)
    
    results = []
    
    for vessels in VESSEL_COUNTS:
        for scenario_key, scenario_name, has_der in SCENARIOS:
            db_path = get_db_path(vessels, scenario_key)
            
            kpis = calculate_kpis(db_path, has_der)
            reliability = analyze_trip_reliability(db_path, vessels)
            
            if kpis is None:
                print(f"Warning: Could not load {vessels} vessels, {scenario_name}")
                continue
            
            # Cost per successful trip
            successful_trips = reliability['on_time'] + reliability['delayed']
            cost_per_trip = kpis['total_cost_eur'] / successful_trips if successful_trips > 0 else float('inf')
            
            results.append({
                'Vessels': vessels,
                'Scenario': scenario_name,
                'On-Time': reliability['on_time'],
                'Delayed': reliability['delayed'],
                'Cancelled': reliability['cancelled'],
                'KPI 6: Reliability (%)': round(reliability['reliability'], 1),
                'Energy (kWh)': round(kpis['total_consumption_kwh'], 1),
                'KPI 1: Cost (€)': round(kpis['total_cost_eur'], 2),
                'KPI 2: €/Trip': round(cost_per_trip, 2),
                'KPI 3: Peak (kW)': round(kpis['peak_import_kw'], 1),
                'PV (kWh)': round(kpis['pv_generated_kwh'], 1) if has_der else '-',
                'KPI 4: Self-Cons (%)': round(kpis['self_consumption_rate'], 1) if has_der else '-',
                'KPI 5: Self-Suff (%)': round(kpis['self_sufficiency_rate'], 1) if has_der else '-',
            })
    
    df = pd.DataFrame(results)
    
    # =========================================================================
    # TABLE 1: RELIABILITY METRICS
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 1: TRIP RELIABILITY PERFORMANCE")
    print("=" * 100)
    cols1 = ['Vessels', 'Scenario', 'On-Time', 'Delayed', 'Cancelled', 'KPI 6: Reliability (%)']
    print(tabulate(df[cols1], headers='keys', tablefmt='grid', showindex=False))
    
    # =========================================================================
    # TABLE 2: ECONOMIC METRICS
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: ECONOMIC PERFORMANCE")
    print("=" * 100)
    cols2 = ['Vessels', 'Scenario', 'Energy (kWh)', 'KPI 1: Cost (€)', 'KPI 2: €/Trip', 'KPI 3: Peak (kW)']
    print(tabulate(df[cols2], headers='keys', tablefmt='grid', showindex=False))
    
    # =========================================================================
    # TABLE 3: RENEWABLE ENERGY METRICS (DER only)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: RENEWABLE ENERGY UTILIZATION (DER Scenarios)")
    print("=" * 100)
    der_df = df[df['Scenario'].str.contains('DER') & ~df['Scenario'].str.contains('No DER')]
    cols3 = ['Vessels', 'Scenario', 'PV (kWh)', 'KPI 4: Self-Cons (%)', 'KPI 5: Self-Suff (%)']
    print(tabulate(der_df[cols3], headers='keys', tablefmt='grid', showindex=False))
    
    # Save full table
    output_path = Path(__file__).parent / "comparison_results" / "kpi_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Full KPI table saved to: {output_path}")
    
    # =========================================================================
    # LATEX TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("LATEX TABLE FOR THESIS")
    print("=" * 100)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Key Performance Indicators for Port Electrification Scenarios}
\label{tab:kpis}
\small
\begin{tabular}{|c|l|c|c|c|c|c|c|}
\hline
\textbf{Vessels} & \textbf{Scenario} & \textbf{Reliability} & \textbf{Cost} & \textbf{€/Trip} & \textbf{Peak} & \textbf{Self-Cons.} & \textbf{Self-Suff.} \\
 &  & \textbf{(\%)} & \textbf{(€)} &  & \textbf{(kW)} & \textbf{(\%)} & \textbf{(\%)} \\
\hline
"""
    
    for _, row in df.iterrows():
        sc = row['KPI 4: Self-Cons (%)'] if row['KPI 4: Self-Cons (%)'] != '-' else '-'
        ss = row['KPI 5: Self-Suff (%)'] if row['KPI 5: Self-Suff (%)'] != '-' else '-'
        latex += f"{row['Vessels']} & {row['Scenario']} & {row['KPI 6: Reliability (%)']} & {row['KPI 1: Cost (€)']} & {row['KPI 2: €/Trip']} & {row['KPI 3: Peak (kW)']} & {sc} & {ss} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    print(latex)
    
    latex_path = Path(__file__).parent / "comparison_results" / "kpi_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 100)
    print("KEY FINDINGS FOR THESIS")
    print("=" * 100)
    
    for vessels in VESSEL_COUNTS:
        v_df = df[df['Vessels'] == vessels]
        baseline = v_df[v_df['Scenario'] == 'No Optimization, No DER'].iloc[0]
        opt_der = v_df[v_df['Scenario'] == 'Optimization + DER'].iloc[0]
        
        reliability_improvement = opt_der['KPI 6: Reliability (%)'] - baseline['KPI 6: Reliability (%)']
        cost_per_trip_change = ((opt_der['KPI 2: €/Trip'] / baseline['KPI 2: €/Trip']) - 1) * 100
        
        print(f"\n{vessels} Vessels:")
        print(f"  Baseline (No Opt, No DER):")
        print(f"    - Reliability: {baseline['KPI 6: Reliability (%)']}%")
        print(f"    - Cost: €{baseline['KPI 1: Cost (€)']} (€{baseline['KPI 2: €/Trip']}/trip)")
        print(f"    - Peak Power: {baseline['KPI 3: Peak (kW)']} kW")
        print(f"  With Optimization + DER:")
        print(f"    - Reliability: {opt_der['KPI 6: Reliability (%)']}% ({reliability_improvement:+.1f}pp improvement)")
        print(f"    - Cost: €{opt_der['KPI 1: Cost (€)']} (€{opt_der['KPI 2: €/Trip']}/trip)")
        print(f"    - Cost per trip change: {cost_per_trip_change:+.1f}%")
        if opt_der['KPI 5: Self-Suff (%)'] != '-':
            print(f"    - Self-sufficiency: {opt_der['KPI 5: Self-Suff (%)']}%")


if __name__ == "__main__":
    main()
