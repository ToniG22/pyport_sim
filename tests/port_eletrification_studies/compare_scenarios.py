"""
Comparison script for port electrification studies.

This script reads the simulation databases and generates comparison plots for:
1. Individual SOC of boats (with departure times as vertical lines)
2. Contracted power vs charger usage
3. Summary of reliability (delayed/canceled trips)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import os

# Configuration
SCENARIOS = {
    "no_opt_no_der": "No Optimization, No DER",
    "opt_no_der": "Optimization, No DER", 
    "opt_der": "Optimization + DER",
}

VESSEL_COUNTS = [5, 10, 20]

# Trip schedule times
TRIP_DEPARTURE_HOURS = [9, 14]  # 9:00 AM and 2:00 PM

# Colors for scenarios
COLORS = {
    "no_opt_no_der": "#e74c3c",  # Red
    "opt_no_der": "#3498db",     # Blue
    "opt_der": "#2ecc71",        # Green
}

def get_db_path(vessels: int, scenario: str) -> str:
    """Get the database path for a specific scenario."""
    # Databases are in the project root
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / f"{vessels}_vessels_{scenario}.db")


def load_boat_soc_data(db_path: str) -> pd.DataFrame:
    """Load boat SOC data from the database."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get SOC metric ID
    soc_query = "SELECT metric_id FROM metric WHERE metric_name = 'soc'"
    soc_id = pd.read_sql_query(soc_query, conn).iloc[0, 0]
    
    # Get all boat sources
    boat_sources = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'boat'",
        conn
    )
    
    # Get SOC measurements for all boats
    all_data = []
    for _, row in boat_sources.iterrows():
        query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as soc, '{row['source_name']}' as boat
            FROM measurements
            WHERE source_id = {row['source_id']} AND metric_id = {soc_id}
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        all_data.append(df)
    
    conn.close()
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        return result
    return pd.DataFrame()


def load_power_data(db_path: str) -> pd.DataFrame:
    """Load power consumption and contracted power data from the database."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get metric IDs
    consumption_id = pd.read_sql_query(
        "SELECT metric_id FROM metric WHERE metric_name = 'power_active_consumption'",
        conn
    ).iloc[0, 0]
    
    contracted_id = pd.read_sql_query(
        "SELECT metric_id FROM metric WHERE metric_name = 'contracted_power'",
        conn
    ).iloc[0, 0]
    
    # Get port source ID
    port_source = pd.read_sql_query(
        "SELECT source_id FROM source WHERE source_type = 'port'",
        conn
    ).iloc[0, 0]
    
    # Get power measurements
    query = f"""
        SELECT 
            m1.timestamp,
            CAST(m1.value AS FLOAT) as consumption,
            CAST(m2.value AS FLOAT) as contracted_power
        FROM measurements m1
        JOIN measurements m2 ON m1.timestamp = m2.timestamp AND m2.source_id = {port_source} AND m2.metric_id = {contracted_id}
        WHERE m1.source_id = {port_source} AND m1.metric_id = {consumption_id}
        ORDER BY m1.timestamp
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_charger_power_data(db_path: str) -> pd.DataFrame:
    """Load individual charger power data from the database."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get power_active metric ID
    power_id = pd.read_sql_query(
        "SELECT metric_id FROM metric WHERE metric_name = 'power_active'",
        conn
    ).iloc[0, 0]
    
    # Get all charger sources
    charger_sources = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'charger'",
        conn
    )
    
    # Get power measurements for all chargers
    all_data = []
    for _, row in charger_sources.iterrows():
        query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as power, '{row['source_name']}' as charger
            FROM measurements
            WHERE source_id = {row['source_id']} AND metric_id = {power_id}
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        all_data.append(df)
    
    conn.close()
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        return result
    return pd.DataFrame()


def load_boat_state_data(db_path: str) -> pd.DataFrame:
    """Load boat state data (sailing = 1.0, not sailing = 0.0) from the database."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get state metric ID
    state_query = "SELECT metric_id FROM metric WHERE metric_name = 'state'"
    state_result = pd.read_sql_query(state_query, conn)
    if state_result.empty:
        conn.close()
        return pd.DataFrame()
    state_id = state_result.iloc[0, 0]
    
    # Get all boat sources
    boat_sources = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'boat'",
        conn
    )
    
    # Get state measurements for all boats
    all_data = []
    for _, row in boat_sources.iterrows():
        query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as state, '{row['source_name']}' as boat
            FROM measurements
            WHERE source_id = {row['source_id']} AND metric_id = {state_id}
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        all_data.append(df)
    
    conn.close()
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        return result
    return pd.DataFrame()


def analyze_reliability(db_path: str, num_vessels: int) -> dict:
    """
    Analyze trip reliability by examining boat state patterns.
    
    A trip is detected when boat state changes to sailing (1.0).
    A trip is on-time if it starts at the scheduled hour (9:00 or 14:00).
    A trip is delayed if it starts later.
    A trip is cancelled if it never starts (second trip slot not detected).
    
    Returns dict with:
    - total_trips: Total scheduled trips
    - on_time_trips: Trips that started on time
    - delayed_trips: Trips that were delayed
    - cancelled_trips: Trips that were cancelled
    """
    # Total trips = 2 trips per vessel per day (Monday schedule)
    total_trips = num_vessels * 2
    
    if not os.path.exists(db_path):
        return {
            "total_trips": total_trips,
            "on_time_trips": 0,
            "delayed_trips": 0,
            "cancelled_trips": total_trips,
            "delay_rate": 100.0,
            "cancel_rate": 100.0,
        }
    
    # Load boat state data
    state_data = load_boat_state_data(db_path)
    
    if state_data.empty:
        return {
            "total_trips": total_trips,
            "on_time_trips": 0,
            "delayed_trips": 0,
            "cancelled_trips": total_trips,
            "delay_rate": 100.0,
            "cancel_rate": 100.0,
        }
    
    # Get unique boats
    boats = state_data['boat'].unique()
    
    on_time_trips = 0
    delayed_trips = 0
    cancelled_trips = 0
    
    for boat in boats:
        boat_data = state_data[state_data['boat'] == boat].sort_values('timestamp')
        
        # Find all trip starts (when state changes from 0 to 1)
        state_values = boat_data['state'].values
        timestamps = boat_data['timestamp'].values
        
        trip_starts = []
        for i in range(1, len(state_values)):
            if state_values[i] == 1.0 and state_values[i-1] == 0.0:
                trip_starts.append(pd.to_datetime(timestamps[i]))
        
        # Analyze each trip slot
        morning_trip_found = False
        afternoon_trip_found = False
        
        for trip_start in trip_starts:
            hour = trip_start.hour
            minute = trip_start.minute
            
            if hour == 9 and minute < 15:
                # Morning trip started on time
                on_time_trips += 1
                morning_trip_found = True
            elif 9 <= hour < 14:
                # Morning trip delayed
                if not morning_trip_found:
                    delayed_trips += 1
                    morning_trip_found = True
            elif hour == 14 and minute < 15:
                # Afternoon trip started on time
                on_time_trips += 1
                afternoon_trip_found = True
            elif 14 <= hour < 18:
                # Afternoon trip delayed but still started
                if not afternoon_trip_found:
                    delayed_trips += 1
                    afternoon_trip_found = True
            elif hour >= 18:
                # Late trip - could be delayed morning or afternoon
                if not morning_trip_found:
                    delayed_trips += 1
                    morning_trip_found = True
                elif not afternoon_trip_found:
                    delayed_trips += 1
                    afternoon_trip_found = True
        
        # Count cancelled trips
        if not morning_trip_found:
            cancelled_trips += 1
        if not afternoon_trip_found:
            cancelled_trips += 1
    
    # Calculate rates
    delay_rate = (delayed_trips / total_trips) * 100 if total_trips > 0 else 0
    cancel_rate = (cancelled_trips / total_trips) * 100 if total_trips > 0 else 0
    
    return {
        "total_trips": total_trips,
        "on_time_trips": on_time_trips,
        "delayed_trips": delayed_trips,
        "cancelled_trips": cancelled_trips,
        "delay_rate": delay_rate,
        "cancel_rate": cancel_rate,
    }


def plot_soc_comparison(vessels: int, output_dir: str):
    """Create SOC comparison plot for a given number of vessels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for idx, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        db_path = get_db_path(vessels, scenario)
        soc_data = load_boat_soc_data(db_path)
        
        if soc_data.empty:
            ax.text(0.5, 0.5, f"No data for {label}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Plot SOC for each boat
        boats = sorted(soc_data['boat'].unique())
        colors_boats = plt.cm.tab20(np.linspace(0, 1, len(boats)))
        
        for i, boat in enumerate(boats):
            boat_data = soc_data[soc_data['boat'] == boat].sort_values('timestamp')
            ax.plot(boat_data['timestamp'], boat_data['soc'], 
                   color=colors_boats[i], alpha=0.7, linewidth=1.5, label=boat)
        
        # Add vertical lines for departure times
        if not soc_data.empty:
            base_date = soc_data['timestamp'].min().date()
            for hour in TRIP_DEPARTURE_HOURS:
                departure_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour))
                ax.axvline(x=departure_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(departure_time, 105, f"{hour}:00", ha='center', va='bottom', 
                       color='red', fontsize=9, fontweight='bold')
        
        # Add minimum SOC requirement line
        ax.axhline(y=62.6, color='orange', linestyle=':', alpha=0.8, linewidth=1.5, label='Min SOC for Trip')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('SOC (%)', fontsize=11) if idx == 0 else None
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 110)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    handles = [mpatches.Patch(color='red', label='Departure Time'),
               mpatches.Patch(color='orange', label='Min SOC (62.6%)')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle(f'Boat SOC Comparison - {vessels} Vessels', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'soc_comparison_{vessels}_vessels.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_power_comparison(vessels: int, output_dir: str):
    """Create power usage comparison plot for a given number of vessels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for idx, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        db_path = get_db_path(vessels, scenario)
        power_data = load_power_data(db_path)
        
        if power_data.empty:
            ax.text(0.5, 0.5, f"No data for {label}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Plot consumption
        ax.fill_between(power_data['timestamp'], 0, power_data['consumption'], 
                       alpha=0.6, color=COLORS[scenario], label='Charger Usage')
        ax.plot(power_data['timestamp'], power_data['consumption'], 
               color=COLORS[scenario], linewidth=1.5)
        
        # Plot contracted power limit
        contracted = power_data['contracted_power'].iloc[0]
        ax.axhline(y=contracted, color='black', linestyle='--', 
                  linewidth=2, label=f'Contracted Power ({contracted} kW)')
        
        # Add departure time markers
        if not power_data.empty:
            base_date = power_data['timestamp'].min().date()
            for hour in TRIP_DEPARTURE_HOURS:
                departure_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour))
                ax.axvline(x=departure_time, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Power (kW)', fontsize=11) if idx == 0 else None
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Power Usage vs Contracted Power - {vessels} Vessels', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'power_comparison_{vessels}_vessels.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_reliability_summary(output_dir: str):
    """Create reliability summary plot comparing all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, vessels in enumerate(VESSEL_COUNTS):
        ax = axes[idx]
        
        scenario_names = []
        on_time = []
        delayed = []
        cancelled = []
        
        for scenario, label in SCENARIOS.items():
            db_path = get_db_path(vessels, scenario)
            reliability = analyze_reliability(db_path, vessels)
            
            scenario_names.append(label.replace(', ', '\n'))
            on_time.append(reliability['on_time_trips'])
            delayed.append(reliability['delayed_trips'])
            cancelled.append(reliability['cancelled_trips'])
        
        x = np.arange(len(scenario_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, on_time, width, label='On-Time', color='#2ecc71')
        bars2 = ax.bar(x, delayed, width, label='Delayed', color='#f39c12')
        bars3 = ax.bar(x + width, cancelled, width, label='Cancelled', color='#e74c3c')
        
        ax.set_xlabel('Scenario', fontsize=11)
        ax.set_ylabel('Number of Trips', fontsize=11) if idx == 0 else None
        ax.set_title(f'{vessels} Vessels', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Trip Reliability Summary by Scenario', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'reliability_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_summary_table(output_dir: str):
    """Create a summary table with key metrics."""
    summary_data = []
    
    for vessels in VESSEL_COUNTS:
        for scenario, label in SCENARIOS.items():
            db_path = get_db_path(vessels, scenario)
            reliability = analyze_reliability(db_path, vessels)
            
            # Load power data for peak analysis
            power_data = load_power_data(db_path)
            peak_power = power_data['consumption'].max() if not power_data.empty else 0
            avg_power = power_data['consumption'].mean() if not power_data.empty else 0
            contracted = power_data['contracted_power'].iloc[0] if not power_data.empty else 80
            
            summary_data.append({
                'Vessels': vessels,
                'Scenario': label,
                'Total Trips': reliability['total_trips'],
                'On-Time': reliability['on_time_trips'],
                'Delayed': reliability['delayed_trips'],
                'Cancelled': reliability['cancelled_trips'],
                'Delay Rate (%)': f"{reliability['delay_rate']:.1f}",
                'Cancel Rate (%)': f"{reliability['cancel_rate']:.1f}",
                'Peak Power (kW)': f"{peak_power:.1f}",
                'Avg Power (kW)': f"{avg_power:.1f}",
                'Contracted (kW)': f"{contracted:.1f}",
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def plot_combined_comparison(output_dir: str):
    """Create a combined comparison plot showing key metrics across all scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cancel rate comparison
    ax1 = axes[0, 0]
    x = np.arange(len(VESSEL_COUNTS))
    width = 0.25
    
    for i, (scenario, label) in enumerate(SCENARIOS.items()):
        cancel_rates = []
        for vessels in VESSEL_COUNTS:
            db_path = get_db_path(vessels, scenario)
            reliability = analyze_reliability(db_path, vessels)
            cancel_rates.append(reliability['cancel_rate'])
        
        ax1.bar(x + i * width, cancel_rates, width, label=label, color=COLORS[scenario])
    
    ax1.set_xlabel('Number of Vessels', fontsize=11)
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=11)
    ax1.set_title('Trip Cancellation Rate by Scenario', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(VESSEL_COUNTS)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Peak power comparison
    ax2 = axes[0, 1]
    for i, (scenario, label) in enumerate(SCENARIOS.items()):
        peak_powers = []
        for vessels in VESSEL_COUNTS:
            db_path = get_db_path(vessels, scenario)
            power_data = load_power_data(db_path)
            peak_powers.append(power_data['consumption'].max() if not power_data.empty else 0)
        
        ax2.bar(x + i * width, peak_powers, width, label=label, color=COLORS[scenario])
    
    # Add contracted power reference line
    ax2.axhline(y=80, color='black', linestyle='--', linewidth=2, label='Contracted Power')
    
    ax2.set_xlabel('Number of Vessels', fontsize=11)
    ax2.set_ylabel('Peak Power (kW)', fontsize=11)
    ax2.set_title('Peak Power Consumption by Scenario', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(VESSEL_COUNTS)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Successful trips comparison
    ax3 = axes[1, 0]
    for i, (scenario, label) in enumerate(SCENARIOS.items()):
        success_rates = []
        for vessels in VESSEL_COUNTS:
            db_path = get_db_path(vessels, scenario)
            reliability = analyze_reliability(db_path, vessels)
            success_rate = ((reliability['on_time_trips'] + reliability['delayed_trips']) / reliability['total_trips']) * 100
            success_rates.append(success_rate)
        
        ax3.bar(x + i * width, success_rates, width, label=label, color=COLORS[scenario])
    
    ax3.set_xlabel('Number of Vessels', fontsize=11)
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Trip Success Rate (On-Time + Delayed)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(VESSEL_COUNTS)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 110)
    
    # 4. Average power utilization
    ax4 = axes[1, 1]
    for i, (scenario, label) in enumerate(SCENARIOS.items()):
        avg_powers = []
        for vessels in VESSEL_COUNTS:
            db_path = get_db_path(vessels, scenario)
            power_data = load_power_data(db_path)
            avg_powers.append(power_data['consumption'].mean() if not power_data.empty else 0)
        
        ax4.bar(x + i * width, avg_powers, width, label=label, color=COLORS[scenario])
    
    ax4.set_xlabel('Number of Vessels', fontsize=11)
    ax4.set_ylabel('Average Power (kW)', fontsize=11)
    ax4.set_title('Average Power Consumption by Scenario', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(VESSEL_COUNTS)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Port Electrification Study - Scenario Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def load_charger_power_data(db_path: str) -> pd.DataFrame:
    """Load individual charger power data from the database."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Get power_active metric ID
    try:
        power_id = pd.read_sql_query(
            "SELECT metric_id FROM metric WHERE metric_name = 'power_active'",
            conn
        ).iloc[0, 0]
    except:
        conn.close()
        return pd.DataFrame()
    
    # Get all charger sources
    charger_sources = pd.read_sql_query(
        "SELECT source_id, source_name FROM source WHERE source_type = 'charger'",
        conn
    )
    
    all_data = []
    for _, row in charger_sources.iterrows():
        query = f"""
            SELECT timestamp, CAST(value AS FLOAT) as power, '{row['source_name']}' as charger
            FROM measurements
            WHERE source_id = {row['source_id']} AND metric_id = {power_id}
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        all_data.append(df)
    
    conn.close()
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        return result
    return pd.DataFrame()


def plot_soc_grid(output_dir: str):
    """
    Generate a 3x3 grid plot for boat SOC.
    
    Rows: 5 vessels (25%), 10 vessels (50%), 20 vessels (100%)
    Columns: No Opt No DER, Opt No DER, Opt + DER
    
    Each subplot shows ONLY:
    - SOC lines for all boats
    - Two vertical lines for departure times (9:00 and 14:00)
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    vessel_labels = {
        5: "5 Vessels (25%)",
        10: "10 Vessels (50%)",
        20: "20 Vessels (100%)"
    }
    
    scenario_order = ["no_opt_no_der", "opt_no_der", "opt_der"]
    scenario_labels = {
        "no_opt_no_der": "No Optimization, No DER",
        "opt_no_der": "Optimization, No DER",
        "opt_der": "Optimization + DER"
    }
    
    # Distinct colors for boats
    boat_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    for row_idx, vessels in enumerate(VESSEL_COUNTS):
        for col_idx, scenario in enumerate(scenario_order):
            ax = axes[row_idx, col_idx]
            db_path = get_db_path(vessels, scenario)
            
            soc_data = load_boat_soc_data(db_path)
            
            if not soc_data.empty:
                boats = sorted(soc_data['boat'].unique())
                
                for b_idx, boat in enumerate(boats):
                    boat_data = soc_data[soc_data['boat'] == boat].sort_values('timestamp')
                    hours = np.array([(t.hour + t.minute/60.0) for t in boat_data['timestamp']])
                    # SOC is already stored as percentage (0-100) in database
                    soc_values = boat_data['soc'].values
                    
                    # Plot SOC line with thicker lines
                    line_width = 2.0 if vessels <= 5 else (1.5 if vessels <= 10 else 1.0)
                    ax.plot(hours, soc_values,
                           color=boat_colors[b_idx % len(boat_colors)], 
                           linewidth=line_width,
                           alpha=0.85)
                
                # Add two vertical lines for departure times
                ax.axvline(x=9, color='red', linestyle='--', linewidth=2.5, label='Departure')
                ax.axvline(x=14, color='red', linestyle='--', linewidth=2.5)
            
            # Formatting
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 105)
            ax.set_xticks([0, 3, 6, 9, 12, 14, 17, 20, 24])
            ax.grid(True, alpha=0.3)
            
            # Labels
            if row_idx == 0:
                ax.set_title(scenario_labels[scenario], fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{vessel_labels[vessels]}\nSOC (%)', fontsize=10)
            if row_idx == 2:
                ax.set_xlabel('Hour of Day', fontsize=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Departure Time (9:00 & 14:00)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle('Boat State of Charge (SOC) Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'soc_grid_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_power_grid(output_dir: str):
    """
    Generate a 3x3 grid plot for charger power consumption.
    
    Rows: 5 vessels (25%), 10 vessels (50%), 20 vessels (100%)
    Columns: No Opt No DER, Opt No DER, Opt + DER
    
    Each subplot shows total charger power + contracted power horizontal line.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    vessel_labels = {
        5: "5 Vessels (25% Fleet)",
        10: "10 Vessels (50% Fleet)",
        20: "20 Vessels (100% Fleet)"
    }
    
    contracted_powers = {5: 80, 10: 80, 20: 80}  # kW
    
    scenario_order = ["no_opt_no_der", "opt_no_der", "opt_der"]
    scenario_labels = {
        "no_opt_no_der": "No Optimization\nNo DER",
        "opt_no_der": "Optimization\nNo DER",
        "opt_der": "Optimization\n+ DER"
    }
    
    # Color palette for chargers
    charger_colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    for row_idx, vessels in enumerate(VESSEL_COUNTS):
        for col_idx, scenario in enumerate(scenario_order):
            ax = axes[row_idx, col_idx]
            db_path = get_db_path(vessels, scenario)
            
            # Load port-level power data
            power_data = load_power_data(db_path)
            charger_data = load_charger_power_data(db_path)
            
            if not power_data.empty:
                hours = [(t.hour + t.minute/60) for t in power_data['timestamp']]
                
                # Plot total consumption
                ax.fill_between(hours, power_data['consumption'], 
                               alpha=0.4, color=COLORS[scenario], label='Total Consumption')
                ax.plot(hours, power_data['consumption'], 
                       color=COLORS[scenario], linewidth=1.5)
                
                # Plot individual chargers as stacked (if data available)
                if not charger_data.empty:
                    chargers = charger_data['charger'].unique()
                    
                    # Create pivot table for stacked area
                    pivot = charger_data.pivot_table(
                        index='timestamp', columns='charger', values='power', 
                        aggfunc='first'
                    ).fillna(0)
                    
                    if len(pivot) > 0:
                        pivot_hours = [(t.hour + t.minute/60) for t in pivot.index]
                        
                        # Plot individual charger lines (thinner)
                        for c_idx, charger in enumerate(sorted(chargers)):
                            if charger in pivot.columns:
                                ax.plot(pivot_hours, pivot[charger], 
                                       color=charger_colors[c_idx % 12], 
                                       linewidth=0.8, alpha=0.5)
            
            # Add contracted power horizontal line
            contracted = contracted_powers[vessels]
            ax.axhline(y=contracted, color='red', linestyle='--', 
                      linewidth=2.5, label=f'Contracted Power ({contracted} kW)')
            
            # Add departure time vertical lines
            for dep_hour in TRIP_DEPARTURE_HOURS:
                ax.axvline(x=dep_hour, color='gray', linestyle=':', 
                          linewidth=1.5, alpha=0.5)
            
            # Formatting
            ax.set_xlim(0, 24)
            ax.set_ylim(0, max(180, contracted * 2))
            ax.set_xticks([0, 6, 9, 12, 14, 18, 24])
            ax.grid(True, alpha=0.3)
            
            # Add row/column labels
            if row_idx == 0:
                ax.set_title(scenario_labels[scenario], fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{vessel_labels[vessels]}\nPower (kW)', fontsize=10)
            else:
                ax.set_ylabel('Power (kW)', fontsize=10)
            if row_idx == 2:
                ax.set_xlabel('Hour of Day', fontsize=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["opt_der"], alpha=0.4, label='Total Consumption'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Contracted Power (80 kW)'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5, label='Departure Time'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
              bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Charger Power Consumption by Scenario and Fleet Size', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'power_grid_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main function to run the comparison analysis."""
    print("=" * 60)
    print("Port Electrification Study - Scenario Comparison")
    print("=" * 60)
    
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate plots for each vessel count
    print("\n1. Generating SOC comparison plots...")
    for vessels in VESSEL_COUNTS:
        plot_soc_comparison(vessels, output_dir)
    
    print("\n2. Generating power usage comparison plots...")
    for vessels in VESSEL_COUNTS:
        plot_power_comparison(vessels, output_dir)
    
    print("\n3. Generating reliability summary plot...")
    plot_reliability_summary(output_dir)
    
    print("\n4. Generating combined comparison plot...")
    plot_combined_comparison(output_dir)
    
    print("\n5. Creating summary table...")
    create_summary_table(output_dir)
    
    print("\n6. Generating SOC grid comparison (3x3)...")
    plot_soc_grid(output_dir)
    
    print("\n7. Generating Power grid comparison (3x3)...")
    plot_power_grid(output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete! All plots saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
