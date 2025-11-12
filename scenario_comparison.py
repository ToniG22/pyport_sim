"""Comparison scenario: Optimizer vs Default Control."""

from models import Port, Boat, Charger, PV, BESS, BESSControlStrategy
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_comparison():
    """Run both scenarios and compare results."""
    print("=" * 70)
    print("COMPARISON: Optimized vs Default Control")
    print("=" * 70)
    
    # Common configuration
    def create_port():
        port = Port(name="Marina del Sol", contracted_power=40, lat=32.64542, lon=-16.90841)
        
        boat1 = Boat(name="SeaBreeze", motor_power=120, weight=2500, length=8.5,
                    battery_capacity=150, range_speed=15.0, soc=0.30)
        boat2 = Boat(name="SeaBreeze_2", motor_power=100, weight=2500, length=8.5,
                    battery_capacity=100, range_speed=16.0, soc=0.50)
        
        charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
        charger2 = Charger(name="FastCharger_B", max_power=22, efficiency=0.95)
        
        pv = PV(name="Solar_Array_1", capacity=30.0, tilt=30.0, azimuth=180.0,
                efficiency=0.85, latitude=port.lat, longitude=port.lon)
        
        bess = BESS(name="Battery_Storage_1", capacity=100.0, max_charge_power=25.0,
                   max_discharge_power=25.0, efficiency=0.90, soc_min=0.10,
                   soc_max=0.90, initial_soc=0.50)
        
        port.add_boat(boat1)
        port.add_boat(boat2)
        port.add_charger(charger1)
        port.add_charger(charger2)
        port.add_pv(pv)
        port.add_bess(bess)
        
        return port
    
    # Test 1: Default Control
    print("\n" + "=" * 70)
    print("TEST 1: Default Rule-Based Control")
    print("=" * 70)
    
    port1 = create_port()
    settings1 = Settings(timestep=900, mode=SimulationMode.BATCH,
                        db_path="comparison_default.db", use_optimizer=False)
    db1 = DatabaseManager(settings1.db_path)
    db1.initialize_schema()
    
    sim1 = SimulationEngine(port1, settings1, db1, days=1)
    sim1.run()
    
    # Test 2: Optimized Control
    print("\n" + "=" * 70)
    print("TEST 2: Optimized Control (SCIP)")
    print("=" * 70)
    
    port2 = create_port()
    settings2 = Settings(timestep=900, mode=SimulationMode.BATCH,
                        db_path="comparison_optimizer.db", use_optimizer=True)
    db2 = DatabaseManager(settings2.db_path)
    db2.initialize_schema()
    
    sim2 = SimulationEngine(port2, settings2, db2, days=1)
    sim2.run()
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    import sqlite3
    
    # Get metrics from both databases
    def get_metrics(db_path, label):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total grid energy
        cursor.execute("""
            SELECT SUM(value) * 0.25 as total_kwh
            FROM measurements
            WHERE source='port' AND metric='power_active_consumption'
        """)
        total_consumption = cursor.fetchone()[0] or 0
        
        # Total PV energy
        cursor.execute("""
            SELECT SUM(value) * 0.25 as total_kwh
            FROM measurements
            WHERE source='port' AND metric='power_active_production'
        """)
        pv_energy = cursor.fetchone()[0] or 0
        
        # BESS discharge
        cursor.execute("""
            SELECT SUM(value) * 0.25 as total_kwh
            FROM measurements
            WHERE source='port' AND metric='bess_discharge'
        """)
        bess_discharge = cursor.fetchone()[0] or 0
        
        conn.close()
        
        grid_energy = total_consumption - pv_energy - bess_discharge
        
        print(f"\n{label}:")
        print(f"  Total Consumption: {total_consumption:.2f} kWh")
        print(f"  PV Energy: {pv_energy:.2f} kWh ({pv_energy/max(total_consumption,1)*100:.1f}%)")
        print(f"  BESS Discharge: {bess_discharge:.2f} kWh ({bess_discharge/max(total_consumption,1)*100:.1f}%)")
        print(f"  Grid Energy: {grid_energy:.2f} kWh ({grid_energy/max(total_consumption,1)*100:.1f}%)")
        print(f"  Renewable: {(pv_energy+bess_discharge)/max(total_consumption,1)*100:.1f}%")
        
        return {
            'consumption': total_consumption,
            'pv': pv_energy,
            'bess': bess_discharge,
            'grid': grid_energy
        }
    
    metrics1 = get_metrics("comparison_default.db", "Default Control")
    metrics2 = get_metrics("comparison_optimizer.db", "Optimized Control")
    
    # Calculate improvement
    print(f"\n{'='*70}")
    print("IMPROVEMENT:")
    print(f"{'='*70}")
    grid_reduction = metrics1['grid'] - metrics2['grid']
    grid_pct = (grid_reduction / max(metrics1['grid'], 1)) * 100
    print(f"  Grid usage reduction: {grid_reduction:.2f} kWh ({grid_pct:+.1f}%)")
    print(f"  Renewable increase: {(metrics2['pv']+metrics2['bess'])-(metrics1['pv']+metrics1['bess']):.2f} kWh")


if __name__ == "__main__":
    run_comparison()

