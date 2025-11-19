"""Scenario with optimization enabled."""

from models import Port, Boat, Charger, PV, BESS, BESSControlStrategy
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_optimizer_scenario():
    """Run a simulation scenario with optimization enabled."""
    print("=" * 60)
    print("SCENARIO: With Optimization (SCIP)")
    print("=" * 60)

    # Create port
    port = Port(
        name="Marina del Sol",
        contracted_power=40,  # 40 kW
        lat=32.64542,
        lon=-16.90841,
    )

    # Create 2 boats
    boat1 = Boat(
        name="SeaBreeze",
        motor_power=120,  # kW
        weight=2500,  # kg
        length=8.5,  # m
        battery_capacity=150,  # kWh
        range_speed=15.0,  # knots
        soc=0.30,  # 30% charged
    )

    boat2 = Boat(
        name="SeaBreeze_2",
        motor_power=100,  # kW
        weight=2500,  # kg
        length=8.5,  # m
        battery_capacity=100,  # kWh
        range_speed=16.0,  # knots
        soc=0.50,  # 50% charged
    )

    # Create 2 chargers
    charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
    charger2 = Charger(name="FastCharger_B", max_power=22, efficiency=0.95)

    # Create PV system
    pv_system = PV(
        name="Solar_Array_1",
        capacity=30.0,  # 30 kW DC
        tilt=30.0,  # 30 degrees
        azimuth=180.0,  # South-facing
        efficiency=0.85,
        latitude=port.lat,
        longitude=port.lon,
    )

    # Create BESS
    bess = BESS(
        name="Battery_Storage_1",
        capacity=100.0,  # 100 kWh
        max_charge_power=25.0,  # 25 kW
        max_discharge_power=25.0,  # 25 kW
        efficiency=0.90,
        soc_min=0.10,
        soc_max=0.90,
        initial_soc=0.50,
        control_strategy=BESSControlStrategy.DEFAULT,
    )

    # Add components to port
    port.add_boat(boat1)
    port.add_boat(boat2)
    port.add_charger(charger1)
    port.add_charger(charger2)
    port.add_pv(pv_system)
    port.add_bess(bess)

    print(f"\nPort: {port}")
    print(f"Boats: {boat1.name}, {boat2.name}")
    print(f"Chargers: {charger1.name}, {charger2.name}")
    print(f"PV: {pv_system}")
    print(f"BESS: {bess}")

    # Configure simulation WITH optimizer
    settings = Settings(
        timestep=900,  # 15 minutes
        mode=SimulationMode.BATCH,
        db_path="optimizer_scenario.db",
        use_optimizer=True,  # ← OPTIMIZATION ENABLED
    )

    print(f"\n⚙️  Mode: Optimized scheduling (SCIP)")
    print(f"   - Forecasting: Enabled")
    print(f"   - Optimization: Enabled")
    print(f"   - Objective: Minimize trip delays + grid usage")

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()

    # Create and run simulation
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date=None,
        days=7,
    )

    sim.run()

    print("\n" + "=" * 60)
    print(f"✓ Results saved to: {settings.db_path}")
    print(f"✓ View data using: streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    run_optimizer_scenario()

