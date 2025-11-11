"""Simple scenario: 1 boat, 1 charger, 1 day simulation."""

from models import Port, Boat, Charger
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_simple_scenario():
    """Run a simple simulation scenario."""
    print("=" * 60)
    print("SIMPLE SCENARIO: 1 Boat, 1 Charger, 1 Day")
    print("=" * 60)

    # Create port
    port = Port(
        name="Marina del Sol",
        contracted_power=100,  # 100 kW
        lat=32.64542,
        lon=-16.90841,
    )

    # Create 1 boat (starting at 30% charge)
    boat = Boat(
        name="SeaBreeze",
        motor_power=120,  # kW
        weight=2500,  # kg
        length=8.5,  # m
        battery_capacity=150,  # kWh (larger battery for longer range)
        range_speed=15.0,  # knots
        soc=0.30,  # 30% charged
    )

    # Create 2 boat with different battery capacities
    boat2 = Boat(
        name="SeaBreeze_2",
        motor_power=100,  # kW
        weight=2500,  # kg
        length=8.5,  # m
        battery_capacity=100,  # kWh
        range_speed=16.0,  # knots
        soc=0.50,  # 50% charged
    )

    # Create 1 charger
    charger = Charger(
        name="FastCharger_A",
        max_power=22,  # kW
        efficiency=0.95,  # 95% efficient
    )
    charger2 = Charger(
        name="FastCharger_B",
        max_power=22,  # kW
        efficiency=0.95,  # 95% efficient
    )

    # Add boats and charger to port
    port.add_boat(boat)
    port.add_boat(boat2)
    port.add_charger(charger)
    port.add_charger(charger2)

    print(f"\nPort: {port}")
    print(f"Boat: {boat}")
    print(f"  - K-factor: {boat.k:.4f}")
    print(f"Charger: {charger}")

    # Configure simulation settings
    settings = Settings(
        timestep=900,  # 15 minute timesteps
        mode=SimulationMode.BATCH,
        db_path="simple_scenario.db",
    )

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()

    # Create and run simulation (only pass port, settings, db_manager)
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date=None,  # Will use today at midnight UTC
        days=1,
    )

    sim.run()

    # Query and display some results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    # Get boat SOC over time
    boat_soc_data = db_manager.get_measurements(source=boat.name, metric="soc")
    print("\nBoat SOC progression (first 10 and last 10 readings):")
    for row in boat_soc_data[:10]:
        print(f"  {row['timestamp']}: SOC={row['value']:5.1f}%")
    if len(boat_soc_data) > 20:
        print("  ...")
        for row in boat_soc_data[-10:]:
            print(f"  {row['timestamp']}: SOC={row['value']:5.1f}%")

    # Get port power usage
    port_power_data = db_manager.get_measurements(
        source="port", metric="total_power_used"
    )
    print("\nPort power usage statistics:")
    if port_power_data:
        power_values = [row["value"] for row in port_power_data]
        print(f"  Average: {sum(power_values)/len(power_values):.2f} kW")
        print(f"  Maximum: {max(power_values):.2f} kW")
        print(f"  Minimum: {min(power_values):.2f} kW")

    # Get charger activity
    charger_state_data = db_manager.get_measurements(
        source=charger.name, metric="state"
    )
    if charger_state_data:
        charging_count = sum(1 for row in charger_state_data if row["value"] == 1.0)
        idle_count = len(charger_state_data) - charging_count
        print("\nCharger activity:")
        print(
            f"  Charging: {charging_count} timesteps ({charging_count*settings.timestep/3600:.1f}h)"
        )
        print(
            f"  Idle: {idle_count} timesteps ({idle_count*settings.timestep/3600:.1f}h)"
        )

    print("\n" + "=" * 60)
    print(f"✓ Results saved to: {settings.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_simple_scenario()
