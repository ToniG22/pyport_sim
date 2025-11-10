"""Main entry point for the electric port simulator."""

from models import Port, Boat, BoatState, Charger, ChargerState
from config import Settings, SimulationMode
from database import DatabaseManager


def main():
    """Main function to run the simulator."""
    print("Electric Port Simulator")
    print("=" * 50)

    # Example: Create a port
    port = Port(
        name="Marina del Sol",
        contracted_power=100,
        lat=32.64542,
        lon=-16.90841,  # 100 kW
    )

    # Example: Create boats
    print("\n" + "=" * 50)
    print("Creating boats...")
    print("=" * 50)

    boat1 = Boat(
        name="SeaBreeze",
        motor_power=50,  # kW
        weight=2500,  # kg
        length=8.5,  # m
        battery_capacity=60,  # kWh
        range_speed=8.0,  # knots
        soc=0.35,  # 35% charge
    )
    print(f"Boat 1: {boat1}")
    print(f"  - K-factor: {boat1.k:.4f}")

    boat2 = Boat(
        motor_power=30,
        weight=1800,
        length=7.0,
        battery_capacity=40,
        range_speed=6.5,
        soc=0.80,
    )
    print(f"\nBoat 2: {boat2}")
    print(f"  - K-factor: {boat2.k:.4f}")

    # Example: Create chargers
    print("\n" + "=" * 50)
    print("Creating chargers...")
    print("=" * 50)

    charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
    print(f"Charger 1: {charger1}")

    charger2 = Charger(max_power=11, efficiency=0.93)
    print(f"\nCharger 2: {charger2}")

    # Add boats and chargers to port
    port.add_boat(boat1)
    port.add_boat(boat2)
    port.add_charger(charger1)
    port.add_charger(charger2)

    print(f"\nPort configured: {port}")

    # Simulate charger connecting to boat
    print("\n" + "=" * 50)
    print("Simulating charging state...")
    print("=" * 50)
    charger1.state = ChargerState.CHARGING
    charger1.power = 20.0  # Charging at 20kW
    charger1.connected_boat = boat1.name
    boat1.state = BoatState.CHARGING
    print(f"\nCharger status: {charger1}")
    print(f"  - Effective power to battery: {charger1.effective_power:.1f}kW")
    print(f"Boat status: {boat1}")

    # Initialize settings
    settings = Settings(
        timestep=900,  # 15 minute timesteps
        mode=SimulationMode.BATCH,
        db_path="port_simulation.db",
    )

    print(f"\nSettings: timestep={settings.timestep}s, mode={settings.mode.value}")

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()
    print(f"\nDatabase initialized: {settings.db_path}")

    # Example: Save some measurements
    print("\n" + "=" * 50)
    print("Example: Saving measurements to database (UTC timestamps)")
    print("=" * 50)

    from datetime import datetime
    current_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Save measurements using batch insert (with UTC timestamp strings)
    sample_measurements = [
        (current_timestamp, "port", "total_power", 125.5),
        (current_timestamp, "port", "contracted_power", 500.0),
        (current_timestamp, "boat_001", "soc", 85.5),
        (current_timestamp, "boat_001", "power_draw", 22.5),
        (current_timestamp, "charger_01", "power_output", 22.5),
        (current_timestamp, "bess_01", "soc", 65.3),
        (current_timestamp, "bess_01", "power_flow", -15.0),
        (current_timestamp, "pv_01", "power_output", 50.2),
    ]

    db_manager.save_measurements_batch(sample_measurements)
    print(f"✓ Saved {len(sample_measurements)} measurements at {current_timestamp} UTC")

    # Query back some data
    print("\nQuerying measurements for 'boat_001':")
    boat_data = db_manager.get_measurements(source="boat_001")
    for row in boat_data:
        print(f"  - {row['timestamp']} | {row['metric']}: {row['value']}")

    print("\nQuerying all 'soc' metrics:")
    soc_data = db_manager.get_measurements(metric="soc")
    for row in soc_data:
        print(f"  - {row['source']}: {row['value']}%")

    print("\n" + "=" * 50)
    print("Ready for simulation setup!")
    print("\nNext steps:")
    print("  1. Define Boat model")
    print("  2. Define Charger model")
    print("  3. Define BESS model")
    print("  4. Define PV model")
    print("  5. Implement simulation engine")


if __name__ == "__main__":
    main()
