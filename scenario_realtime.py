"""Real-time simulation demo: Quick 1-hour test."""

from datetime import datetime
from models import Port, Boat, Charger
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_realtime_demo():
    """Run a quick real-time simulation demo (1 hour, 60 second timesteps)."""
    print("=" * 60)
    print("REAL-TIME SIMULATION DEMO: 1 Hour")
    print("=" * 60)

    # Create port
    port = Port(
        name="Marina del Sol",
        contracted_power=100,
        lat=32.64542,
        lon=-16.90841,
    )

    # Create 1 boat at low charge
    boat = Boat(
        name="QuickBoat",
        motor_power=30,
        weight=1800,
        length=7.0,
        battery_capacity=80,
        range_speed=6.0,
        soc=0.20,  # 20% charged
    )

    # Create 1 charger
    charger = Charger(
        name="Charger_1",
        max_power=22,
        efficiency=0.95,
    )

    # Add to port
    port.add_boat(boat)
    port.add_charger(charger)

    print(f"\nPort: {port}")
    print(f"Boat: {boat.name} - SOC: {boat.soc:.0%}")
    print(f"Charger: {charger.name} - {charger.max_power}kW")

    # Configure for real-time with 60 second timesteps
    settings = Settings(
        timestep=60,  # 1 minute timesteps
        mode=SimulationMode.REAL_TIME,
        db_path="realtime_demo.db",
    )

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()

    # Run simulation for 1 hour
    # Note: This will take 1 hour in real time!
    print("\n⚠️  WARNING: Real-time mode will take 1 hour to complete!")
    print("Press Ctrl+C to stop early if needed.\n")

    # Get current time for start
    now = datetime.utcnow()
    start_hour = datetime(now.year, now.month, now.day, now.hour, 0, 0)

    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date=start_hour,
        days=1,  # Will stop after 1 hour due to timestep count
    )

    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")

    print("\n" + "=" * 60)
    print("Real-time demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nNOTE: For a faster test, run scenario_simple.py in BATCH mode instead.")
    print("This real-time demo is mainly for demonstrating the real-time feature.\n")

    response = input("Continue with real-time demo? (y/n): ")
    if response.lower() == "y":
        run_realtime_demo()
    else:
        print("Demo cancelled. Run scenario_simple.py for a quick batch simulation.")
