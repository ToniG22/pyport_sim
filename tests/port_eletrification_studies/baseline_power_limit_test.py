"""Baseline test: Power-limited charging without optimization.

This test enforces the contracted power limit without using optimization.
This provides a baseline to compare against optimizer performance - it shows
the minimum performance that should be achievable when respecting power limits.

The power-limited mode:
- Assigns boats to chargers using first-come-first-served (FCFS)
- Enforces contracted power limit by capping total charging power
- Distributes available power proportionally when limit is hit
- No optimization - just simple power capping

This baseline should help identify if optimizer issues are due to:
1. Power limit enforcement problems, or
2. More complex optimization logic issues
"""

from models import Port, Boat, Charger
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def main():
    """Main function to run the baseline power-limited simulation."""
    print("=" * 60)
    print("Baseline Power-Limited Test")
    print("=" * 60)
    print("\nThis test enforces contracted power limits WITHOUT optimization.")
    print("This is the minimum baseline - optimizer should perform at least this well.\n")

    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================

    # Port configuration
    port = Port(
        name="Funchal",
        contracted_power=80,  # Contracted power limit (kW)
        lat=32.64542,
        lon=-16.90841,
        tariff_path="assets/tariff/default_tariff.json",
    )

    # Boat configuration - 5 boats with trips
    boat1 = Boat(
        name="SeaBreeze_1",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.30,  # Low initial SOC to test charging
    )

    boat2 = Boat(
        name="SeaBreeze_2",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.50,
    )

    boat3 = Boat(
        name="SeaBreeze_3",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.50,
    )

    boat4 = Boat(
        name="SeaBreeze_4",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.50,
    )

    boat5 = Boat(
        name="SeaBreeze_5",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.50,
    )

    # Charger configuration - 5 chargers
    charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
    charger2 = Charger(name="FastCharger_B", max_power=22, efficiency=0.95)
    charger3 = Charger(name="FastCharger_C", max_power=22, efficiency=0.95)
    charger4 = Charger(name="FastCharger_D", max_power=22, efficiency=0.95)
    charger5 = Charger(name="FastCharger_E", max_power=22, efficiency=0.95)

    # Add components to port
    port.add_boat(boat1)
    port.add_boat(boat2)
    port.add_boat(boat3)
    port.add_boat(boat4)
    port.add_boat(boat5)
    port.add_charger(charger1)
    port.add_charger(charger2)
    port.add_charger(charger3)
    port.add_charger(charger4)
    port.add_charger(charger5)
    # No PV or BESS - testing pure power limit constraint

    print(f"\nPort: {port.name}")
    print(f"Contracted Power: {port.contracted_power} kW")
    print(f"Boats: {boat1.name}, {boat2.name}, {boat3.name}, {boat4.name}, {boat5.name}")
    print(f"Chargers: {charger1.name}, {charger2.name}, {charger3.name}, {charger4.name}, {charger5.name}")
    print(f"Total Charger Capacity: {sum(c.max_power for c in port.chargers)} kW")
    print(f"\n⚠️  Note: Total charger capacity ({sum(c.max_power for c in port.chargers)} kW) > ")
    print(f"   Contracted power ({port.contracted_power} kW), so power limiting will be active.\n")

    # Simulation settings - power_limit_mode enabled, optimizer disabled
    settings = Settings(
        timestep=900,  # 15 minutes
        mode=SimulationMode.BATCH,
        db_path="baseline_power_limit_test.db",
        use_optimizer=False,  # No optimization
        power_limit_mode=True,  # Enable power limiting
    )

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()
    db_manager.initialize_default_metrics()

    # Create and run simulation
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date="2025-09-01",
        days=1,
    )

    sim.run()

    print("\n" + "=" * 60)
    print("✓ Baseline test completed")
    print(f"✓ Results saved to: {settings.db_path}")
    print("=" * 60)
    print("\n📊 Expected outcomes:")
    print("  - All boats should be able to charge (within power limit)")
    print("  - Total charging power should never exceed contracted power")
    print("  - Some boats may need to wait if all chargers are at capacity")
    print("  - This represents the MINIMUM baseline - optimizer should do at least this well")
    print("=" * 60)


if __name__ == "__main__":
    main()
