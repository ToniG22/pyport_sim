"""Baseline test: 20 vessels with power-limited charging without optimization.

This test enforces the contracted power limit without using optimization.
This provides a baseline to compare against optimizer performance for 20 vessels.

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
    print("Baseline Power-Limited Test (20 Vessels, No DER)")
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

    # Boat configuration - 20 boats with trips
    boat1 = Boat(
        name="SeaBreeze_1",
        motor_power=100,
        weight=2500,
        length=8.5,
        battery_capacity=100,
        range_speed=16.0,
        soc=0.30,  # Low initial SOC to test charging
    )

    boat2 = Boat(name="SeaBreeze_2", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat3 = Boat(name="SeaBreeze_3", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat4 = Boat(name="SeaBreeze_4", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat5 = Boat(name="SeaBreeze_5", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat6 = Boat(name="SeaBreeze_6", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat7 = Boat(name="SeaBreeze_7", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat8 = Boat(name="SeaBreeze_8", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat9 = Boat(name="SeaBreeze_9", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat10 = Boat(name="SeaBreeze_10", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat11 = Boat(name="SeaBreeze_11", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat12 = Boat(name="SeaBreeze_12", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat13 = Boat(name="SeaBreeze_13", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat14 = Boat(name="SeaBreeze_14", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat15 = Boat(name="SeaBreeze_15", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat16 = Boat(name="SeaBreeze_16", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat17 = Boat(name="SeaBreeze_17", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat18 = Boat(name="SeaBreeze_18", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat19 = Boat(name="SeaBreeze_19", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)
    boat20 = Boat(name="SeaBreeze_20", motor_power=100, weight=2500, length=8.5, battery_capacity=100, range_speed=16.0, soc=0.50)

    # Charger configuration - 20 chargers
    charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
    charger2 = Charger(name="FastCharger_B", max_power=22, efficiency=0.95)
    charger3 = Charger(name="FastCharger_C", max_power=22, efficiency=0.95)
    charger4 = Charger(name="FastCharger_D", max_power=22, efficiency=0.95)
    charger5 = Charger(name="FastCharger_E", max_power=22, efficiency=0.95)
    charger6 = Charger(name="FastCharger_F", max_power=22, efficiency=0.95)
    charger7 = Charger(name="FastCharger_G", max_power=22, efficiency=0.95)
    charger8 = Charger(name="FastCharger_H", max_power=22, efficiency=0.95)
    charger9 = Charger(name="FastCharger_I", max_power=22, efficiency=0.95)
    charger10 = Charger(name="FastCharger_J", max_power=22, efficiency=0.95)
    charger11 = Charger(name="FastCharger_K", max_power=22, efficiency=0.95)
    charger12 = Charger(name="FastCharger_L", max_power=22, efficiency=0.95)
    charger13 = Charger(name="FastCharger_M", max_power=22, efficiency=0.95)
    charger14 = Charger(name="FastCharger_N", max_power=22, efficiency=0.95)
    charger15 = Charger(name="FastCharger_O", max_power=22, efficiency=0.95)
    charger16 = Charger(name="FastCharger_P", max_power=22, efficiency=0.95)
    charger17 = Charger(name="FastCharger_Q", max_power=22, efficiency=0.95)
    charger18 = Charger(name="FastCharger_R", max_power=22, efficiency=0.95)
    charger19 = Charger(name="FastCharger_S", max_power=22, efficiency=0.95)
    charger20 = Charger(name="FastCharger_T", max_power=22, efficiency=0.95)

    # Add components to port
    port.add_boat(boat1)
    port.add_boat(boat2)
    port.add_boat(boat3)
    port.add_boat(boat4)
    port.add_boat(boat5)
    port.add_boat(boat6)
    port.add_boat(boat7)
    port.add_boat(boat8)
    port.add_boat(boat9)
    port.add_boat(boat10)
    port.add_boat(boat11)
    port.add_boat(boat12)
    port.add_boat(boat13)
    port.add_boat(boat14)
    port.add_boat(boat15)
    port.add_boat(boat16)
    port.add_boat(boat17)
    port.add_boat(boat18)
    port.add_boat(boat19)
    port.add_boat(boat20)
    
    # Add chargers in the same order as the optimizer test
    port.add_charger(charger11)
    port.add_charger(charger12)
    port.add_charger(charger13)
    port.add_charger(charger14)
    port.add_charger(charger15)
    port.add_charger(charger16)
    port.add_charger(charger17)
    port.add_charger(charger18)
    port.add_charger(charger19)
    port.add_charger(charger20)
    port.add_charger(charger1)
    port.add_charger(charger2)
    port.add_charger(charger3)
    port.add_charger(charger4)
    port.add_charger(charger5)
    port.add_charger(charger6)
    port.add_charger(charger7)
    port.add_charger(charger8)
    port.add_charger(charger9)
    port.add_charger(charger10)
    # No PV or BESS - testing pure power limit constraint

    print(f"\nPort: {port.name}")
    print(f"Contracted Power: {port.contracted_power} kW")
    print(f"Boats: 20 boats (SeaBreeze_1 through SeaBreeze_20)")
    print(f"Chargers: 20 chargers (FastCharger_A through FastCharger_T)")
    print(f"Total Charger Capacity: {sum(c.max_power for c in port.chargers)} kW")
    print(f"\n⚠️  Note: Total charger capacity ({sum(c.max_power for c in port.chargers)} kW) > ")
    print(f"   Contracted power ({port.contracted_power} kW), so power limiting will be active.\n")

    # Simulation settings - power_limit_mode enabled, optimizer disabled
    settings = Settings(
        timestep=900,  # 15 minutes
        mode=SimulationMode.BATCH,
        db_path="20_vessels_baseline_no_der.db",
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
    print("✓ Baseline test completed (20 vessels, no DER)")
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
