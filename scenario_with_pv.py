"""Scenario with PV solar generation."""

from models import Port, Boat, Charger, PV
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_pv_scenario():
    """Run a simulation scenario with solar PV."""
    print("=" * 60)
    print("SCENARIO: With Solar PV Generation")
    print("=" * 60)

    # Create port (Madeira coordinates)
    port = Port(
        name="Marina del Sol",
        contracted_power=100,  # Reduced to 50 kW to see PV impact
        lat=32.64542,
        lon=-16.90841,
    )

    # Create 1 boat (starting at 30% charge)
    boat1 = Boat(
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

    # Create 2 chargers
    charger1 = Charger(name="FastCharger_A", max_power=22, efficiency=0.95)
    charger2 = Charger(name="FastCharger_B", max_power=22, efficiency=0.95)

    # Create PV system (30 kW solar array)
    pv_system = PV(
        name="Solar_Array_1",
        capacity=22.0,  # 22 kW DC
        tilt=30.0,  # 30 degrees (optimal for Madeira)
        azimuth=180.0,  # South-facing
        efficiency=0.95,  # 95% system efficiency
        latitude=port.lat,
        longitude=port.lon,
    )

    # Add components to port
    port.add_boat(boat1)
    port.add_boat(boat2)
    port.add_charger(charger1)
    port.add_charger(charger2)
    port.add_pv(pv_system)

    print(f"\nPort: {port}")
    print(f"Boats: {boat1.name}, {boat2.name}")
    print(f"Chargers: {charger1.name}, {charger2.name}")
    print(f"PV: {pv_system}")
    print(f"\nPort Power Budget:")
    print(f"  Contracted: {port.contracted_power} kW")
    print(f"  PV Capacity: {pv_system.capacity} kW")
    print(f"  Total (daytime): up to {port.contracted_power + pv_system.capacity} kW")

    # Configure simulation
    settings = Settings(
        timestep=900,  # 15 minutes
        mode=SimulationMode.BATCH,
        db_path="pv_scenario.db",
    )

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()

    # Create and run simulation
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date=None,
        days=1,
    )

    sim.run()

    # Query and display results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    # Get PV production data
    pv_data = db_manager.get_measurements(source=pv_system.name, metric="production")
    if pv_data:
        print(f"\nPV Production Summary:")
        production_values = [row["value"] for row in pv_data]
        total_energy = sum(production_values) * (settings.timestep / 3600)  # kWh
        avg_power = sum(production_values) / len(production_values)
        max_power = max(production_values)
        print(f"  Total Energy: {total_energy:.2f} kWh")
        print(f"  Average Power: {avg_power:.2f} kW")
        print(f"  Peak Power: {max_power:.2f} kW")
        print(f"  Capacity Factor: {(avg_power / pv_system.capacity) * 100:.1f}%")

    # Get port power data
    port_power = db_manager.get_measurements(source="port", metric="pv_production")
    if port_power:
        print(f"\nPort Power Summary:")
        # Get contracted power usage
        used_data = db_manager.get_measurements(
            source="port", metric="total_power_used"
        )
        if used_data:
            used_values = [row["value"] for row in used_data]
            avg_used = sum(used_values) / len(used_values)
            max_used = max(used_values)
            print(f"  Avg Charger Load: {avg_used:.2f} kW")
            print(f"  Peak Charger Load: {max_used:.2f} kW")

        # Calculate grid vs solar split
        pv_values = [row["value"] for row in port_power]
        total_pv_energy = sum(pv_values) * (settings.timestep / 3600)
        total_used_energy = sum(used_values) * (settings.timestep / 3600)
        grid_energy = max(0, total_used_energy - total_pv_energy)

        print(f"\nEnergy Sources:")
        print(
            f"  Solar: {total_pv_energy:.2f} kWh ({(total_pv_energy/max(total_used_energy, 0.1))*100:.1f}%)"
        )
        print(
            f"  Grid: {grid_energy:.2f} kWh ({(grid_energy/max(total_used_energy, 0.1))*100:.1f}%)"
        )
        print(f"  Total Used: {total_used_energy:.2f} kWh")

    # Show sample weather data
    weather_data = db_manager.get_forecasts(source="openmeteo", metric="ghi")
    if weather_data:
        print(f"\nWeather Forecast Loaded:")
        print(f"  {len(weather_data)} hours of data")
        # Show midday irradiance
        noon_data = [row for row in weather_data if "12:00:00" in row["timestamp"]]
        if noon_data:
            print(f"  Midday GHI: {noon_data[0]['value']:.0f} W/m²")

    print("\n" + "=" * 60)
    print(f"✓ Results saved to: {settings.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_pv_scenario()
