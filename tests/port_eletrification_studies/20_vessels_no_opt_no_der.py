"""Main entry point for the electric port simulator."""

from models import Port, Boat, Charger, PV, BESS, BESSControlStrategy
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def main():
    """Main function to run the simulator."""
    print("=" * 60)
    print("Electric Port Simulator")
    print("=" * 60)

    # ========================================================================
    # CONFIGURATION SECTION - Modify these values to customize your simulation
    # ========================================================================

    # Port configuration
    #  Change port name, contracted power, and location as needed
    port = Port(
        name="Funchal",
        contracted_power=80,  #  Adjust contracted power limit (kW)
        lat=32.64542,  #  Set port latitude
        lon=-16.90841,  #  Set port longitude
        tariff_path="assets/tariff/default_tariff.json",
    )

    # Boat configuration
    #  Add/remove boats, modify boat parameters (motor_power, weight, length,
    #       battery_capacity, range_speed, soc) as needed
    boat1 = Boat(
        name="SeaBreeze_1",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.30,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat2 = Boat(
        name="SeaBreeze_2",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat3 = Boat(
        name="SeaBreeze_3",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat4 = Boat(
        name="SeaBreeze_4",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat5 = Boat(
        # Charger configuration
        name="SeaBreeze_5",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat6 = Boat(
        name="SeaBreeze_6",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat7 = Boat(
        name="SeaBreeze_7",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat8 = Boat(
        name="SeaBreeze_8",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat9 = Boat(
        name="SeaBreeze_9",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat10 = Boat(
        name="SeaBreeze_10",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat11 = Boat(
        name="SeaBreeze_11",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat12 = Boat(
        name="SeaBreeze_12",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat13 = Boat(
        name="SeaBreeze_13",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat14 = Boat(
        name="SeaBreeze_14",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat15 = Boat(
        name="SeaBreeze_15",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat16 = Boat(
        name="SeaBreeze_16",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat17 = Boat(
        name="SeaBreeze_17",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat18 = Boat(
        name="SeaBreeze_18",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat19 = Boat(
        name="SeaBreeze_19",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    boat20 = Boat(
        name="SeaBreeze_20",
        motor_power=100,  #  Adjust motor power (kW)
        weight=2500,  #  Adjust weight (kg)
        length=8.5,  #  Adjust length (m)
        battery_capacity=100,  #  Adjust battery capacity (kWh)
        range_speed=16.0,  #  Adjust range speed (knots)
        soc=0.50,  #  Adjust initial state of charge (0.0-1.0)
    )

    charger1 = Charger(
        name="FastCharger_A", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger2 = Charger(
        name="FastCharger_B", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger3 = Charger(
        name="FastCharger_C", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger4 = Charger(
        name="FastCharger_D", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger5 = Charger(
        name="FastCharger_E", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger6 = Charger(
        name="FastCharger_F", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger7 = Charger(
        name="FastCharger_G", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger8 = Charger(
        name="FastCharger_H", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger9 = Charger(
        name="FastCharger_I", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger10 = Charger(
        name="FastCharger_J", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger11 = Charger(
        name="FastCharger_K", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger12 = Charger(
        name="FastCharger_L", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger13 = Charger(
        name="FastCharger_M", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger14 = Charger(
        name="FastCharger_N", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger15 = Charger(
        name="FastCharger_O", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger16 = Charger(
        name="FastCharger_P", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger17 = Charger(
        name="FastCharger_Q", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger18 = Charger(
        name="FastCharger_R", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger19 = Charger(
        name="FastCharger_S", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency
    charger20 = Charger(
        name="FastCharger_T", max_power=22, efficiency=0.95
    )  #  Adjust max_power (kW) and efficiency

    # PV system configuration
    #  Modify PV capacity, tilt, azimuth, efficiency, or remove PV system entirely
    pv_system = PV(
        name="Solar_Array_1",
        capacity=10.0,  #  Adjust PV capacity (kW DC)
        tilt=30.0,  #  Adjust panel tilt angle (degrees)
        azimuth=180.0,  #  Adjust panel azimuth (degrees, 180 = South-facing)
        efficiency=0.95,  #  Adjust system efficiency (0.0-1.0)
        latitude=port.lat,
        longitude=port.lon,
    )

    # BESS (Battery Energy Storage System) configuration
    #  Modify BESS parameters or remove BESS entirely
    bess = BESS(
        name="Battery_Storage_1",
        capacity=100.0,  #  Adjust BESS capacity (kWh)
        max_charge_power=25.0,  #  Adjust max charge power (kW)
        max_discharge_power=25.0,  #  Adjust max discharge power (kW)
        efficiency=0.90,  #  Adjust round-trip efficiency (0.0-1.0)
        soc_min=0.10,  #  Adjust minimum SOC (0.0-1.0)
        soc_max=0.90,  #  Adjust maximum SOC (0.0-1.0)
        initial_soc=0.50,  #  Adjust initial SOC (0.0-1.0)
        control_strategy=BESSControlStrategy.DEFAULT,  #  Change control strategy if needed
    )

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
    # No PV system or BESS
    # port.add_pv(pv_system)
    # port.add_bess(bess)

    print(f"\nPort: {port}")
    print(
        f"Boats: {boat1.name}, {boat2.name}, {boat3.name}, {boat4.name}, {boat5.name} {boat6.name}, {boat7.name}, {boat8.name}, {boat9.name}, {boat10.name}, {boat11.name}, {boat12.name}, {boat13.name}, {boat14.name}, {boat15.name}, {boat16.name}, {boat17.name}, {boat18.name}, {boat19.name}, {boat20.name}"
    )
    print(
        f"Chargers: {charger1.name}, {charger2.name}, {charger3.name}, {charger4.name}, {charger5.name}, {charger6.name}, {charger7.name}, {charger8.name}, {charger9.name}, {charger10.name}, {charger11.name}, {charger12.name}, {charger13.name}, {charger14.name}, {charger15.name}, {charger16.name}, {charger17.name}, {charger18.name}, {charger19.name}, {charger20.name}"
    )
    # print(f"PV: {pv_system}")
    # print(f"BESS: {bess}")

    # Simulation settings
    #  Modify simulation parameters (timestep, mode, db_path, use_optimizer, start_date, days)
    settings = Settings(
        timestep=900,  #  Adjust timestep duration (seconds, 900 = 15 minutes)
        mode=SimulationMode.BATCH,  #  Change to SimulationMode.REALTIME for real-time simulation
        db_path="20_vessels_no_opt_no_der.db",  #  Change database file path if needed
        use_optimizer=False,  #  Set to False to disable optimization (SCIP)
        power_limit_mode=True,
    )

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    db_manager.initialize_schema()
    db_manager.initialize_default_metrics()

    # Create and run simulation
    #  Modify start_date (None = current time) and days (simulation duration)
    sim = SimulationEngine(
        port=port,
        settings=settings,
        db_manager=db_manager,
        start_date="2025-09-01",  #  Set start date as "YYYY-MM-DD" string, datetime, or None for current time
        days=1,  #  Adjust simulation duration (days)
    )

    sim.run()

    print("\n" + "=" * 60)
    print(f"✓ Results saved to: {settings.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
