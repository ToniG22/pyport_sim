"""Test configuration: 1 Boat, 1 Charger, 1 PV, 1 BESS (Full configuration)."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Port, Boat, Charger, PV, BESS, BESSControlStrategy
from config import Settings, SimulationMode
from database import DatabaseManager
from simulation import SimulationEngine


def run_simulation(use_optimizer: bool = False):
    """
    Run simulation with 1 boat, 1 charger, 1 PV, and 1 BESS.
    
    Args:
        use_optimizer: Whether to use the optimizer (SCIP)
    """
    optimizer_str = "with_optimizer" if use_optimizer else "no_optimizer"
    db_name = f"test_boat_charger_pv_bess_{optimizer_str}.db"
    
    print("=" * 60)
    print(f"Test: 1 Boat, 1 Charger, 1 PV, 1 BESS ({optimizer_str})")
    print("=" * 60)

    # Port configuration
    port = Port(
        name="Test Port - Full Config",
        contracted_power=40,
        lat=32.64542,
        lon=-16.90841,
    )

    # Boat configuration
    boat = Boat(
        name="TestBoat",
        motor_power=120,
        weight=2500,
        length=8.5,
        battery_capacity=150,
        range_speed=15.0,
        soc=0.30,
    )

    # Charger configuration
    charger = Charger(
        name="TestCharger",
        max_power=22,
        efficiency=0.95,
    )

    # PV system configuration
    pv_system = PV(
        name="TestPV",
        capacity=30.0,
        tilt=30.0,
        azimuth=180.0,
        efficiency=0.85,
        latitude=port.lat,
        longitude=port.lon,
    )

    # BESS (Battery Energy Storage System) configuration
    bess = BESS(
        name="TestBESS",
        capacity=100.0,
        max_charge_power=25.0,
        max_discharge_power=25.0,
        efficiency=0.90,
        soc_min=0.10,
        soc_max=0.90,
        initial_soc=0.50,
        control_strategy=BESSControlStrategy.DEFAULT,
    )

    # Add components to port
    port.add_boat(boat)
    port.add_charger(charger)
    port.add_pv(pv_system)
    port.add_bess(bess)

    print(f"\nPort: {port}")
    print(f"Boats: {boat.name}")
    print(f"Chargers: {charger.name}")
    print(f"PV: {pv_system}")
    print(f"BESS: {bess}")
    print(f"Optimizer: {use_optimizer}")

    # Simulation settings
    settings = Settings(
        timestep=900,
        mode=SimulationMode.BATCH,
        db_path=db_name,
        use_optimizer=use_optimizer,
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
    print(f"✓ Results saved to: {settings.db_path}")
    print("=" * 60)


def run_with_optimizer():
    """Run simulation with optimizer enabled."""
    run_simulation(use_optimizer=True)


def run_without_optimizer():
    """Run simulation without optimizer."""
    run_simulation(use_optimizer=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test: 1 Boat, 1 Charger, 1 PV, 1 BESS")
    parser.add_argument(
        "--optimizer",
        action="store_true",
        help="Enable optimizer (SCIP)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both with and without optimizer",
    )
    
    args = parser.parse_args()
    
    if args.both:
        print("\n>>> Running WITHOUT optimizer <<<\n")
        run_without_optimizer()
        print("\n>>> Running WITH optimizer <<<\n")
        run_with_optimizer()
    else:
        run_simulation(use_optimizer=args.optimizer)
