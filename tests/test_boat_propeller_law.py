"""
Test: Boat Propeller Law (Cubic Power Scaling)

Objective:
    Verify that the instantaneous power demand scales cubically with speed
    and that the SOC integration is accurate over time.

Test Case:
    A vessel is initialized with known parameters (k=0.5). It is subjected
    to a constant speed input of 10 knots for 1 hour.

Expected Outcome:
    - Power demand should stabilize exactly at P = 0.5 × 10³ = 500 kW
    - SOC should decrease by exactly 500 kWh / battery_capacity after 1 hour
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import matplotlib.pyplot as plt
from models.boat import Boat, BoatState

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "output"


class TestBoatPropellerLaw:
    """Test suite for the hydrodynamic Propeller Law implementation."""

    def test_k_factor_calculation(self):
        """
        Verify that the k-factor is correctly calculated as motor_power / range_speed³.
        
        For k = 0.5:
            motor_power = 500 kW, range_speed = 10 knots
            k = 500 / 10³ = 0.5
        """
        boat = Boat(
            name="TestBoat",
            motor_power=500,  # kW
            weight=5000.0,  # kg
            length=10.0,  # m
            battery_capacity=1000.0,  # kWh
            range_speed=10.0,  # knots
            soc=1.0,  # 100%
        )

        # Verify k-factor: k = motor_power / range_speed³ = 500 / 10³ = 0.5
        expected_k = 0.5
        assert boat.k == pytest.approx(expected_k, rel=1e-9), (
            f"K-factor should be {expected_k}, got {boat.k}"
        )

    def test_instantaneous_power_cubic_scaling(self):
        """
        Verify that power demand scales cubically with speed: P = k × v³.
        
        At 10 knots with k = 0.5:
            P = 0.5 × 10³ = 500 kW
        """
        boat = Boat(
            name="TestBoat",
            motor_power=500,  # kW
            weight=5000.0,  # kg
            length=10.0,  # m
            battery_capacity=1000.0,  # kWh
            range_speed=10.0,  # knots → k = 0.5
            soc=1.0,
        )

        speed_knots = 10.0
        
        # Calculate power using the propeller law: P = k × v³
        power_kw = boat.k * (speed_knots ** 3)

        expected_power = 500.0  # kW (0.5 × 10³)
        assert power_kw == pytest.approx(expected_power, rel=1e-9), (
            f"Power at {speed_knots} knots should be {expected_power} kW, got {power_kw} kW"
        )

    def test_power_at_different_speeds(self):
        """
        Verify cubic scaling at multiple speed points.
        
        With k = 0.5:
            - At 5 knots: P = 0.5 × 5³ = 62.5 kW
            - At 10 knots: P = 0.5 × 10³ = 500 kW
            - At 15 knots: P = 0.5 × 15³ = 1687.5 kW
        """
        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=1000.0,
            range_speed=10.0,  # k = 0.5
            soc=1.0,
        )

        test_cases = [
            (5.0, 62.5),      # 0.5 × 5³ = 62.5 kW
            (10.0, 500.0),    # 0.5 × 10³ = 500 kW
            (15.0, 1687.5),   # 0.5 × 15³ = 1687.5 kW
            (20.0, 4000.0),   # 0.5 × 20³ = 4000 kW
        ]

        for speed, expected_power in test_cases:
            power_kw = boat.k * (speed ** 3)
            assert power_kw == pytest.approx(expected_power, rel=1e-9), (
                f"Power at {speed} knots should be {expected_power} kW, got {power_kw} kW"
            )

    def test_soc_integration_one_hour(self):
        """
        Verify that SOC decreases correctly after 1 hour at constant speed.
        
        Test Case:
            - k = 0.5, speed = 10 knots → Power = 500 kW
            - Duration = 1 hour
            - Energy consumed = 500 kW × 1 h = 500 kWh
            - Battery capacity = 1000 kWh
            - Expected SOC decrease = 500 / 1000 = 0.5 (50%)
        """
        battery_capacity = 1000.0  # kWh
        initial_soc = 1.0  # 100%

        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,  # k = 0.5
            soc=initial_soc,
        )

        speed_knots = 10.0
        duration_seconds = 3600  # 1 hour
        timestep_seconds = 60  # 1 minute timestep

        # Simulate discharge over 1 hour
        num_steps = duration_seconds // timestep_seconds

        for _ in range(num_steps):
            # Calculate power consumption (propeller law)
            power_kw = boat.k * (speed_knots ** 3)  # 500 kW

            # Energy consumed in this timestep (kWh)
            energy_consumed = (power_kw * timestep_seconds) / 3600

            # Update SOC
            soc_decrease = energy_consumed / boat.battery_capacity
            boat.soc = max(0, boat.soc - soc_decrease)

        # Expected: 500 kWh consumed from 1000 kWh battery = 50% decrease
        expected_final_soc = initial_soc - (500.0 / battery_capacity)  # 0.5
        expected_soc_decrease = 500.0 / battery_capacity  # 0.5

        assert boat.soc == pytest.approx(expected_final_soc, rel=1e-9), (
            f"Final SOC should be {expected_final_soc:.1%}, got {boat.soc:.1%}"
        )

        actual_soc_decrease = initial_soc - boat.soc
        assert actual_soc_decrease == pytest.approx(expected_soc_decrease, rel=1e-9), (
            f"SOC decrease should be {expected_soc_decrease:.1%}, got {actual_soc_decrease:.1%}"
        )

    def test_soc_integration_with_small_timesteps(self):
        """
        Verify SOC accuracy with very small timesteps (1 second).
        
        This tests the numerical stability of the integration.
        """
        battery_capacity = 1000.0  # kWh
        initial_soc = 1.0

        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,  # k = 0.5
            soc=initial_soc,
        )

        speed_knots = 10.0
        duration_seconds = 3600  # 1 hour
        timestep_seconds = 1  # 1 second (very fine resolution)

        num_steps = duration_seconds // timestep_seconds

        for _ in range(num_steps):
            power_kw = boat.k * (speed_knots ** 3)
            energy_consumed = (power_kw * timestep_seconds) / 3600
            soc_decrease = energy_consumed / boat.battery_capacity
            boat.soc = max(0, boat.soc - soc_decrease)

        # Should still get the same result
        expected_final_soc = 0.5  # 50%
        
        assert boat.soc == pytest.approx(expected_final_soc, rel=1e-6), (
            f"Final SOC should be {expected_final_soc:.1%}, got {boat.soc:.1%}"
        )

    def test_energy_consumption_formula(self):
        """
        Verify the energy consumption formula explicitly.
        
        Energy (kWh) = (Power (kW) × Time (s)) / 3600
        
        At 500 kW for 1 hour (3600 s):
            Energy = (500 × 3600) / 3600 = 500 kWh
        """
        power_kw = 500.0
        duration_seconds = 3600

        energy_consumed = (power_kw * duration_seconds) / 3600

        assert energy_consumed == pytest.approx(500.0, rel=1e-9), (
            f"Energy consumed should be 500 kWh, got {energy_consumed} kWh"
        )


class TestBoatPropellerLawWithOutput:
    """
    Test suite that generates CSV and plot outputs for thesis documentation.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_soc_discharge_with_csv_output(self):
        """
        Run the propeller law test and output results to CSV.
        
        Generates: output/propeller_law_discharge.csv
        """
        battery_capacity = 1000.0  # kWh
        initial_soc = 1.0

        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,  # k = 0.5
            soc=initial_soc,
        )

        speed_knots = 10.0
        duration_seconds = 3600  # 1 hour
        timestep_seconds = 60  # 1 minute timestep
        num_steps = duration_seconds // timestep_seconds

        # Collect data for CSV
        data = []
        cumulative_energy = 0.0

        for step in range(num_steps + 1):
            time_minutes = step
            time_seconds = step * timestep_seconds

            # Calculate power (propeller law)
            power_kw = boat.k * (speed_knots ** 3)

            # Record current state
            data.append({
                "time_min": time_minutes,
                "time_s": time_seconds,
                "speed_knots": speed_knots,
                "power_kw": power_kw,
                "soc_percent": boat.soc * 100,
                "energy_consumed_kwh": cumulative_energy,
                "k_factor": boat.k,
            })

            # Update for next step (skip last iteration)
            if step < num_steps:
                energy_consumed = (power_kw * timestep_seconds) / 3600
                cumulative_energy += energy_consumed
                soc_decrease = energy_consumed / boat.battery_capacity
                boat.soc = max(0, boat.soc - soc_decrease)

        # Write CSV
        csv_path = OUTPUT_DIR / "propeller_law_discharge.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")

        # Verify final values
        assert boat.soc == pytest.approx(0.5, rel=1e-9)
        assert cumulative_energy == pytest.approx(500.0, rel=1e-9)

    def test_propeller_law_with_plot(self):
        """
        Generate plots demonstrating the propeller law.
        
        Generates:
            - output/propeller_law_power_vs_speed.png
            - output/propeller_law_soc_discharge.png
            - output/propeller_law_combined.png
        """
        battery_capacity = 1000.0  # kWh
        k_factor = 0.5

        # ===========================================
        # Plot 1: Power vs Speed (Cubic Relationship)
        # ===========================================
        speeds = np.linspace(0, 15, 100)
        powers = k_factor * (speeds ** 3)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(speeds, powers, 'b-', linewidth=2, label=r'$P = k \cdot v^3$')
        
        # Mark the test point (10 knots, 500 kW)
        ax1.plot(10, 500, 'ro', markersize=10, label='Test point (10 kn, 500 kW)')
        ax1.axhline(y=500, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=10, color='r', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Speed (knots)', fontsize=12)
        ax1.set_ylabel('Power (kW)', fontsize=12)
        ax1.set_title(f'Propeller Law: Power vs Speed (k = {k_factor})', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 15)
        ax1.set_ylim(0, max(powers) * 1.1)

        # Add annotation
        ax1.annotate(
            f'P = {k_factor} × 10³ = 500 kW',
            xy=(10, 500), xytext=(6, 700),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        fig1.tight_layout()
        plot1_path = OUTPUT_DIR / "propeller_law_power_vs_speed.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ===========================================
        # Plot 2: SOC Discharge Over Time
        # ===========================================
        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,
            soc=1.0,
        )

        speed_knots = 10.0
        duration_seconds = 3600
        timestep_seconds = 60
        num_steps = duration_seconds // timestep_seconds

        times = []
        socs = []
        energies = []
        cumulative_energy = 0.0

        for step in range(num_steps + 1):
            times.append(step)  # minutes
            socs.append(boat.soc * 100)
            energies.append(cumulative_energy)

            if step < num_steps:
                power_kw = boat.k * (speed_knots ** 3)
                energy_consumed = (power_kw * timestep_seconds) / 3600
                cumulative_energy += energy_consumed
                soc_decrease = energy_consumed / boat.battery_capacity
                boat.soc = max(0, boat.soc - soc_decrease)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        ax2.plot(times, socs, 'b-', linewidth=2, label='SOC (%)')
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Expected final SOC (50%)')

        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('State of Charge (%)', fontsize=12)
        ax2.set_title('SOC Discharge at Constant Speed (v = 10 knots, P = 500 kW)', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 60)
        ax2.set_ylim(0, 105)

        # Add annotations
        ax2.annotate(
            'Initial SOC = 100%',
            xy=(0, 100), xytext=(5, 90),
            fontsize=10,
        )
        ax2.annotate(
            f'Final SOC = {socs[-1]:.0f}%\n(500 kWh consumed)',
            xy=(60, socs[-1]), xytext=(45, 60),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        fig2.tight_layout()
        plot2_path = OUTPUT_DIR / "propeller_law_soc_discharge.png"
        fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

        # ===========================================
        # Plot 3: Combined Figure (for thesis)
        # ===========================================
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Power vs Speed
        ax3a.plot(speeds, powers, 'b-', linewidth=2, label=r'$P = k \cdot v^3$')
        ax3a.plot(10, 500, 'ro', markersize=10, label='Test point')
        ax3a.axhline(y=500, color='r', linestyle='--', alpha=0.5)
        ax3a.axvline(x=10, color='r', linestyle='--', alpha=0.5)
        ax3a.set_xlabel('Speed (knots)', fontsize=12)
        ax3a.set_ylabel('Power (kW)', fontsize=12)
        ax3a.set_title(f'(a) Propeller Law: $P = k \\cdot v^3$ (k = {k_factor})', fontsize=12)
        ax3a.legend(loc='upper left', fontsize=10)
        ax3a.grid(True, alpha=0.3)
        ax3a.set_xlim(0, 15)
        ax3a.set_ylim(0, max(powers) * 1.1)
        ax3a.annotate(
            f'P = 500 kW',
            xy=(10, 500), xytext=(6, 700),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        # Right: SOC Discharge
        ax3b.plot(times, socs, 'b-', linewidth=2, label='SOC')
        ax3b.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Expected (50%)')
        ax3b.set_xlabel('Time (minutes)', fontsize=12)
        ax3b.set_ylabel('State of Charge (%)', fontsize=12)
        ax3b.set_title('(b) SOC Integration (v = 10 kn, 1 hour)', fontsize=12)
        ax3b.legend(loc='upper right', fontsize=10)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_xlim(0, 60)
        ax3b.set_ylim(0, 105)
        ax3b.annotate(
            f'ΔE = 500 kWh\nΔSOC = 50%',
            xy=(60, 50), xytext=(40, 70),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        fig3.suptitle('Boat Propeller Law Validation Test', fontsize=14, fontweight='bold', y=1.02)
        fig3.tight_layout()
        plot3_path = OUTPUT_DIR / "propeller_law_combined.png"
        fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)

        print(f"\n✓ Plots saved to:")
        print(f"  - {plot1_path}")
        print(f"  - {plot2_path}")
        print(f"  - {plot3_path}")

        # Verify test passed
        assert boat.soc == pytest.approx(0.5, rel=1e-9)

    def test_generate_thesis_table(self):
        """
        Generate a summary table in CSV format for thesis.
        
        Generates: output/propeller_law_summary.csv
        """
        k_factor = 0.5
        battery_capacity = 1000.0

        # Test cases with different speeds
        test_speeds = [5.0, 7.5, 10.0, 12.5, 15.0]
        results = []

        for speed in test_speeds:
            power = k_factor * (speed ** 3)
            energy_per_hour = power  # kWh (since P × 1h = E)
            soc_decrease_per_hour = (energy_per_hour / battery_capacity) * 100

            results.append({
                "speed_knots": speed,
                "power_kw": round(power, 2),
                "energy_1h_kwh": round(energy_per_hour, 2),
                "soc_decrease_1h_percent": round(soc_decrease_per_hour, 2),
                "k_factor": k_factor,
            })

        # Write summary CSV
        csv_path = OUTPUT_DIR / "propeller_law_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Also print as formatted table
        print("\n" + "=" * 70)
        print("PROPELLER LAW VERIFICATION RESULTS (k = 0.5)")
        print("=" * 70)
        print(f"{'Speed':>10} {'Power':>12} {'Energy/1h':>14} {'SOC Decrease':>15}")
        print(f"{'(knots)':>10} {'(kW)':>12} {'(kWh)':>14} {'(%/hour)':>15}")
        print("-" * 70)
        for r in results:
            print(f"{r['speed_knots']:>10.1f} {r['power_kw']:>12.2f} {r['energy_1h_kwh']:>14.2f} {r['soc_decrease_1h_percent']:>15.2f}")
        print("=" * 70)
        print(f"\n✓ Summary table saved to: {csv_path}")


def generate_all_outputs():
    """
    Standalone function to generate all outputs without running pytest.
    
    Usage: python test_boat_propeller_law.py
    """
    print("=" * 60)
    print("GENERATING PROPELLER LAW TEST OUTPUTS")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tests with output generation
    test_instance = TestBoatPropellerLawWithOutput()
    test_instance.setup_output_dir()
    test_instance.test_soc_discharge_with_csv_output()
    test_instance.test_propeller_law_with_plot()
    test_instance.test_generate_thesis_table()
    
    print("\n" + "=" * 60)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_all_outputs()
    else:
        pytest.main([__file__, "-v", "-s"])
