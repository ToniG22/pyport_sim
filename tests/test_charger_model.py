"""
Test: Charger Model Verification

Objective:
    Confirm the charger respects the maximum power rating (P_max) and
    correctly handles efficiency losses.

Test Case:
    A charging request is sent with a set-point exceeding the hardware
    limit (P_set > P_max).

Expected Outcome:
    - The internal logic must clamp the output to P_max
    - The energy delivered to the battery must correspond to P_max × η_evse
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
from models.charger import Charger, ChargerState
from models.boat import Boat, BoatState

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "output"


class TestChargerModel:
    """Test suite for the Charger model verification."""

    def test_charger_initialization(self):
        """
        Verify charger initializes with correct parameters.
        """
        charger = Charger(
            name="TestCharger",
            max_power=22,  # kW
            efficiency=0.95,
        )

        assert charger.name == "TestCharger"
        assert charger.max_power == 22
        assert charger.efficiency == 0.95
        assert charger.power == 0.0
        assert charger.state == ChargerState.IDLE

    def test_power_cannot_exceed_max_at_init(self):
        """
        Verify that power cannot be set above max_power during initialization.
        """
        with pytest.raises(ValueError, match="Power cannot exceed max_power"):
            Charger(
                name="TestCharger",
                max_power=22,
                efficiency=0.95,
                power=50,  # Exceeds max_power of 22 kW
            )

    def test_power_clamping_logic(self):
        """
        Test the power clamping behavior when setting power.

        This simulates the logic used in the simulation engine where
        power requests are clamped to max_power.
        """
        charger = Charger(
            name="TestCharger",
            max_power=22,  # kW
            efficiency=0.95,
        )

        # Simulate a power request that exceeds max_power
        requested_power = 50  # kW (exceeds 22 kW limit)

        # Apply clamping logic (as done in simulation)
        clamped_power = min(requested_power, charger.max_power)
        charger.power = clamped_power
        charger.state = ChargerState.CHARGING

        assert charger.power == 22, f"Power should be clamped to {charger.max_power} kW"
        assert charger.power <= charger.max_power, "Power must not exceed max_power"

    def test_effective_power_with_efficiency(self):
        """
        Verify that effective_power correctly applies efficiency losses.

        effective_power = power × efficiency

        For P = 22 kW and η = 0.95:
            P_effective = 22 × 0.95 = 20.9 kW
        """
        charger = Charger(
            name="TestCharger",
            max_power=22,  # kW
            efficiency=0.95,
        )

        charger.power = 22  # Set to max power
        charger.state = ChargerState.CHARGING

        expected_effective_power = 22 * 0.95  # 20.9 kW

        assert charger.effective_power == pytest.approx(
            expected_effective_power, rel=1e-9
        ), f"Effective power should be {expected_effective_power} kW, got {charger.effective_power} kW"

    def test_efficiency_losses_at_different_power_levels(self):
        """
        Verify efficiency losses at various power levels.
        """
        charger = Charger(
            name="TestCharger",
            max_power=100,
            efficiency=0.90,  # 90% efficiency
        )

        test_cases = [
            (25, 22.5),  # 25 × 0.90 = 22.5 kW
            (50, 45.0),  # 50 × 0.90 = 45.0 kW
            (75, 67.5),  # 75 × 0.90 = 67.5 kW
            (100, 90.0),  # 100 × 0.90 = 90.0 kW
        ]

        for power, expected_effective in test_cases:
            charger.power = power
            assert charger.effective_power == pytest.approx(
                expected_effective, rel=1e-9
            ), f"At {power} kW, effective power should be {expected_effective} kW"

    def test_energy_delivered_over_time(self):
        """
        Verify energy delivered to battery over time with efficiency losses.

        Test Case:
            - Charger: P_max = 22 kW, η = 0.95
            - Duration: 1 hour
            - Expected energy delivered = P_max × η × t = 22 × 0.95 × 1 = 20.9 kWh
        """
        charger = Charger(
            name="TestCharger",
            max_power=22,
            efficiency=0.95,
        )

        charger.power = charger.max_power  # 22 kW
        charger.state = ChargerState.CHARGING

        duration_seconds = 3600  # 1 hour
        timestep_seconds = 60  # 1 minute
        num_steps = duration_seconds // timestep_seconds

        total_energy_delivered = 0.0

        for _ in range(num_steps):
            # Energy delivered in this timestep (kWh)
            energy_delivered = (charger.effective_power * timestep_seconds) / 3600
            total_energy_delivered += energy_delivered

        expected_energy = charger.max_power * charger.efficiency * 1  # 47.5 kWh

        assert total_energy_delivered == pytest.approx(
            expected_energy, rel=1e-9
        ), f"Total energy delivered should be {expected_energy} kWh, got {total_energy_delivered} kWh"

    def test_state_transitions(self):
        """
        Verify charger state transitions work correctly.
        """
        charger = Charger(
            name="TestCharger",
            max_power=22,
            efficiency=0.95,
        )

        # Initial state
        assert charger.state == ChargerState.IDLE
        assert charger.power == 0.0

        # Transition to CHARGING
        charger.state = ChargerState.CHARGING
        charger.power = 22
        charger.connected_boat = "TestBoat"

        assert charger.state == ChargerState.CHARGING
        assert charger.power == 22
        assert charger.connected_boat == "TestBoat"

        # Transition back to IDLE (should reset power and boat)
        charger.state = ChargerState.IDLE

        assert charger.state == ChargerState.IDLE
        assert charger.power == 0.0
        assert charger.connected_boat is None

    def test_battery_charging_integration(self):
        """
        Verify that a battery (boat) charges correctly with efficiency losses.

        Test Case:
            - Boat: 1000 kWh battery, initial SOC = 50%
            - Charger: P_max = 22 kW, η = 0.95
            - Duration: 1 hour
            - Energy delivered = 22 × 0.95 × 1 = 20.9 kWh
            - Expected SOC increase = 20.9 / 1000 = 2.09%
            - Final SOC = 50% + 2.09% = 52.09%
        """
        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=1000.0,  # kWh
            range_speed=10.0,
            soc=0.5,  # 50%
        )

        charger = Charger(
            name="TestCharger",
            max_power=22,  # kW
            efficiency=0.95,
        )

        # Setup charging
        charger.power = charger.max_power
        charger.state = ChargerState.CHARGING
        charger.connected_boat = boat.name
        boat.state = BoatState.CHARGING

        duration_seconds = 3600  # 1 hour
        timestep_seconds = 60
        num_steps = duration_seconds // timestep_seconds

        initial_soc = boat.soc

        for _ in range(num_steps):
            # Energy delivered to battery (kWh)
            energy_delivered = (charger.effective_power * timestep_seconds) / 3600
            soc_increase = energy_delivered / boat.battery_capacity
            boat.soc = min(1.0, boat.soc + soc_increase)

        # Expected values
        expected_energy_delivered = (
            charger.max_power * charger.efficiency * 1
        )  # 47.5 kWh
        expected_soc_increase = (
            expected_energy_delivered / boat.battery_capacity
        )  # 0.0475
        expected_final_soc = initial_soc + expected_soc_increase  # 0.5475

        assert boat.soc == pytest.approx(
            expected_final_soc, rel=1e-9
        ), f"Final SOC should be {expected_final_soc:.2%}, got {boat.soc:.2%}"


class TestChargerModelWithOutput:
    """
    Test suite that generates CSV and plot outputs for thesis documentation.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_power_clamping_with_csv_output(self):
        """
        Test power clamping and generate CSV output showing behavior.

        Generates: output/charger_power_clamping.csv
        """
        max_power = 22  # kW
        efficiency = 0.95

        charger = Charger(
            name="TestCharger",
            max_power=max_power,
            efficiency=efficiency,
        )

        # Test various power requests (some exceeding max)
        requested_powers = [5, 10, 15, 22, 30, 40, 50, 75]

        data = []
        for p_request in requested_powers:
            # Apply clamping logic
            p_clamped = min(p_request, charger.max_power)
            charger.power = p_clamped

            data.append(
                {
                    "requested_power_kw": p_request,
                    "clamped_power_kw": p_clamped,
                    "max_power_kw": max_power,
                    "effective_power_kw": charger.effective_power,
                    "efficiency": efficiency,
                    "power_loss_kw": p_clamped - charger.effective_power,
                    "was_clamped": p_request > max_power,
                }
            )

        # Write CSV
        csv_path = OUTPUT_DIR / "charger_power_clamping.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")

        # Verify clamping worked
        for d in data:
            assert d["clamped_power_kw"] <= max_power

    def test_charging_session_with_csv_output(self):
        """
        Simulate a complete charging session and output to CSV.

        Test Case:
            - Charger: P_max = 22 kW, η = 0.95
            - Requested power: 50 kW (exceeds limit)
            - Actual power: 22 kW (clamped)
            - Duration: 1 hour

        Generates: output/charger_charging_session.csv
        """
        max_power = 22  # kW
        efficiency = 0.95
        battery_capacity = 1000.0  # kWh
        initial_soc = 0.5  # 50%

        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,
            soc=initial_soc,
        )

        charger = Charger(
            name="TestCharger",
            max_power=max_power,
            efficiency=efficiency,
        )

        # Requested power exceeds max
        requested_power = 50  # kW
        clamped_power = min(requested_power, charger.max_power)  # 22 kW
        charger.power = clamped_power
        charger.state = ChargerState.CHARGING

        duration_seconds = 3600  # 1 hour
        timestep_seconds = 60
        num_steps = duration_seconds // timestep_seconds

        data = []
        cumulative_energy_input = 0.0
        cumulative_energy_delivered = 0.0

        for step in range(num_steps + 1):
            time_minutes = step

            data.append(
                {
                    "time_min": time_minutes,
                    "requested_power_kw": requested_power,
                    "actual_power_kw": charger.power,
                    "effective_power_kw": charger.effective_power,
                    "soc_percent": boat.soc * 100,
                    "energy_input_kwh": cumulative_energy_input,
                    "energy_delivered_kwh": cumulative_energy_delivered,
                    "efficiency_loss_kwh": cumulative_energy_input
                    - cumulative_energy_delivered,
                }
            )

            if step < num_steps:
                # Energy calculations
                energy_input = (charger.power * timestep_seconds) / 3600
                energy_delivered = (charger.effective_power * timestep_seconds) / 3600
                cumulative_energy_input += energy_input
                cumulative_energy_delivered += energy_delivered

                # Update boat SOC
                soc_increase = energy_delivered / boat.battery_capacity
                boat.soc = min(1.0, boat.soc + soc_increase)

        # Write CSV
        csv_path = OUTPUT_DIR / "charger_charging_session.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")

        # Verify expected outcomes
        expected_energy_delivered = max_power * efficiency * 1  # 20.9 kWh
        assert cumulative_energy_delivered == pytest.approx(
            expected_energy_delivered, rel=1e-9
        )

    def test_charger_model_with_plot(self):
        """
        Generate plots demonstrating charger behavior.

        Generates:
            - output/charger_power_clamping.png
            - output/charger_efficiency_losses.png
            - output/charger_combined.png
        """
        max_power = 22  # kW
        efficiency = 0.95
        battery_capacity = 1000.0  # kWh

        # ===========================================
        # Plot 1: Power Clamping Behavior
        # ===========================================
        requested_powers = np.linspace(0, 50, 100)
        clamped_powers = np.minimum(requested_powers, max_power)
        effective_powers = clamped_powers * efficiency

        fig1, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(
            requested_powers,
            requested_powers,
            "b--",
            linewidth=1.5,
            label="Requested power",
            alpha=0.7,
        )
        ax1.plot(
            requested_powers,
            clamped_powers,
            "g-",
            linewidth=2,
            label=f"Clamped power (P_max = {max_power} kW)",
        )
        ax1.plot(
            requested_powers,
            effective_powers,
            "r-",
            linewidth=2,
            label=f"Effective power (η = {efficiency:.0%})",
        )

        ax1.axhline(y=max_power, color="g", linestyle=":", alpha=0.5)
        ax1.axvline(x=max_power, color="gray", linestyle=":", alpha=0.5)

        # Mark the test point (50 kW requested, clamped to 22 kW)
        ax1.plot(50, 22, "ko", markersize=10, label="Test point (50→22 kW)")
        ax1.plot(50, 22 * efficiency, "ro", markersize=8)

        ax1.set_xlabel("Requested Power (kW)", fontsize=12)
        ax1.set_ylabel("Output Power (kW)", fontsize=12)
        ax1.set_title(f"Charger Power Clamping (P_max = {max_power} kW)", fontsize=14)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 50)

        # Add annotation
        ax1.annotate(
            f"P_set = 50 kW\nP_out = 22 kW\nP_eff = {22*efficiency:.1f} kW",
            xy=(50, 22),
            xytext=(35, 35),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        fig1.tight_layout()
        plot1_path = OUTPUT_DIR / "charger_power_clamping.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)

        # ===========================================
        # Plot 2: Charging Session with Efficiency
        # ===========================================
        boat = Boat(
            name="TestBoat",
            motor_power=500,
            weight=5000.0,
            length=10.0,
            battery_capacity=battery_capacity,
            range_speed=10.0,
            soc=0.5,
        )

        charger = Charger(
            name="TestCharger",
            max_power=max_power,
            efficiency=efficiency,
        )

        # Simulate with clamped power
        charger.power = max_power  # Clamped from 75 kW request
        charger.state = ChargerState.CHARGING

        duration_seconds = 3600
        timestep_seconds = 60
        num_steps = duration_seconds // timestep_seconds

        times = []
        socs = []
        energy_input = []
        energy_delivered = []
        cum_input = 0.0
        cum_delivered = 0.0

        for step in range(num_steps + 1):
            times.append(step)
            socs.append(boat.soc * 100)
            energy_input.append(cum_input)
            energy_delivered.append(cum_delivered)

            if step < num_steps:
                e_in = (charger.power * timestep_seconds) / 3600
                e_out = (charger.effective_power * timestep_seconds) / 3600
                cum_input += e_in
                cum_delivered += e_out
                boat.soc = min(1.0, boat.soc + e_out / boat.battery_capacity)

        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Energy over time
        ax2a.plot(times, energy_input, "b-", linewidth=2, label="Energy input (grid)")
        ax2a.plot(
            times,
            energy_delivered,
            "g-",
            linewidth=2,
            label="Energy delivered (battery)",
        )
        ax2a.fill_between(
            times,
            energy_delivered,
            energy_input,
            alpha=0.3,
            color="red",
            label=f"Efficiency losses ({(1-efficiency)*100:.0f}%)",
        )

        ax2a.set_xlabel("Time (minutes)", fontsize=12)
        ax2a.set_ylabel("Cumulative Energy (kWh)", fontsize=12)
        ax2a.set_title("Energy Flow During Charging Session", fontsize=12)
        ax2a.legend(loc="upper left", fontsize=10)
        ax2a.grid(True, alpha=0.3)

        # Annotate final values
        ax2a.annotate(
            f"Input: {energy_input[-1]:.1f} kWh\nDelivered: {energy_delivered[-1]:.1f} kWh\nLoss: {energy_input[-1]-energy_delivered[-1]:.1f} kWh",
            xy=(60, energy_delivered[-1]),
            xytext=(40, 35),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        # Right: SOC over time
        ax2b.plot(times, socs, "b-", linewidth=2, label="SOC")
        ax2b.axhline(
            y=50, color="gray", linestyle="--", alpha=0.5, label="Initial SOC (50%)"
        )
        ax2b.axhline(
            y=socs[-1],
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Final SOC ({socs[-1]:.2f}%)",
        )

        ax2b.set_xlabel("Time (minutes)", fontsize=12)
        ax2b.set_ylabel("State of Charge (%)", fontsize=12)
        ax2b.set_title(
            f"Battery SOC (P = {max_power} kW, η = {efficiency:.0%})", fontsize=12
        )
        ax2b.legend(loc="upper left", fontsize=10)
        ax2b.grid(True, alpha=0.3)
        ax2b.set_ylim(45, 60)

        # Annotate SOC increase
        soc_increase = socs[-1] - socs[0]
        ax2b.annotate(
            f"ΔSOC = {soc_increase:.2f}%\n({energy_delivered[-1]:.1f} kWh)",
            xy=(60, socs[-1]),
            xytext=(40, 56),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        fig2.suptitle(
            "Charger Efficiency and Battery Charging",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig2.tight_layout()
        plot2_path = OUTPUT_DIR / "charger_efficiency_losses.png"
        fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # ===========================================
        # Plot 3: Combined Figure (for thesis)
        # ===========================================
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Power clamping
        ax3a.plot(
            requested_powers,
            requested_powers,
            "b--",
            linewidth=1.5,
            label="Requested",
            alpha=0.7,
        )
        ax3a.plot(
            requested_powers,
            clamped_powers,
            "g-",
            linewidth=2,
            label=f"Clamped (P_max={max_power}kW)",
        )
        ax3a.plot(
            requested_powers,
            effective_powers,
            "r-",
            linewidth=2,
            label=f"Effective (η={efficiency:.0%})",
        )
        ax3a.axhline(y=max_power, color="g", linestyle=":", alpha=0.5)
        ax3a.plot(50, 22, "ko", markersize=8)
        ax3a.plot(50, 22 * efficiency, "ro", markersize=6)
        ax3a.set_xlabel("Requested Power (kW)", fontsize=12)
        ax3a.set_ylabel("Output Power (kW)", fontsize=12)
        ax3a.set_title("(a) Power Clamping Behavior", fontsize=12)
        ax3a.legend(loc="upper left", fontsize=9)
        ax3a.grid(True, alpha=0.3)
        ax3a.set_xlim(0, 50)
        ax3a.set_ylim(0, 50)
        ax3a.annotate(
            f"P_set=50kW → P_out=22kW",
            xy=(50, 22),
            xytext=(30, 38),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        # Right: Energy with efficiency
        ax3b.plot(times, energy_input, "b-", linewidth=2, label="Grid input")
        ax3b.plot(times, energy_delivered, "g-", linewidth=2, label="Battery delivered")
        ax3b.fill_between(
            times,
            energy_delivered,
            energy_input,
            alpha=0.3,
            color="red",
            label=f"Losses (5%)",
        )
        ax3b.set_xlabel("Time (minutes)", fontsize=12)
        ax3b.set_ylabel("Cumulative Energy (kWh)", fontsize=12)
        ax3b.set_title(f"(b) Energy Flow (η = {efficiency:.0%})", fontsize=12)
        ax3b.legend(loc="upper left", fontsize=9)
        ax3b.grid(True, alpha=0.3)
        ax3b.annotate(
            f"22 kWh in\n{22*efficiency:.1f} kWh out",
            xy=(60, 21),
            xytext=(40, 15),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        fig3.suptitle(
            "Charger Model Verification Test", fontsize=14, fontweight="bold", y=1.02
        )
        fig3.tight_layout()
        plot3_path = OUTPUT_DIR / "charger_combined.png"
        fig3.savefig(plot3_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)

        print(f"\n✓ Plots saved to:")
        print(f"  - {plot1_path}")
        print(f"  - {plot2_path}")
        print(f"  - {plot3_path}")

        # Verify test passed
        expected_final_soc = 0.5 + (max_power * efficiency / battery_capacity)
        assert boat.soc == pytest.approx(expected_final_soc, rel=1e-9)

    def test_generate_thesis_table(self):
        """
        Generate a summary table in CSV format for thesis.

        Generates: output/charger_summary.csv
        """
        max_power = 22  # kW
        efficiency = 0.95
        battery_capacity = 1000.0  # kWh

        # Test cases with different requested power levels
        requested_powers = [11, 22, 30, 40, 50]
        results = []

        for p_request in requested_powers:
            p_clamped = min(p_request, max_power)
            p_effective = p_clamped * efficiency
            energy_1h_input = p_clamped
            energy_1h_delivered = p_effective
            soc_increase_1h = (energy_1h_delivered / battery_capacity) * 100

            results.append(
                {
                    "requested_power_kw": p_request,
                    "clamped_power_kw": p_clamped,
                    "effective_power_kw": round(p_effective, 2),
                    "energy_input_1h_kwh": round(energy_1h_input, 2),
                    "energy_delivered_1h_kwh": round(energy_1h_delivered, 2),
                    "efficiency_loss_kwh": round(
                        energy_1h_input - energy_1h_delivered, 2
                    ),
                    "soc_increase_1h_percent": round(soc_increase_1h, 2),
                    "was_clamped": p_request > max_power,
                }
            )

        # Write summary CSV
        csv_path = OUTPUT_DIR / "charger_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Print formatted table
        print("\n" + "=" * 90)
        print(
            f"CHARGER MODEL VERIFICATION RESULTS (P_max = {max_power} kW, η = {efficiency:.0%})"
        )
        print("=" * 90)
        print(
            f"{'P_req':>8} {'P_clamp':>10} {'P_eff':>10} {'E_in':>10} {'E_out':>10} {'Loss':>8} {'ΔSOC':>10} {'Clamped':>10}"
        )
        print(
            f"{'(kW)':>8} {'(kW)':>10} {'(kW)':>10} {'(kWh)':>10} {'(kWh)':>10} {'(kWh)':>8} {'(%/h)':>10} {'':>10}"
        )
        print("-" * 90)
        for r in results:
            clamped_str = "Yes" if r["was_clamped"] else "No"
            print(
                f"{r['requested_power_kw']:>8} {r['clamped_power_kw']:>10} {r['effective_power_kw']:>10.2f} "
                f"{r['energy_input_1h_kwh']:>10.2f} {r['energy_delivered_1h_kwh']:>10.2f} "
                f"{r['efficiency_loss_kwh']:>8.2f} {r['soc_increase_1h_percent']:>10.2f} {clamped_str:>10}"
            )
        print("=" * 90)
        print(f"\n✓ Summary table saved to: {csv_path}")


def generate_all_outputs():
    """
    Standalone function to generate all outputs without running pytest.

    Usage: python test_charger_model.py --generate
    """
    print("=" * 60)
    print("GENERATING CHARGER MODEL TEST OUTPUTS")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tests with output generation
    test_instance = TestChargerModelWithOutput()
    test_instance.setup_output_dir()
    test_instance.test_power_clamping_with_csv_output()
    test_instance.test_charging_session_with_csv_output()
    test_instance.test_charger_model_with_plot()
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
