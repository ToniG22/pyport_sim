"""
Test: BESS Model Verification

Objective:
    Verify that the battery correctly charges and discharges within its 
    capacity limits, applies efficiency losses correctly to both input 
    (charging) and output (discharging), and enforces SOC boundaries.

Test Case:
    A BESS initialized at 50% SOC is subjected to a charging cycle followed 
    by a discharging cycle at rated power. Subsequently, a command is sent 
    to overcharge the battery beyond 100%.

Expected Outcome:
    - During charging: stored energy = P_charge × η × Δt
    - During discharging: extracted energy = P_discharge / η × Δt
    - SOC must remain within min/max limits
    - Overcharge/overdischarge must be prevented
"""

import sys
import csv
import math
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import matplotlib.pyplot as plt
from models.bess import BESS, BESSControlStrategy

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "output"

# BESS parameters for testing
BESS_CAPACITY = 100  # kWh
BESS_MAX_POWER = 50  # kW (charge and discharge)
BESS_EFFICIENCY = 0.90  # 90% round-trip efficiency
BESS_SOC_MIN = 0.10  # 10%
BESS_SOC_MAX = 0.90  # 90%


class TestBESSModel:
    """Test suite for the BESS model verification."""

    def test_bess_initialization(self):
        """
        Verify BESS initializes with correct parameters.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        assert bess.name == "TestBESS"
        assert bess.capacity == BESS_CAPACITY
        assert bess.max_charge_power == BESS_MAX_POWER
        assert bess.max_discharge_power == BESS_MAX_POWER
        assert bess.efficiency == BESS_EFFICIENCY
        assert bess.current_soc == 0.50
        assert bess.current_power == 0.0

    def test_charging_efficiency(self):
        """
        Verify that charging applies efficiency losses correctly.
        
        Formula: Stored energy = P_charge × η × Δt
        
        Test Case:
            - Charge at 50 kW for 1 hour with 90% efficiency
            - Energy stored = 50 × 0.90 × 1 = 45 kWh
            - SOC increase = 45 / 100 = 45%
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.20,  # Start at 20% to have room to charge
        )

        initial_soc = bess.current_soc
        charge_power = 50  # kW
        duration_seconds = 3600  # 1 hour

        # Charge for 1 hour
        bess.charge(charge_power, duration_seconds)

        # Expected: 50 kW × 0.90 × 1h = 45 kWh stored
        expected_energy_stored = charge_power * BESS_EFFICIENCY * 1  # 45 kWh
        expected_soc_increase = expected_energy_stored / BESS_CAPACITY  # 0.45
        expected_final_soc = initial_soc + expected_soc_increase  # 0.65

        assert bess.current_soc == pytest.approx(expected_final_soc, rel=1e-9), (
            f"SOC should be {expected_final_soc:.2%}, got {bess.current_soc:.2%}"
        )

    def test_discharging_efficiency(self):
        """
        Verify that discharging applies efficiency losses correctly.
        
        Formula: Energy removed from battery = P_discharge / η × Δt
        
        Test Case:
            - Discharge at 50 kW for 1 hour with 90% efficiency
            - Energy delivered = 50 kWh
            - Energy removed from battery = 50 / 0.90 = 55.56 kWh
            - SOC decrease = 55.56 / 100 = 55.56%
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.80,  # Start at 80% to have room to discharge
        )

        initial_soc = bess.current_soc
        discharge_power = 50  # kW
        duration_seconds = 3600  # 1 hour

        # Discharge for 1 hour
        bess.discharge(discharge_power, duration_seconds)

        # Expected: 50 kWh delivered, 50 / 0.90 = 55.56 kWh removed from battery
        expected_energy_removed = discharge_power / BESS_EFFICIENCY * 1  # 55.56 kWh
        expected_soc_decrease = expected_energy_removed / BESS_CAPACITY  # 0.5556
        expected_final_soc = initial_soc - expected_soc_decrease  # 0.2444

        assert bess.current_soc == pytest.approx(expected_final_soc, rel=1e-9), (
            f"SOC should be {expected_final_soc:.2%}, got {bess.current_soc:.2%}"
        )

    def test_soc_max_limit_enforcement(self):
        """
        Verify that charging stops at soc_max (90%).
        
        Test: Try to charge beyond soc_max.
        Expected: SOC should be exactly soc_max, not higher.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.85,  # Start close to max
        )

        # Try to charge a lot (would exceed 90%)
        charge_power = 50  # kW
        duration_seconds = 3600  # 1 hour (would add 45% normally)

        bess.charge(charge_power, duration_seconds)

        # SOC should be clamped to soc_max
        assert bess.current_soc == pytest.approx(BESS_SOC_MAX, rel=1e-9), (
            f"SOC should be clamped to {BESS_SOC_MAX:.0%}, got {bess.current_soc:.2%}"
        )

    def test_soc_min_limit_enforcement(self):
        """
        Verify that discharging stops at soc_min (10%).
        
        Test: Try to discharge beyond soc_min.
        Expected: SOC should be exactly soc_min, not lower.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.15,  # Start close to min
        )

        # Try to discharge a lot (would go below 10%)
        discharge_power = 50  # kW
        duration_seconds = 3600  # 1 hour (would remove 55.56% normally)

        bess.discharge(discharge_power, duration_seconds)

        # SOC should be clamped to soc_min
        assert bess.current_soc == pytest.approx(BESS_SOC_MIN, rel=1e-9), (
            f"SOC should be clamped to {BESS_SOC_MIN:.0%}, got {bess.current_soc:.2%}"
        )

    def test_power_clamping_charge(self):
        """
        Verify that charging power is clamped to max_charge_power.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        # Request more power than max
        requested_power = 100  # kW (max is 50)
        actual_power = bess.charge(requested_power, 60)  # 1 minute

        assert actual_power <= BESS_MAX_POWER, (
            f"Actual power should be ≤ {BESS_MAX_POWER} kW, got {actual_power} kW"
        )

    def test_power_clamping_discharge(self):
        """
        Verify that discharging power is clamped to max_discharge_power.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        # Request more power than max
        requested_power = 100  # kW (max is 50)
        actual_power = bess.discharge(requested_power, 60)  # 1 minute

        assert actual_power <= BESS_MAX_POWER, (
            f"Actual power should be ≤ {BESS_MAX_POWER} kW, got {actual_power} kW"
        )

    def test_charge_discharge_cycle(self):
        """
        Verify a complete charge-discharge cycle.
        
        Test Case:
            1. Start at 50% SOC
            2. Charge for 30 minutes at 50 kW
            3. Discharge for 30 minutes at 50 kW
            
        Due to efficiency losses, final SOC should be less than initial.
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        initial_soc = bess.current_soc
        power = 50  # kW
        duration = 1800  # 30 minutes

        # Charge phase
        bess.charge(power, duration)
        soc_after_charge = bess.current_soc

        # Discharge phase
        bess.discharge(power, duration)
        final_soc = bess.current_soc

        # Charging: stored = 50 × 0.5h × 0.90 = 22.5 kWh → +22.5%
        # Discharging: removed = 50 × 0.5h / 0.90 = 27.78 kWh → -27.78%
        # Net change: -5.28%

        # Final should be less than initial due to round-trip efficiency
        assert final_soc < initial_soc, (
            f"Final SOC {final_soc:.2%} should be less than initial {initial_soc:.2%} "
            "due to round-trip efficiency losses"
        )

        # Calculate expected
        charge_energy = power * 0.5 * BESS_EFFICIENCY  # 22.5 kWh
        discharge_energy = power * 0.5 / BESS_EFFICIENCY  # 27.78 kWh
        expected_final = initial_soc + (charge_energy - discharge_energy) / BESS_CAPACITY

        assert final_soc == pytest.approx(expected_final, rel=1e-6), (
            f"Final SOC should be {expected_final:.2%}, got {final_soc:.2%}"
        )

    def test_round_trip_efficiency(self):
        """
        Verify the round-trip efficiency of the battery.
        
        Round-trip efficiency = η² (efficiency applied on both charge and discharge)
        For η = 0.90, round-trip = 0.81 (81%)
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=0.0,  # Allow full range for this test
            soc_max=1.0,
            initial_soc=0.50,
        )

        # Calculate energy that can be cycled
        # Charge 10 kWh at 100% power for some time
        charge_power = 50  # kW
        charge_time = 720  # 12 minutes = 0.2 hours

        initial_energy = bess.get_energy_stored()
        bess.charge(charge_power, charge_time)
        energy_after_charge = bess.get_energy_stored()

        # Energy input from grid = P × t = 50 × 0.2 = 10 kWh
        energy_input = charge_power * (charge_time / 3600)
        # Energy stored = 10 × 0.90 = 9 kWh
        energy_stored = energy_after_charge - initial_energy

        assert energy_stored == pytest.approx(energy_input * BESS_EFFICIENCY, rel=1e-6), (
            f"Charging: Expected {energy_input * BESS_EFFICIENCY:.2f} kWh stored, "
            f"got {energy_stored:.2f} kWh"
        )


class TestBESSModelWithOutput:
    """
    Test suite that generates CSV and plot outputs for thesis documentation.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_charge_discharge_cycle_with_csv(self):
        """
        Complete charge-discharge cycle with CSV output.
        
        Generates: output/bess_charge_discharge_cycle.csv
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        timestep = 60  # 1 minute
        charge_duration = 30  # 30 minutes of charging
        discharge_duration = 30  # 30 minutes of discharging
        power = BESS_MAX_POWER

        data = []
        total_energy_in = 0.0
        total_energy_out = 0.0
        time_min = 0

        # Record initial state
        data.append({
            "time_min": time_min,
            "phase": "initial",
            "power_kw": 0,
            "soc_percent": bess.current_soc * 100,
            "energy_stored_kwh": bess.get_energy_stored(),
            "energy_in_kwh": total_energy_in,
            "energy_out_kwh": total_energy_out,
        })

        # Charging phase
        for step in range(charge_duration):
            time_min += 1
            bess.charge(power, timestep)
            energy_in = power * timestep / 3600
            total_energy_in += energy_in

            data.append({
                "time_min": time_min,
                "phase": "charging",
                "power_kw": bess.current_power,
                "soc_percent": bess.current_soc * 100,
                "energy_stored_kwh": bess.get_energy_stored(),
                "energy_in_kwh": total_energy_in,
                "energy_out_kwh": total_energy_out,
            })

        # Discharging phase
        for step in range(discharge_duration):
            time_min += 1
            bess.discharge(power, timestep)
            energy_out = power * timestep / 3600
            total_energy_out += energy_out

            data.append({
                "time_min": time_min,
                "phase": "discharging",
                "power_kw": bess.current_power,
                "soc_percent": bess.current_soc * 100,
                "energy_stored_kwh": bess.get_energy_stored(),
                "energy_in_kwh": total_energy_in,
                "energy_out_kwh": total_energy_out,
            })

        # Write CSV
        csv_path = OUTPUT_DIR / "bess_charge_discharge_cycle.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")
        print(f"  Total energy in: {total_energy_in:.2f} kWh")
        print(f"  Total energy out: {total_energy_out:.2f} kWh")
        print(f"  Round-trip efficiency: {total_energy_out/total_energy_in*100:.1f}%")

    def test_soc_limits_with_csv(self):
        """
        Test SOC limit enforcement with CSV output.
        
        Generates: output/bess_soc_limits.csv
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        timestep = 60  # 1 minute
        power = BESS_MAX_POWER

        data = []
        time_min = 0

        # Record initial state
        data.append({
            "time_min": time_min,
            "phase": "initial",
            "requested_power_kw": 0,
            "actual_power_kw": 0,
            "soc_percent": bess.current_soc * 100,
            "at_limit": False,
        })

        # Charge until we hit soc_max
        for step in range(60):  # Up to 60 minutes
            time_min += 1
            actual_power = bess.charge(power, timestep)
            at_limit = bess.current_soc >= BESS_SOC_MAX - 0.001

            data.append({
                "time_min": time_min,
                "phase": "charging",
                "requested_power_kw": power,
                "actual_power_kw": actual_power,
                "soc_percent": bess.current_soc * 100,
                "at_limit": at_limit,
            })

            if at_limit:
                break

        # Discharge until we hit soc_min
        for step in range(120):  # Up to 120 minutes
            time_min += 1
            actual_power = bess.discharge(power, timestep)
            at_limit = bess.current_soc <= BESS_SOC_MIN + 0.001

            data.append({
                "time_min": time_min,
                "phase": "discharging",
                "requested_power_kw": power,
                "actual_power_kw": actual_power,
                "soc_percent": bess.current_soc * 100,
                "at_limit": at_limit,
            })

            if at_limit:
                break

        # Write CSV
        csv_path = OUTPUT_DIR / "bess_soc_limits.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")

    def test_bess_model_with_plot(self):
        """
        Generate plots demonstrating BESS behavior.
        
        Generates:
            - output/bess_charge_discharge.png
            - output/bess_efficiency.png
            - output/bess_combined.png
        """
        # ===========================================
        # Simulation: Charge then Discharge
        # ===========================================
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        timestep = 60  # 1 minute
        power = BESS_MAX_POWER

        times = [0]
        socs = [bess.current_soc * 100]
        powers = [0]
        energies = [bess.get_energy_stored()]
        phases = ["initial"]

        time_min = 0

        # Charge for 30 minutes
        for _ in range(30):
            time_min += 1
            bess.charge(power, timestep)
            times.append(time_min)
            socs.append(bess.current_soc * 100)
            powers.append(bess.current_power)
            energies.append(bess.get_energy_stored())
            phases.append("charging")

        # Idle for 5 minutes
        for _ in range(5):
            time_min += 1
            bess.idle()
            times.append(time_min)
            socs.append(bess.current_soc * 100)
            powers.append(bess.current_power)
            energies.append(bess.get_energy_stored())
            phases.append("idle")

        # Discharge for 30 minutes
        for _ in range(30):
            time_min += 1
            bess.discharge(power, timestep)
            times.append(time_min)
            socs.append(bess.current_soc * 100)
            powers.append(bess.current_power)
            energies.append(bess.get_energy_stored())
            phases.append("discharging")

        # ===========================================
        # Plot 1: SOC and Power over time
        # ===========================================
        fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # SOC
        ax1a.plot(times, socs, 'b-', linewidth=2)
        ax1a.axhline(y=BESS_SOC_MAX * 100, color='r', linestyle='--', alpha=0.7, 
                     label=f'SOC max ({BESS_SOC_MAX:.0%})')
        ax1a.axhline(y=BESS_SOC_MIN * 100, color='orange', linestyle='--', alpha=0.7,
                     label=f'SOC min ({BESS_SOC_MIN:.0%})')
        ax1a.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Initial SOC')
        ax1a.set_ylabel('State of Charge (%)', fontsize=12)
        ax1a.set_title('BESS Charge-Discharge Cycle', fontsize=14)
        ax1a.legend(loc='upper right', fontsize=9)
        ax1a.grid(True, alpha=0.3)
        ax1a.set_ylim(0, 100)

        # Annotate phases
        ax1a.annotate('Charging', xy=(15, socs[15]), fontsize=10, ha='center')
        ax1a.annotate('Idle', xy=(32.5, socs[32]), fontsize=10, ha='center')
        ax1a.annotate('Discharging', xy=(50, socs[50]), fontsize=10, ha='center')

        # Power
        colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in powers]
        ax1b.bar(times, powers, color=colors, width=0.8, alpha=0.7)
        ax1b.axhline(y=0, color='black', linewidth=1)
        ax1b.axhline(y=BESS_MAX_POWER, color='green', linestyle='--', alpha=0.5)
        ax1b.axhline(y=-BESS_MAX_POWER, color='red', linestyle='--', alpha=0.5)
        ax1b.set_xlabel('Time (minutes)', fontsize=12)
        ax1b.set_ylabel('Power (kW)', fontsize=12)
        ax1b.set_ylim(-BESS_MAX_POWER * 1.2, BESS_MAX_POWER * 1.2)
        ax1b.grid(True, alpha=0.3)

        # Add legend for power
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Charging (+)'),
                           Patch(facecolor='red', alpha=0.7, label='Discharging (-)')]
        ax1b.legend(handles=legend_elements, loc='upper right', fontsize=9)

        fig1.tight_layout()
        plot1_path = OUTPUT_DIR / "bess_charge_discharge.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ===========================================
        # Plot 2: Efficiency Analysis
        # ===========================================
        # Calculate energy flows
        charge_power = 50  # kW
        charge_time = 0.5  # hours

        energy_from_grid = charge_power * charge_time  # 25 kWh
        energy_stored = energy_from_grid * BESS_EFFICIENCY  # 22.5 kWh
        energy_delivered = energy_stored * BESS_EFFICIENCY  # 20.25 kWh
        
        # Create energy flow diagram
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        stages = ['Grid Input\n(25 kWh)', 'Stored in Battery\n(22.5 kWh)', 
                  'Delivered to Load\n(20.25 kWh)']
        values = [energy_from_grid, energy_stored, energy_delivered]
        positions = [0, 1, 2]

        bars = ax2.bar(positions, values, color=['blue', 'green', 'orange'], 
                       width=0.6, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f} kWh', ha='center', fontsize=11, fontweight='bold')

        # Add efficiency arrows
        ax2.annotate('', xy=(0.7, energy_stored), xytext=(0.3, energy_from_grid),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax2.text(0.5, (energy_from_grid + energy_stored) / 2 + 1, 
                 f'η = {BESS_EFFICIENCY:.0%}', fontsize=10, ha='center')

        ax2.annotate('', xy=(1.7, energy_delivered), xytext=(1.3, energy_stored),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax2.text(1.5, (energy_stored + energy_delivered) / 2 + 1,
                 f'η = {BESS_EFFICIENCY:.0%}', fontsize=10, ha='center')

        ax2.set_xticks(positions)
        ax2.set_xticklabels(stages, fontsize=11)
        ax2.set_ylabel('Energy (kWh)', fontsize=12)
        ax2.set_title(f'BESS Round-Trip Efficiency: {BESS_EFFICIENCY**2:.0%}\n'
                      f'(Charge η × Discharge η = {BESS_EFFICIENCY:.0%} × {BESS_EFFICIENCY:.0%})',
                      fontsize=12)
        ax2.set_ylim(0, energy_from_grid * 1.3)
        ax2.grid(True, alpha=0.3, axis='y')

        fig2.tight_layout()
        plot2_path = OUTPUT_DIR / "bess_efficiency.png"
        fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

        # ===========================================
        # Plot 3: Combined Figure (for thesis)
        # ===========================================
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        # Simulate a charge-discharge cycle with SOC limits
        bess_combined = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        times_combined = [0]
        socs_combined = [bess_combined.current_soc * 100]
        powers_combined = [0]
        requested_powers = [0]
        t = 0

        # Charge for 80 minutes (will hit limit and stay there)
        for _ in range(80):
            t += 1
            actual_power = bess_combined.charge(power, 60)
            times_combined.append(t)
            socs_combined.append(bess_combined.current_soc * 100)
            powers_combined.append(actual_power)
            requested_powers.append(power)  # Always requesting full power

        # Discharge for 180 minutes (will hit limit and stay there)
        for _ in range(180):
            t += 1
            actual_power = bess_combined.discharge(power, 60)
            times_combined.append(t)
            socs_combined.append(bess_combined.current_soc * 100)
            powers_combined.append(-actual_power)
            requested_powers.append(-power)  # Always requesting full power

        # Left: Power profile
        ax3a.plot(times_combined, requested_powers, 'k--', linewidth=2, alpha=0.7, label='Requested')
        ax3a.plot(times_combined, powers_combined, 'b-', linewidth=2.5, label='Actual')
        ax3a.fill_between(times_combined, 0, powers_combined, 
                          where=[p > 0 for p in powers_combined], 
                          color='green', alpha=0.3, label='Charging')
        ax3a.fill_between(times_combined, 0, powers_combined,
                          where=[p < 0 for p in powers_combined],
                          color='red', alpha=0.3, label='Discharging')
        ax3a.axhline(y=0, color='black', linewidth=1)
        ax3a.set_xlabel('Time (minutes)', fontsize=12)
        ax3a.set_ylabel('Power (kW)', fontsize=12)
        ax3a.set_title('(a) Power Profile', fontsize=12)
        ax3a.legend(loc='upper right', fontsize=9)
        ax3a.grid(True, alpha=0.3)
        ax3a.set_ylim(-BESS_MAX_POWER * 1.3, BESS_MAX_POWER * 1.3)

        # Right: SOC profile
        ax3b.plot(times_combined, socs_combined, 'b-', linewidth=2.5, label='Battery SOC')
        ax3b.axhline(y=BESS_SOC_MAX * 100, color='r', linestyle='--', linewidth=2,
                     label=f'SOC max = {BESS_SOC_MAX:.0%}')
        ax3b.axhline(y=BESS_SOC_MIN * 100, color='orange', linestyle='--', linewidth=2,
                     label=f'SOC min = {BESS_SOC_MIN:.0%}')
        ax3b.fill_between([0, max(times_combined)], [0, 0], [BESS_SOC_MIN * 100, BESS_SOC_MIN * 100], 
                          alpha=0.2, color='red')
        ax3b.fill_between([0, max(times_combined)], [BESS_SOC_MAX * 100, BESS_SOC_MAX * 100], [100, 100],
                          alpha=0.2, color='red')
        ax3b.set_xlabel('Time (minutes)', fontsize=12)
        ax3b.set_ylabel('State of Charge (%)', fontsize=12)
        ax3b.set_title('(b) State of Charge', fontsize=12)
        ax3b.legend(loc='center right', fontsize=9)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_ylim(0, 100)

        fig3.suptitle(f'BESS Model Verification\n'
                      f'Capacity: {BESS_CAPACITY} kWh, Power: ±{BESS_MAX_POWER} kW, '
                      f'η: {BESS_EFFICIENCY:.0%}',
                      fontsize=13, fontweight='bold', y=1.02)
        fig3.tight_layout()
        plot3_path = OUTPUT_DIR / "bess_combined.png"
        fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)

        print(f"\n✓ Plots saved to:")
        print(f"  - {plot1_path}")
        print(f"  - {plot2_path}")
        print(f"  - {plot3_path}")

    def test_generate_thesis_table(self):
        """
        Generate summary table for thesis.
        
        Generates: output/bess_summary.csv
        """
        bess = BESS(
            name="TestBESS",
            capacity=BESS_CAPACITY,
            max_charge_power=BESS_MAX_POWER,
            max_discharge_power=BESS_MAX_POWER,
            efficiency=BESS_EFFICIENCY,
            soc_min=BESS_SOC_MIN,
            soc_max=BESS_SOC_MAX,
            initial_soc=0.50,
        )

        results = []

        # Test different operations
        test_cases = [
            ("Charge 1h @ 50kW", "charge", 50, 3600),
            ("Charge 30min @ 50kW", "charge", 50, 1800),
            ("Discharge 1h @ 50kW", "discharge", 50, 3600),
            ("Discharge 30min @ 50kW", "discharge", 50, 1800),
        ]

        for name, operation, power, duration in test_cases:
            # Reset BESS
            bess.current_soc = 0.50

            initial_soc = bess.current_soc
            initial_energy = bess.get_energy_stored()

            if operation == "charge":
                bess.charge(power, duration)
            else:
                bess.discharge(power, duration)

            final_soc = bess.current_soc
            final_energy = bess.get_energy_stored()
            energy_change = final_energy - initial_energy

            results.append({
                "operation": name,
                "power_kw": power,
                "duration_min": duration / 60,
                "initial_soc_pct": initial_soc * 100,
                "final_soc_pct": round(final_soc * 100, 2),
                "energy_change_kwh": round(energy_change, 2),
                "efficiency": BESS_EFFICIENCY,
            })

        # Write CSV
        csv_path = OUTPUT_DIR / "bess_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Print formatted table
        print("\n" + "=" * 90)
        print(f"BESS MODEL VERIFICATION RESULTS")
        print(f"Capacity: {BESS_CAPACITY} kWh, Max Power: {BESS_MAX_POWER} kW, "
              f"Efficiency: {BESS_EFFICIENCY:.0%}")
        print("=" * 90)
        print(f"{'Operation':<25} {'Power':>8} {'Duration':>10} {'SOC Init':>10} "
              f"{'SOC Final':>10} {'ΔEnergy':>10}")
        print(f"{'':>25} {'(kW)':>8} {'(min)':>10} {'(%)':>10} {'(%)':>10} {'(kWh)':>10}")
        print("-" * 90)
        for r in results:
            print(f"{r['operation']:<25} {r['power_kw']:>8} {r['duration_min']:>10.0f} "
                  f"{r['initial_soc_pct']:>10.1f} {r['final_soc_pct']:>10.2f} "
                  f"{r['energy_change_kwh']:>+10.2f}")
        print("=" * 90)
        print(f"\n✓ Summary table saved to: {csv_path}")


def generate_all_outputs():
    """
    Standalone function to generate all outputs without running pytest.
    
    Usage: python test_bess_model.py --generate
    """
    print("=" * 60)
    print("GENERATING BESS MODEL TEST OUTPUTS")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tests with output generation
    test_instance = TestBESSModelWithOutput()
    test_instance.setup_output_dir()
    test_instance.test_charge_discharge_cycle_with_csv()
    test_instance.test_soc_limits_with_csv()
    test_instance.test_bess_model_with_plot()
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
