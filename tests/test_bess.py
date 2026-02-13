"""
BESS Model Verification (BESS class).

Objective:
    Verify correct energy balance during charging and discharging, enforcement
    of maximum power limits, and compliance with SOC constraints.

Test Case 1 – Charging Dynamics:
    The battery is initialized at a defined SOC and subjected to a charging
    command at rated power for a known timestep. The stored energy is
        E_stored = P_charge * η * Δt
    and the resulting SOC variation is compared against the analytical
    expectation.

Test Case 2 – Discharging Dynamics:
    The battery is discharged at rated power for a fixed timestep. The energy
    removed from the battery is
        E_removed = (P_discharge / η) * Δt
    and the SOC reduction is verified accordingly.

Test Case 3 – SOC Limit Enforcement:
    Charging and discharging commands are applied near SOC_max and SOC_min.
    The model must prevent overcharge and overdischarge by adjusting the
    actual power and clamping SOC within its allowable range.

Test Case 4 – Power Constraints and Round-Trip Behavior:
    Requested power values exceeding rated limits are evaluated to confirm
    internal clamping. A complete charge–discharge cycle is performed to
    verify the expected round-trip efficiency behaviour.

Run with --generate to produce plots and CSV in tests/output/.
"""

import csv
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from models.bess import BESS


OUTPUT_DIR = Path(__file__).parent / "output" / "bess"

BESS_CAPACITY = 100.0  # kWh
BESS_MAX_POWER = 50.0  # kW (charge and discharge)
BESS_EFFICIENCY = 0.90  # round-trip efficiency factor
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90


def _make_bess(**overrides) -> BESS:
    base = {
        "name": "TestBESS",
        "capacity": BESS_CAPACITY,
        "max_charge_power": BESS_MAX_POWER,
        "max_discharge_power": BESS_MAX_POWER,
        "efficiency": BESS_EFFICIENCY,
        "soc_min": BESS_SOC_MIN,
        "soc_max": BESS_SOC_MAX,
        "initial_soc": 0.5,
    }
    base.update(overrides)
    return BESS(**base)


class TestChargingDynamics:
    """Test Case 1: Charging dynamics and energy balance."""

    def test_charging_energy_balance(self):
        """
        Charge at rated power for a known duration and compare SOC change to
        E_stored = P_charge * η * Δt.
        """
        bess = _make_bess(initial_soc=0.2)
        initial_soc = bess.current_soc

        power = BESS_MAX_POWER  # kW
        duration_seconds = 3600  # 1 hour

        actual_power = bess.charge(power, duration_seconds)
        assert actual_power <= BESS_MAX_POWER + 1e-9

        # Analytical expectation
        stored_energy = power * BESS_EFFICIENCY * (duration_seconds / 3600.0)
        expected_delta_soc = stored_energy / BESS_CAPACITY
        expected_soc = initial_soc + expected_delta_soc

        assert bess.current_soc == pytest.approx(expected_soc, rel=1e-6)


class TestDischargingDynamics:
    """Test Case 2: Discharging dynamics and energy balance."""

    def test_discharging_energy_balance(self):
        """
        Discharge at rated power for a known duration and compare SOC change to
        E_removed = (P_discharge / η) * Δt.
        """
        bess = _make_bess(initial_soc=0.8)
        initial_soc = bess.current_soc

        power = BESS_MAX_POWER  # kW
        duration_seconds = 3600  # 1 hour

        actual_power = bess.discharge(power, duration_seconds)
        assert actual_power <= BESS_MAX_POWER + 1e-9

        delivered_energy = power * (duration_seconds / 3600.0)
        removed_from_battery = delivered_energy / BESS_EFFICIENCY
        expected_delta_soc = removed_from_battery / BESS_CAPACITY
        expected_soc = initial_soc - expected_delta_soc

        assert bess.current_soc == pytest.approx(expected_soc, rel=1e-6)


class TestSocLimitEnforcement:
    """Test Case 3: SOC clamping at SOC_min and SOC_max."""

    def test_prevents_overcharge_at_soc_max(self):
        """Charging near SOC_max must clamp SOC and adjust actual power."""
        bess = _make_bess(initial_soc=BESS_SOC_MAX - 0.02)

        power = BESS_MAX_POWER
        duration_seconds = 3600  # enough to overshoot without clamp

        actual_power = bess.charge(power, duration_seconds)

        assert bess.current_soc == pytest.approx(BESS_SOC_MAX, rel=1e-9)
        assert 0.0 < actual_power <= BESS_MAX_POWER + 1e-9

    def test_prevents_overdischarge_at_soc_min(self):
        """Discharging near SOC_min must clamp SOC and adjust actual power."""
        bess = _make_bess(initial_soc=BESS_SOC_MIN + 0.02)

        power = BESS_MAX_POWER
        duration_seconds = 3600

        actual_power = bess.discharge(power, duration_seconds)

        assert bess.current_soc == pytest.approx(BESS_SOC_MIN, rel=1e-9)
        assert 0.0 < actual_power <= BESS_MAX_POWER + 1e-9


class TestPowerConstraintsAndRoundTrip:
    """Test Case 4: Power limits and round-trip efficiency behaviour."""

    def test_charge_and_discharge_power_clamping(self):
        """Requested power above ratings must be internally clamped."""
        bess = _make_bess(initial_soc=0.5)

        # Charge side
        requested_charge = BESS_MAX_POWER * 2
        actual_charge = bess.charge(requested_charge, 60)
        assert actual_charge <= BESS_MAX_POWER + 1e-9

        # Discharge side
        requested_discharge = BESS_MAX_POWER * 2
        actual_discharge = bess.discharge(requested_discharge, 60)
        assert actual_discharge <= BESS_MAX_POWER + 1e-9

    def test_round_trip_cycle(self):
        """
        Charge then discharge at rated power; final SOC must be below initial
        and match analytical round-trip expectation.
        """
        bess = _make_bess(initial_soc=0.5)

        power = BESS_MAX_POWER
        duration = 1800  # 0.5 h each phase
        initial_soc = bess.current_soc

        # Charge
        bess.charge(power, duration)
        soc_after_charge = bess.current_soc
        assert soc_after_charge > initial_soc

        # Discharge
        bess.discharge(power, duration)
        final_soc = bess.current_soc
        assert final_soc < initial_soc

        # Analytical expectation:
        e_charge = power * (duration / 3600.0) * BESS_EFFICIENCY
        e_discharge = power * (duration / 3600.0) / BESS_EFFICIENCY
        expected_final_soc = initial_soc + (e_charge - e_discharge) / BESS_CAPACITY

        assert final_soc == pytest.approx(expected_final_soc, rel=1e-6)


# ---------------------------------------------------------------------------
# Output generation: CSV and plots for documentation / thesis
# ---------------------------------------------------------------------------


class TestBESSWithOutput:
    """Generate CSV and plots from BESS tests for reporting."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_charge_discharge_profile_csv_and_plot(self):
        """
        Time-series of SOC and power for a charge–idle–discharge sequence.
        CSV + plot saved under tests/output/bess.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        bess = _make_bess(initial_soc=0.5)

        timestep_s = 60
        minutes_charge = 30
        minutes_idle = 5
        minutes_discharge = 30

        t_min = 0
        rows = []
        times = []
        socs = []
        powers = []

        def record(phase: str):
            rows.append(
                {
                    "time_min": t_min,
                    "phase": phase,
                    "soc_percent": round(bess.current_soc * 100.0, 3),
                    "power_kw": round(bess.current_power, 3),
                    "energy_stored_kwh": round(bess.get_energy_stored(), 3),
                }
            )
            times.append(t_min)
            socs.append(bess.current_soc * 100.0)
            powers.append(bess.current_power)

        record("initial")

        # Charging phase
        for _ in range(minutes_charge):
            t_min += 1
            bess.charge(BESS_MAX_POWER, timestep_s)
            record("charging")

        # Idle phase
        for _ in range(minutes_idle):
            t_min += 1
            bess.idle()
            record("idle")

        # Discharging phase
        for _ in range(minutes_discharge):
            t_min += 1
            bess.discharge(BESS_MAX_POWER, timestep_s)
            record("discharging")

        csv_path = OUTPUT_DIR / "bess_charge_discharge_profile.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time_min",
                    "phase",
                    "soc_percent",
                    "power_kw",
                    "energy_stored_kwh",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        fig, (ax_soc, ax_p) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax_soc.plot(times, socs, "b-", linewidth=2)
        ax_soc.axhline(BESS_SOC_MIN * 100, color="orange", linestyle="--", alpha=0.7)
        ax_soc.axhline(BESS_SOC_MAX * 100, color="red", linestyle="--", alpha=0.7)
        ax_soc.set_ylabel("SOC (%)")
        ax_soc.set_ylim(0, 100)
        ax_soc.grid(True, alpha=0.3)

        ax_p.bar(times, powers, color=["green" if p > 0 else "red" if p < 0 else "gray" for p in powers])
        ax_p.axhline(0.0, color="black", linewidth=1)
        ax_p.axhline(BESS_MAX_POWER, color="green", linestyle="--", alpha=0.5)
        ax_p.axhline(-BESS_MAX_POWER, color="red", linestyle="--", alpha=0.5)
        ax_p.set_xlabel("Time (minutes)")
        ax_p.set_ylabel("Power (kW)")
        ax_p.grid(True, alpha=0.3)

        fig.suptitle(
            f"BESS charge–idle–discharge sequence\n"
            f"Capacity {BESS_CAPACITY} kWh, ±{BESS_MAX_POWER} kW, η={BESS_EFFICIENCY:.0%}"
        )
        fig.tight_layout()

        plot_path = OUTPUT_DIR / "bess_charge_discharge_profile.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\u2713 Plot saved: {plot_path}")

        # Sanity checks: SOC remains within [SOC_min, SOC_max]
        assert min(socs) >= BESS_SOC_MIN * 100 - 1e-6
        assert max(socs) <= BESS_SOC_MAX * 100 + 1e-6

    def test_soc_limits_summary_csv(self):
        """CSV summarising operation near SOC_min and SOC_max constraints."""
        bess = _make_bess(initial_soc=0.5)

        rows = []

        # Near SOC_max: repeated charging
        for _ in range(10):
            before_soc = bess.current_soc
            actual_power = bess.charge(BESS_MAX_POWER, 600)  # 10 min
            after_soc = bess.current_soc
            rows.append(
                {
                    "phase": "charge_near_max",
                    "soc_before": round(before_soc, 4),
                    "soc_after": round(after_soc, 4),
                    "requested_power_kw": BESS_MAX_POWER,
                    "actual_power_kw": round(actual_power, 4),
                }
            )

        # Then near SOC_min: repeated discharging
        for _ in range(20):
            before_soc = bess.current_soc
            actual_power = bess.discharge(BESS_MAX_POWER, 600)
            after_soc = bess.current_soc
            rows.append(
                {
                    "phase": "discharge_near_min",
                    "soc_before": round(before_soc, 4),
                    "soc_after": round(after_soc, 4),
                    "requested_power_kw": BESS_MAX_POWER,
                    "actual_power_kw": round(actual_power, 4),
                }
            )

        csv_path = OUTPUT_DIR / "bess_soc_limits.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "phase",
                    "soc_before",
                    "soc_after",
                    "requested_power_kw",
                    "actual_power_kw",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        # All SOC values must remain within [SOC_min, SOC_max].
        for r in rows:
            assert BESS_SOC_MIN <= r["soc_after"] <= BESS_SOC_MAX + 1e-9

    def test_round_trip_summary_csv(self):
        """CSV summarising a simple charge–discharge round-trip."""
        bess = _make_bess(initial_soc=0.5)

        power = BESS_MAX_POWER
        duration = 1800  # 0.5 h

        initial_energy = bess.get_energy_stored()
        bess.charge(power, duration)
        energy_after_charge = bess.get_energy_stored()
        bess.discharge(power, duration)
        final_energy = bess.get_energy_stored()

        energy_in = power * (duration / 3600.0)
        stored = energy_after_charge - initial_energy
        delivered = initial_energy - final_energy + stored  # net to load in cycle

        rows = [
            {
                "capacity_kwh": BESS_CAPACITY,
                "max_power_kw": BESS_MAX_POWER,
                "efficiency": BESS_EFFICIENCY,
                "energy_input_kwh": round(energy_in, 4),
                "energy_stored_kwh": round(stored, 4),
                "energy_delivered_kwh": round(delivered, 4),
            }
        ]

        csv_path = OUTPUT_DIR / "bess_round_trip_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "capacity_kwh",
                    "max_power_kw",
                    "efficiency",
                    "energy_input_kwh",
                    "energy_stored_kwh",
                    "energy_delivered_kwh",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")


def generate_all_outputs():
    """Generate all BESS test outputs (CSV + plots) without running full pytest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = TestBESSWithOutput()
    instance.setup_output_dir()
    instance.test_charge_discharge_profile_csv_and_plot()
    instance.test_soc_limits_summary_csv()
    instance.test_round_trip_summary_csv()
    print(f"\nBESS test outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_all_outputs()
    else:
        pytest.main([__file__, "-v", "-s"])

