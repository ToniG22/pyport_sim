"""
Charger Model Verification (Charger class).

Objective:
    Verify that the charger enforces its rated power limits, correctly applies
    efficiency losses, and maintains consistent operational states.

Test Case 1 – Power Constraints:
    The charger is initialized with a maximum power rating (P_max). The
    constructor is tested to ensure that invalid configurations raise
    exceptions. During operation, requested power values exceeding P_max are
    externally clamped, and the assigned power is verified to remain within
    allowable limits.

Test Case 2 – Efficiency Handling:
    The effective charging power delivered to the battery is evaluated using
        P_effective = P * η_evse
    for different power levels. The returned value is compared against
    reference calculations to confirm correct loss modeling.

Test Case 3 – State Transitions:
    The state setter is tested to ensure that only valid enumerated states
    (IDLE, CHARGING) are accepted. Additionally, transitions to the IDLE
    state are verified to automatically reset the output power and disconnect
    any associated vessel.

Run with --generate to produce plots and CSV in tests/output/.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from models.charger import Charger, ChargerState

OUTPUT_DIR = Path(__file__).parent / "output" / "charger"


def _valid_charger_kwargs(**overrides):
    base = dict(
        name="TestCharger",
        max_power=22,  # kW
        efficiency=0.95,  # 95 %
        power=0.0,
    )
    base.update(overrides)
    return base


class TestPowerConstraints:
    """Test Case 1: Constructor validation and power limits."""

    def test_invalid_max_power_raises(self):
        """Non-positive max_power must raise a ValueError."""
        with pytest.raises(ValueError, match="Max power must be positive"):
            Charger(**_valid_charger_kwargs(max_power=0))
        with pytest.raises(ValueError, match="Max power must be positive"):
            Charger(**_valid_charger_kwargs(max_power=-5))

    def test_invalid_efficiency_raises(self):
        """Efficiency must lie strictly in (0, 1]."""
        with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
            Charger(**_valid_charger_kwargs(efficiency=0.0))
        with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
            Charger(**_valid_charger_kwargs(efficiency=-0.1))
        with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
            Charger(**_valid_charger_kwargs(efficiency=1.1))

    def test_invalid_initial_power_raises(self):
        """Initial power must be within [0, P_max]."""
        with pytest.raises(ValueError, match="Power cannot be negative"):
            Charger(**_valid_charger_kwargs(power=-1.0))
        with pytest.raises(ValueError, match="Power cannot exceed max_power"):
            Charger(**_valid_charger_kwargs(power=23.0))

    def test_requested_power_is_clamped_externally(self):
        """
        Simulate external clamping: requested power above P_max
        must not be assigned to the charger.
        """
        charger = Charger(**_valid_charger_kwargs())

        cases = [
            (0.0, 0.0),
            (10.0, 10.0),
            (22.0, 22.0),
            (30.0, 22.0),
            (50.0, 22.0),
        ]

        for requested, expected_assigned in cases:
            assigned = min(requested, charger.max_power)
            charger.power = assigned
            assert 0.0 <= charger.power <= charger.max_power
            assert charger.power == pytest.approx(
                expected_assigned, rel=1e-9
            ), f"Requested {requested} kW should result in {expected_assigned} kW assigned"


class TestEfficiencyHandling:
    """Test Case 2: P_effective = P * η_evse for different power levels."""

    def test_effective_power_at_max_rating(self):
        """
        For P = P_max = 22 kW and η = 0.95, P_effective must be 20.9 kW.
        """
        charger = Charger(**_valid_charger_kwargs())
        charger.power = charger.max_power

        expected = charger.max_power * charger.efficiency  # 22 * 0.95 = 20.9
        assert charger.effective_power == pytest.approx(
            expected, rel=1e-9
        ), f"Effective power should be {expected} kW, got {charger.effective_power} kW"

    def test_effective_power_for_multiple_power_levels(self):
        """
        Check P_effective = P * η_evse over a range of powers below and at P_max.
        """
        charger = Charger(**_valid_charger_kwargs())

        test_cases = [
            0.0,
            5.0,
            10.0,
            15.0,
            22.0,
        ]

        for power in test_cases:
            charger.power = power
            expected = power * charger.efficiency
            assert charger.effective_power == pytest.approx(
                expected, rel=1e-9
            ), f"At P={power} kW, effective power should be {expected} kW"


class TestStateTransitions:
    """Test Case 3: Valid states and automatic reset on IDLE."""

    def test_accepts_valid_states(self):
        """Only ChargerState.IDLE and ChargerState.CHARGING are accepted."""
        charger = Charger(**_valid_charger_kwargs())

        # Initial state from dataclass
        assert charger.state is ChargerState.IDLE

        charger.state = ChargerState.CHARGING
        assert charger.state is ChargerState.CHARGING

        charger.state = ChargerState.IDLE
        assert charger.state is ChargerState.IDLE

    def test_rejects_invalid_state_type(self):
        """Setting state to non-enum types must raise a ValueError."""
        charger = Charger(**_valid_charger_kwargs())

        with pytest.raises(ValueError, match="State must be a ChargerState enum"):
            charger.state = "idle"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="State must be a ChargerState enum"):
            charger.state = 1  # type: ignore[assignment]

    def test_idle_state_resets_power_and_connected_boat(self):
        """
        Transitioning to IDLE resets power to 0 kW and disconnects any boat.
        """
        charger = Charger(**_valid_charger_kwargs())

        charger.state = ChargerState.CHARGING
        charger.power = 10.0
        charger.connected_boat = "DemoBoat"

        assert charger.state is ChargerState.CHARGING
        assert charger.power == 10.0
        assert charger.connected_boat == "DemoBoat"

        charger.state = ChargerState.IDLE

        assert charger.state is ChargerState.IDLE
        assert charger.power == 0.0
        assert charger.connected_boat is None


# ---------------------------------------------------------------------------
# Output generation: CSV and plots for documentation / thesis
# ---------------------------------------------------------------------------


class TestChargerWithOutput:
    """Generate CSV and plots from charger tests for reporting."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_power_clamping_csv_and_plot(self):
        """
        Power clamping: requested vs assigned vs effective power.
        CSV + plot saved under tests/output/charger.
        """
        pytest.importorskip("matplotlib")
        import numpy as np
        import matplotlib.pyplot as plt

        charger = Charger(**_valid_charger_kwargs())

        requested_powers = [0.0, 5.0, 10.0, 15.0, 22.0, 30.0, 40.0, 50.0]
        rows = []
        for p_req in requested_powers:
            p_assigned = min(p_req, charger.max_power)
            charger.power = p_assigned
            p_eff = charger.effective_power
            rows.append(
                {
                    "requested_power_kw": p_req,
                    "assigned_power_kw": p_assigned,
                    "effective_power_kw": round(p_eff, 4),
                    "max_power_kw": charger.max_power,
                    "efficiency": charger.efficiency,
                    "was_clamped": p_req > charger.max_power,
                }
            )

        csv_path = OUTPUT_DIR / "charger_power_clamping.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "requested_power_kw",
                    "assigned_power_kw",
                    "effective_power_kw",
                    "max_power_kw",
                    "efficiency",
                    "was_clamped",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        # Smooth curves for requested, assigned (clamped) and effective power
        p_req_curve = np.linspace(0.0, 50.0, 200)
        p_assigned_curve = np.minimum(p_req_curve, charger.max_power)
        p_eff_curve = p_assigned_curve * charger.efficiency

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            p_req_curve,
            p_req_curve,
            "b--",
            linewidth=1.5,
            label="Requested power",
            alpha=0.7,
        )
        ax.plot(
            p_req_curve,
            p_assigned_curve,
            "g-",
            linewidth=2,
            label=f"Assigned power (clamped, $P_{{max}}$ = {charger.max_power} kW)",
        )
        ax.plot(
            p_req_curve,
            p_eff_curve,
            "r-",
            linewidth=2,
            label=r"Effective power $P_{\mathrm{eff}} = P \cdot \eta_{\mathrm{evse}}$",
        )

        # Highlight a representative over-limit request (e.g. 50 kW)
        p_test = 50.0
        p_assigned_test = min(p_test, charger.max_power)
        p_eff_test = p_assigned_test * charger.efficiency
        ax.plot(p_test, p_assigned_test, "ko", markersize=8, label="Example clamped point")
        ax.plot(p_test, p_eff_test, "ro", markersize=6)

        ax.axhline(charger.max_power, color="g", linestyle=":", alpha=0.5)
        ax.axvline(charger.max_power, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Requested power (kW)")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Charger power clamping and efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        fig.tight_layout()

        plot_path = OUTPUT_DIR / "charger_power_clamping.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\u2713 Plot saved: {plot_path}")

        # Sanity check on one of the tabulated rows
        ref = next(r for r in rows if r["requested_power_kw"] == 22.0)
        assert ref["assigned_power_kw"] == pytest.approx(22.0, rel=1e-9)
        assert ref["effective_power_kw"] == pytest.approx(22.0 * charger.efficiency, rel=1e-9)

    def test_efficiency_summary_csv(self):
        """Summary of P_effective = P * η_evse for several operating points."""
        charger = Charger(**_valid_charger_kwargs())

        powers = [5.0, 10.0, 15.0, 22.0]
        rows = []
        for p in powers:
            charger.power = p
            rows.append(
                {
                    "power_kw": p,
                    "efficiency": charger.efficiency,
                    "effective_power_kw": round(charger.effective_power, 4),
                }
            )

        csv_path = OUTPUT_DIR / "charger_efficiency_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["power_kw", "efficiency", "effective_power_kw"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

    def test_states_plot(self):
        """Plot valid charger states (IDLE, CHARGING)."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        states = [ChargerState.IDLE, ChargerState.CHARGING]
        labels = [s.value for s in states]
        x = range(len(states))
        colors = ["#2ecc71", "#3498db"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, [1] * len(states), color=colors, edgecolor="black", linewidth=1.2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Valid")
        ax.set_title(
            "Charger state enum (only these values accepted by state setter)"
        )
        ax.set_ylim(0, 1.2)
        fig.tight_layout()

        plot_path = OUTPUT_DIR / "charger_states.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\u2713 Plot saved: {plot_path}")

        # Simple behavioural check: switching to CHARGING is accepted.
        charger = Charger(**_valid_charger_kwargs())
        charger.state = ChargerState.CHARGING
        assert charger.state is ChargerState.CHARGING

    def test_charger_summary_csv(self):
        """Single summary CSV: P_max, efficiency, and effective power at P_max."""
        charger = Charger(**_valid_charger_kwargs())
        charger.power = charger.max_power

        summary = [
            {
                "parameter": "P_max",
                "value": charger.max_power,
                "unit": "kW",
            },
            {
                "parameter": "efficiency",
                "value": charger.efficiency,
                "unit": "-",
            },
            {
                "parameter": "P_effective at P_max",
                "value": round(charger.effective_power, 4),
                "unit": "kW",
            },
        ]

        csv_path = OUTPUT_DIR / "charger_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parameter", "value", "unit"])
            writer.writeheader()
            writer.writerows(summary)
        print(f"\n\u2713 CSV saved: {csv_path}")


def generate_all_outputs():
    """Generate all charger test outputs (CSV + plots) without running full pytest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = TestChargerWithOutput()
    instance.setup_output_dir()
    instance.test_power_clamping_csv_and_plot()
    instance.test_efficiency_summary_csv()
    instance.test_states_plot()
    instance.test_charger_summary_csv()
    print(f"\nCharger test outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_all_outputs()
    else:
        pytest.main([__file__, "-v", "-s"])

