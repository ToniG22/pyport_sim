"""
Vessel Model Verification (Boat class).

Objective:
    Verify the correct computation of the hydrodynamic coefficient k and
    the enforcement of constructor validation rules and state management.

Test Case 1 – Propeller Law Coefficient:
    k = P_motor / v_range^3; with P_motor=500 kW, v_range=10 knots → k=0.5.
    Cubic relation P = k v^3 is checked for multiple speed values.

Test Case 2 – Physical Boundary Conditions:
    Non-physical inputs (non-positive motor power, invalid battery capacity,
    SOC not in [0,1]) must raise appropriate exceptions.

Test Case 3 – State Management:
    Only valid enumerated states (IDLE, CHARGING, SAILING) are accepted.

Run with --generate to produce plots and CSV in tests/output/.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.boat import Boat, BoatState

OUTPUT_DIR = Path(__file__).parent / "output" / "boat"


def _valid_boat_kwargs(**overrides):
    base = dict(
        name="TestBoat",
        motor_power=500,
        weight=5000.0,
        length=10.0,
        battery_capacity=1000.0,
        range_speed=10.0,
        soc=1.0,
    )
    base.update(overrides)
    return base


class TestPropellerLawCoefficient:
    """Test Case 1: Hydrodynamic coefficient k and cubic scaling P = k v^3."""

    def test_k_computation(self):
        """k = P_motor / v_range^3; with 500 kW and 10 knots, k must be 0.5."""
        boat = Boat(**_valid_boat_kwargs())
        expected_k = 0.5
        assert boat.k == pytest.approx(expected_k, rel=1e-9), (
            f"k should be {expected_k}, got {boat.k}"
        )

    def test_cubic_scaling_at_multiple_speeds(self):
        """P = k v^3 for several speeds; k=0.5 from P_motor=500 kW, v_range=10 kn."""
        boat = Boat(**_valid_boat_kwargs())
        k = boat.k
        assert k == pytest.approx(0.5, rel=1e-9)

        cases = [
            (5.0, 62.5),
            (10.0, 500.0),
            (15.0, 1687.5),
        ]
        for speed, expected_power in cases:
            P = k * (speed ** 3)
            assert P == pytest.approx(expected_power, rel=1e-9), (
                f"At v={speed} kn, P should be {expected_power} kW, got {P}"
            )


class TestPhysicalBoundaryConditions:
    """Test Case 2: Constructor raises on non-physical inputs."""

    def test_non_positive_motor_power_raises(self):
        with pytest.raises(ValueError, match="Motor power must be positive"):
            Boat(**_valid_boat_kwargs(motor_power=0))
        with pytest.raises(ValueError, match="Motor power must be positive"):
            Boat(**_valid_boat_kwargs(motor_power=-100))

    def test_invalid_battery_capacity_raises(self):
        with pytest.raises(ValueError, match="Battery capacity must be positive"):
            Boat(**_valid_boat_kwargs(battery_capacity=0))
        with pytest.raises(ValueError, match="Battery capacity must be positive"):
            Boat(**_valid_boat_kwargs(battery_capacity=-50))

    def test_soc_out_of_range_raises(self):
        with pytest.raises(ValueError, match="SOC must be between 0 and 1"):
            Boat(**_valid_boat_kwargs(soc=-0.1))
        with pytest.raises(ValueError, match="SOC must be between 0 and 1"):
            Boat(**_valid_boat_kwargs(soc=1.1))


class TestStateManagement:
    """Test Case 3: Only valid BoatState values (IDLE, CHARGING, SAILING) accepted."""

    def test_accepts_valid_states(self):
        boat = Boat(**_valid_boat_kwargs())
        boat.state = BoatState.IDLE
        assert boat.state is BoatState.IDLE
        boat.state = BoatState.CHARGING
        assert boat.state is BoatState.CHARGING
        boat.state = BoatState.SAILING
        assert boat.state is BoatState.SAILING

    def test_rejects_invalid_state_type(self):
        boat = Boat(**_valid_boat_kwargs())
        with pytest.raises(ValueError, match="State must be a BoatState enum"):
            boat.state = "idle"
        with pytest.raises(ValueError, match="State must be a BoatState enum"):
            boat.state = 42


# ---------------------------------------------------------------------------
# Output generation: CSV and plots for documentation / thesis
# ---------------------------------------------------------------------------


class TestBoatWithOutput:
    """Generate CSV and plots from boat tests for reporting."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_propeller_law_csv_and_plot(self):
        """Propeller law: k = P/v^3, P = k*v^3. CSV + power-vs-speed plot."""
        pytest.importorskip("matplotlib")
        import numpy as np
        import matplotlib.pyplot as plt

        boat = Boat(**_valid_boat_kwargs())
        k = boat.k
        speeds = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
        rows = []
        for v in speeds:
            P = k * (v ** 3)
            rows.append({"speed_knots": v, "power_kw": round(P, 2), "k": k})

        csv_path = OUTPUT_DIR / "boat_propeller_law.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["speed_knots", "power_kw", "k"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ CSV saved: {csv_path}")

        v_curve = np.linspace(0, 16, 200)
        P_curve = k * (v_curve ** 3)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(v_curve, P_curve, "b-", linewidth=2, label=r"$P = k \, v^3$")
        ax.plot(10, 500, "ro", markersize=10, label="Reference (10 kn, 500 kW)")
        ax.axhline(500, color="r", linestyle="--", alpha=0.5)
        ax.axvline(10, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Speed (knots)")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"Boat propeller law (k = {k}, $P_{{motor}}$=500 kW, $v_{{range}}$=10 kn)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, None)
        fig.tight_layout()
        plot_path = OUTPUT_DIR / "boat_propeller_law.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {plot_path}")

        assert k == pytest.approx(0.5, rel=1e-9)

    def test_validation_summary_csv(self):
        """Summary of physical boundary checks: what was tested and expected."""
        rows = [
            {"check": "motor_power > 0", "invalid_example": "0, -100", "expected": "ValueError"},
            {"check": "battery_capacity > 0", "invalid_example": "0, -50", "expected": "ValueError"},
            {"check": "0 <= SOC <= 1", "invalid_example": "-0.1, 1.1", "expected": "ValueError"},
        ]
        csv_path = OUTPUT_DIR / "boat_validation_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["check", "invalid_example", "expected"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ CSV saved: {csv_path}")

    def test_states_plot(self):
        """Plot valid boat states (IDLE, CHARGING, SAILING)."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        boat = Boat(**_valid_boat_kwargs())
        states = [BoatState.IDLE, BoatState.CHARGING, BoatState.SAILING]
        labels = [s.value for s in states]
        x = range(len(states))
        colors = ["#2ecc71", "#3498db", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(x, [1] * len(states), color=colors, edgecolor="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Valid")
        ax.set_title("Boat state enum (only these values accepted by state setter)")
        ax.set_ylim(0, 1.2)
        fig.tight_layout()
        plot_path = OUTPUT_DIR / "boat_states.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {plot_path}")

        boat.state = BoatState.SAILING
        assert boat.state is BoatState.SAILING

    def test_boat_summary_csv(self):
        """Single summary CSV: k, reference speeds/powers, test result."""
        boat = Boat(**_valid_boat_kwargs())
        k = boat.k
        ref_speeds = [5.0, 10.0, 15.0]
        summary = [
            {
                "parameter": "k (P_motor / v_range^3)",
                "value": k,
                "unit": "-",
            },
            {
                "parameter": "P at 5 kn",
                "value": round(k * 5**3, 2),
                "unit": "kW",
            },
            {
                "parameter": "P at 10 kn",
                "value": round(k * 10**3, 2),
                "unit": "kW",
            },
            {
                "parameter": "P at 15 kn",
                "value": round(k * 15**3, 2),
                "unit": "kW",
            },
        ]
        csv_path = OUTPUT_DIR / "boat_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parameter", "value", "unit"])
            writer.writeheader()
            writer.writerows(summary)
        print(f"\n✓ CSV saved: {csv_path}")


def generate_all_outputs():
    """Generate all boat test outputs (CSV + plots) without running full pytest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = TestBoatWithOutput()
    instance.setup_output_dir()
    instance.test_propeller_law_csv_and_plot()
    instance.test_validation_summary_csv()
    instance.test_states_plot()
    instance.test_boat_summary_csv()
    print(f"\nBoat test outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_all_outputs()
    else:
        pytest.main([__file__, "-v", "-s"])
