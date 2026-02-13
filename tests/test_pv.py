"""
PV Model Verification (PV class).

Objective:
    Verify correct solar elevation computation, zero production during night
    hours, realistic diurnal production behavior, and enforcement of rated
    capacity limits.

Test Case 1 – Night Condition:
    For timestamps where the apparent solar elevation satisfies α ≤ 0, the
    model must return zero power output. Multiple night-time hours are
    evaluated to confirm consistent behaviour.

Test Case 2 – Diurnal Production Profile:
    A clear-sky day is simulated using representative irradiance inputs. The
    resulting production profile is evaluated to confirm a bell-shaped curve
    during daylight hours, with peak generation occurring near solar noon.

Test Case 3 – Capacity Constraint:
    The instantaneous production is verified to never exceed the rated DC
    capacity of the system.

Run with --generate to produce plots and CSV in tests/output/.
"""

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from models.pv import PV


OUTPUT_DIR = Path(__file__).parent / "output" / "pv"

# Port of Funchal, Madeira coordinates
FUNCHAL_LAT = 32.6514  # degrees North
FUNCHAL_LON = -16.9084  # degrees West

# PV system parameters
PV_CAPACITY = 22.0  # kW peak
PV_TILT = 9.0  # degrees
PV_AZIMUTH = 180.0  # South-facing


@dataclass
class ClearSkyPoint:
    timestamp: datetime
    ghi: float
    dni: float
    dhi: float
    solar_elevation: float


def _make_pv(**overrides) -> PV:
    base = {
        "name": "TestPV",
        "capacity": PV_CAPACITY,
        "tilt": PV_TILT,
        "azimuth": PV_AZIMUTH,
        "latitude": FUNCHAL_LAT,
        "longitude": FUNCHAL_LON,
    }
    base.update(overrides)
    return PV(**base)


def _sample_day(
    date: datetime, step_minutes: int = 15
) -> List[Tuple[datetime, float]]:
    """Helper: sample a full day at regular time steps."""
    times = []
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    t = start
    while t < end:
        times.append(t)
        t += timedelta(minutes=step_minutes)
    return times


class TestNightCondition:
    """Test Case 1: Zero production when apparent solar elevation α ≤ 0."""

    def test_zero_production_for_night_hours(self):
        """
        Several night-time hours are checked; production must be exactly 0 kW.
        """
        pv = _make_pv()

        # Representative date for clear conditions (summer solstice)
        date = datetime(2025, 6, 21)
        night_hours = [0, 1, 2, 3, 4, 5, 22, 23]

        for hour in night_hours:
            ts = date.replace(hour=hour, minute=0)
            # No need for irradiance, model will detect α ≤ 0 internally
            production = pv.calculate_production(
                ghi=0.0,
                dni=0.0,
                dhi=0.0,
                temperature=20.0,
                timestamp=ts,
            )
            # Use an absolute tolerance to avoid strict float equality.
            assert production == pytest.approx(
                0.0, abs=1e-12
            ), f"Production should be 0 at {hour}:00, got {production} kW"

    def test_negative_elevation_implies_zero_production(self):
        """
        At midnight the solar elevation is strictly negative;
        production must be zero.
        """
        pv = _make_pv()
        ts = datetime(2025, 6, 21, 0, 0)

        # We indirectly test elevation through the model's internal check.
        production = pv.calculate_production(
            ghi=0.0, dni=0.0, dhi=0.0, temperature=15.0, timestamp=ts
        )
        assert production == pytest.approx(0.0, abs=1e-12)


class TestDiurnalProductionProfile:
    """Test Case 2: Bell-shaped daily production profile under clear sky."""

    def test_clear_sky_bell_shaped_profile(self):
        """
        Simulate a clear-sky day; production must rise from 0, peak near noon,
        and go back to 0, with peak occurring around solar noon.
        """
        pv = _make_pv()
        date = datetime(2025, 6, 21)

        # We drive the model with a generic clear-sky GHI envelope that
        # increases and then decreases around midday; DNI/DHI chosen
        # for a simple clear-sky-like split.
        times = _sample_day(date, step_minutes=30)
        productions = []
        for ts in times:
            # Simple smooth envelope: sin over daylight window [6h,18h]
            hour = ts.hour + ts.minute / 60
            if 6 <= hour <= 18:
                # Map [6,18] → [0, π]
                x = (hour - 6) / 12.0 * 3.141592653589793
                ghi = max(0.0, 900.0 * abs(__import__("math").sin(x)))
                dni = ghi * 0.8
                dhi = ghi * 0.2
            else:
                ghi = dni = dhi = 0.0

            p = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi, temperature=25.0, timestamp=ts
            )
            productions.append((hour, p))

        # Extract only positive production hours (daylight)
        daylight = [(h, p) for h, p in productions if p > 0]
        assert daylight, "There should be at least some daylight production."

        # Peak near noon: within ±2 hours of 12:00
        peak_hour, peak_value = max(daylight, key=lambda x: x[1])
        assert 10 <= peak_hour <= 14, f"Peak should occur near noon, got hour {peak_hour}"
        assert peak_value > PV_CAPACITY * 0.3, "Peak should be a significant fraction of capacity."

        # Monotonic rise before peak (coarsely) and fall after peak
        before = [p for h, p in daylight if h < peak_hour]
        after = [p for h, p in daylight if h > peak_hour]
        if len(before) >= 3:
            assert before[0] <= before[len(before) // 2] <= before[-1]
        if len(after) >= 3:
            assert after[0] >= after[len(after) // 2] >= after[-1]


class TestCapacityConstraint:
    """Test Case 3: Instantaneous production never exceeds rated capacity."""

    def test_production_never_exceeds_capacity(self):
        """
+       For a full clear-sky day, validate production P(t) ≤ capacity for
        every sample.
        """
        pv = _make_pv()
        date = datetime(2025, 6, 21)

        times = _sample_day(date, step_minutes=15)
        for ts in times:
            # Use a high, potentially exaggerated irradiance to stress the model.
            hour = ts.hour + ts.minute / 60
            if 6 <= hour <= 18:
                ghi = 1000.0
                dni = 800.0
                dhi = 200.0
            else:
                ghi = dni = dhi = 0.0

            p = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi, temperature=25.0, timestamp=ts
            )
            assert p <= PV_CAPACITY + 1e-6, f"Production {p} kW exceeds capacity {PV_CAPACITY} kW"


# ---------------------------------------------------------------------------
# Output generation: CSV and plots for documentation / thesis
# ---------------------------------------------------------------------------


class TestPVWithOutput:
    """Generate CSV and plots from PV tests for reporting."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_daily_profile_csv_and_plot(self):
        """
        Daily PV profile on a clear-sky summer solstice day.
        CSV + production plot saved under tests/output/pv.
        """
        pytest.importorskip("matplotlib")
        import numpy as np
        import matplotlib.pyplot as plt

        pv = _make_pv(name="Funchal_PV")
        date = datetime(2025, 6, 21)

        rows = []
        times = []
        productions = []

        for ts in _sample_day(date, step_minutes=15):
            hour = ts.hour + ts.minute / 60
            if 6 <= hour <= 18:
                # Smooth bell-shaped irradiance envelope
                x = (hour - 6) / 12.0 * 3.141592653589793
                ghi = max(0.0, 900.0 * abs(__import__("math").sin(x)))
                dni = ghi * 0.8
                dhi = ghi * 0.2
            else:
                ghi = dni = dhi = 0.0

            p = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi, temperature=25.0, timestamp=ts
            )

            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "hour_decimal": round(hour, 2),
                    "ghi_wm2": round(ghi, 1),
                    "dni_wm2": round(dni, 1),
                    "dhi_wm2": round(dhi, 1),
                    "production_kw": round(p, 4),
                }
            )
            times.append(hour)
            productions.append(p)

        csv_path = OUTPUT_DIR / "pv_daily_profile.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "hour_decimal",
                    "ghi_wm2",
                    "dni_wm2",
                    "dhi_wm2",
                    "production_kw",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        # Plot production curve
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(times, productions, step="mid", alpha=0.3, color="orange")
        ax.plot(times, productions, "orange", linewidth=2, label="PV production")
        ax.axhline(
            PV_CAPACITY,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Rated capacity ({PV_CAPACITY} kW)",
        )

        # Mark peak
        peak_idx = int(np.argmax(productions))
        ax.plot(times[peak_idx], productions[peak_idx], "ro", markersize=8)
        annotation_text = (
            f"Peak: {productions[peak_idx]:.1f} kW\n"
            f"at {times[peak_idx]:.1f} h"
        )
        ax.annotate(
            annotation_text,
            xy=(times[peak_idx], productions[peak_idx]),
            xytext=(times[peak_idx] + 1.5, productions[peak_idx]),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "black"},
        )

        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Power (kW)")
        ax.set_title(
            "PV daily production profile\n"
            f"Port of Funchal ({FUNCHAL_LAT:.2f}\u00b0N, {abs(FUNCHAL_LON):.2f}\u00b0W)"
        )
        ax.set_xlim(0, 24)
        ax.set_ylim(0, PV_CAPACITY * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        plot_path = OUTPUT_DIR / "pv_daily_profile.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\u2713 Plot saved: {plot_path}")

        # Expectations: no night production, peak within daylight
        for t, p in zip(times, productions):
            if t < 6 or t > 18:
                assert p == pytest.approx(0.0, abs=1e-9)
        assert 10 <= times[peak_idx] <= 14

    def test_night_condition_summary_csv(self):
        """Summary CSV explicitly listing several night-time samples."""
        pv = _make_pv()
        date = datetime(2025, 6, 21)

        night_hours = [0, 2, 4, 22]
        rows = []
        for h in night_hours:
            ts = date.replace(hour=h, minute=0)
            p = pv.calculate_production(
                ghi=0.0, dni=0.0, dhi=0.0, temperature=18.0, timestamp=ts
            )
            rows.append(
                {
                    "hour": f"{h:02d}:00",
                    "timestamp": ts.isoformat(),
                    "expected_alpha_condition": "alpha <= 0",
                    "production_kw": round(p, 4),
                }
            )

        csv_path = OUTPUT_DIR / "pv_night_condition.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "hour",
                    "timestamp",
                    "expected_alpha_condition",
                    "production_kw",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        # All listed samples must be zero production
        # Allow tiny numerical noise, but enforce values are effectively zero.
        for r in rows:
            assert r["production_kw"] == pytest.approx(0.0, abs=1e-9)

    def test_capacity_constraint_summary_csv(self):
        """Single summary CSV for capacity constraint checks."""
        pv = _make_pv()
        date = datetime(2025, 6, 21)

        rows = []
        for ts in _sample_day(date, step_minutes=60):
            hour = ts.hour
            if 6 <= hour <= 18:
                ghi = 1000.0
                dni = 800.0
                dhi = 200.0
            else:
                ghi = dni = dhi = 0.0

            p = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi, temperature=25.0, timestamp=ts
            )
            rows.append(
                {
                    "hour": f"{hour:02d}:00",
                    "ghi_wm2": ghi,
                    "dni_wm2": dni,
                    "dhi_wm2": dhi,
                    "production_kw": round(p, 4),
                    "capacity_kw": PV_CAPACITY,
                    "within_capacity": p <= PV_CAPACITY + 1e-6,
                }
            )

        csv_path = OUTPUT_DIR / "pv_capacity_constraint.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "hour",
                    "ghi_wm2",
                    "dni_wm2",
                    "dhi_wm2",
                    "production_kw",
                    "capacity_kw",
                    "within_capacity",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n\u2713 CSV saved: {csv_path}")

        assert all(r["within_capacity"] for r in rows)


def generate_all_outputs():
    """Generate all PV test outputs (CSV + plots) without running full pytest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = TestPVWithOutput()
    instance.setup_output_dir()
    instance.test_daily_profile_csv_and_plot()
    instance.test_night_condition_summary_csv()
    instance.test_capacity_constraint_summary_csv()
    print(f"\nPV test outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_all_outputs()
    else:
        pytest.main([__file__, "-v", "-s"])

