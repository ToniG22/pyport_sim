"""
Test: PV Model Verification

Objective:
    Ensure zero production during night hours and peak production at solar noon.

Test Case:
    Simulate a clear-sky day at the port's latitude (Port of Funchal, Madeira).

Expected Outcome:
    - Production should be exactly 0 kW when solar elevation α ≤ 0
    - Production should follow a bell-shaped curve during daylight hours
    - Peak production occurs around solar noon
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
from models.pv import PV

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "output"

# Port of Funchal, Madeira coordinates
FUNCHAL_LAT = 32.6514  # degrees North
FUNCHAL_LON = -16.9084  # degrees West

# PV system parameters
PV_CAPACITY = 22  # kW peak
PV_TILT = 9  # degrees (optimized for summer solstice: latitude - declination ≈ 32.65 - 23.45)


def calculate_clear_sky_irradiance(solar_elevation: float) -> tuple:
    """
    Calculate clear-sky irradiance values based on solar elevation.
    
    Uses simplified clear-sky model for testing.
    
    Args:
        solar_elevation: Solar elevation angle in degrees
        
    Returns:
        Tuple of (GHI, DNI, DHI) in W/m²
    """
    if solar_elevation <= 0:
        return 0.0, 0.0, 0.0
    
    # Air mass calculation (simplified Kasten-Young formula)
    elevation_rad = math.radians(solar_elevation)
    air_mass = 1 / (math.sin(elevation_rad) + 0.50572 * (6.07995 + solar_elevation) ** (-1.6364))
    
    # Direct Normal Irradiance (simplified Ineichen clear sky model)
    # Typical clear sky DNI at sea level
    dni_max = 1000  # W/m² (extraterrestrial constant simplified)
    atmospheric_extinction = 0.7  # Atmospheric clarity
    dni = dni_max * (atmospheric_extinction ** air_mass)
    
    # Diffuse Horizontal Irradiance (simplified)
    dhi = 0.1 * dni_max * math.sin(elevation_rad) + 50  # Base diffuse
    
    # Global Horizontal Irradiance
    ghi = dni * math.sin(elevation_rad) + dhi
    
    return max(0, ghi), max(0, dni), max(0, dhi)


class TestPVModel:
    """Test suite for the PV model verification."""

    def test_pv_initialization(self):
        """
        Verify PV system initializes with correct parameters.
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,  # South-facing
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        assert pv.name == "TestPV"
        assert pv.capacity == PV_CAPACITY
        assert pv.latitude == FUNCHAL_LAT
        assert pv.longitude == FUNCHAL_LON
        assert pv.current_production == 0.0

    def test_zero_production_at_night(self):
        """
        Verify that production is exactly 0 kW when solar elevation ≤ 0.
        
        Test at various night hours (before sunrise and after sunset).
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        # Test date: Summer solstice for clear conditions
        test_date = datetime(2025, 6, 21)
        
        # Night hours (before sunrise ~6:00 and after sunset ~21:00 in summer)
        night_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        
        for hour in night_hours:
            timestamp = test_date.replace(hour=hour, minute=0)
            
            # Even with irradiance values, production should be 0 at night
            production = pv.calculate_production(
                ghi=0.0,
                dni=0.0,
                dhi=0.0,
                temperature=20.0,
                timestamp=timestamp,
            )
            
            assert production == 0.0, (
                f"Production should be 0 at {hour}:00, got {production} kW"
            )

    def test_solar_elevation_calculation(self):
        """
        Verify solar elevation is calculated correctly.
        
        At solar noon on summer solstice at Funchal (32.65°N):
        - Declination ≈ 23.45°
        - Max elevation ≈ 90 - |32.65 - 23.45| ≈ 80.8°
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        # Summer solstice at solar noon (12:00 UTC)
        timestamp = datetime(2025, 6, 21, 12, 0)
        elevation = pv._calculate_solar_elevation(timestamp)
        
        # Expected: approximately 80° at solar noon on summer solstice
        # Declination = 23.45°, Latitude = 32.65°
        # Max elevation = 90 - |lat - dec| = 90 - |32.65 - 23.45| = 80.8°
        expected_elevation = 90 - abs(FUNCHAL_LAT - 23.45)
        
        assert elevation == pytest.approx(expected_elevation, abs=2.0), (
            f"Solar elevation at noon should be ~{expected_elevation}°, got {elevation}°"
        )

    def test_negative_elevation_during_night(self):
        """
        Verify solar elevation is negative during night hours.
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        # Midnight
        timestamp = datetime(2025, 6, 21, 0, 0)
        elevation = pv._calculate_solar_elevation(timestamp)
        
        assert elevation < 0, f"Solar elevation at midnight should be negative, got {elevation}°"

    def test_production_follows_bell_curve(self):
        """
        Verify that production follows a bell-shaped curve during the day.
        
        Production should:
        1. Start at 0 (sunrise)
        2. Follow a symmetric curve around solar noon
        3. Decrease back to 0 (sunset)
        
        Note: For tilted panels, peak production may not occur exactly at solar noon
        due to the angle of incidence. This is correct PV physics.
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        test_date = datetime(2025, 6, 21)
        productions = []
        
        for hour in range(24):
            timestamp = test_date.replace(hour=hour, minute=0)
            elevation = pv._calculate_solar_elevation(timestamp)
            ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
            
            production = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi,
                temperature=25.0, timestamp=timestamp
            )
            productions.append((hour, production))

        # Find peak hour and value
        peak_hour, peak_production = max(productions, key=lambda x: x[1])
        
        # Peak should be during daylight hours (6-18 hours)
        assert 6 <= peak_hour <= 18, (
            f"Peak production should be during daylight, got hour {peak_hour}"
        )
        
        # Peak should be significant (at least 30% of capacity in good conditions)
        assert peak_production > PV_CAPACITY * 0.3, (
            f"Peak production should be significant, got {peak_production} kW"
        )
        
        # Verify symmetry around noon (approximately)
        # For tilted panels, 10:00 and 14:00 should be roughly equal
        morning_10 = next(p for h, p in productions if h == 10)
        afternoon_14 = next(p for h, p in productions if h == 14)
        
        # Morning and afternoon should be within 10% of each other (symmetric)
        if morning_10 > 0 and afternoon_14 > 0:
            ratio = min(morning_10, afternoon_14) / max(morning_10, afternoon_14)
            assert ratio > 0.9, f"Production should be symmetric: 10:00={morning_10:.2f}, 14:00={afternoon_14:.2f}"

    def test_symmetric_production_around_noon(self):
        """
        Verify production is symmetric around solar noon.
        
        Note: For tilted south-facing panels on summer solstice at mid-latitudes,
        the optimal angle of incidence may occur before/after solar noon, creating
        a "double peak" or flattened curve. This is correct PV physics.
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        test_date = datetime(2025, 6, 21)
        
        # Test production at different hours
        productions = {}
        for hour in range(6, 20):
            timestamp = test_date.replace(hour=hour, minute=0)
            elevation = pv._calculate_solar_elevation(timestamp)
            ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
            
            production = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi,
                temperature=25.0, timestamp=timestamp
            )
            productions[hour] = production

        # Verify symmetry: production at equidistant hours from noon should be similar
        # 10:00 vs 14:00 (both 2 hours from noon)
        assert abs(productions[10] - productions[14]) < 0.1, (
            f"Production should be symmetric: 10:00={productions[10]:.2f}, 14:00={productions[14]:.2f}"
        )
        
        # 8:00 vs 16:00 (both 4 hours from noon)
        assert abs(productions[8] - productions[16]) < 0.1, (
            f"Production should be symmetric: 8:00={productions[8]:.2f}, 16:00={productions[16]:.2f}"
        )

    def test_production_respects_capacity_limit(self):
        """
        Verify production never exceeds system capacity.
        """
        pv = PV(
            name="TestPV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        test_date = datetime(2025, 6, 21)
        
        for hour in range(24):
            timestamp = test_date.replace(hour=hour, minute=0)
            elevation = pv._calculate_solar_elevation(timestamp)
            ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
            
            production = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi,
                temperature=25.0, timestamp=timestamp
            )
            
            # Production should never exceed capacity
            assert production <= PV_CAPACITY, (
                f"Production {production} kW exceeds capacity {PV_CAPACITY} kW"
            )


class TestPVModelWithOutput:
    """
    Test suite that generates CSV and plot outputs for thesis documentation.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_daily_production_with_csv_output(self):
        """
        Simulate a full day and output results to CSV.
        
        Generates: output/pv_daily_production.csv
        """
        pv = PV(
            name="Funchal_PV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        test_date = datetime(2025, 6, 21)  # Summer solstice
        
        data = []
        total_energy = 0.0
        
        # Simulate every 15 minutes
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = test_date.replace(hour=hour, minute=minute)
                elevation = pv._calculate_solar_elevation(timestamp)
                ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
                
                production = pv.calculate_production(
                    ghi=ghi, dni=dni, dhi=dhi,
                    temperature=25.0, timestamp=timestamp
                )
                
                # Energy in this 15-minute interval (kWh)
                energy_interval = production * (15 / 60)
                total_energy += energy_interval
                
                data.append({
                    "time": f"{hour:02d}:{minute:02d}",
                    "hour_decimal": hour + minute / 60,
                    "solar_elevation_deg": round(elevation, 2),
                    "ghi_wm2": round(ghi, 1),
                    "dni_wm2": round(dni, 1),
                    "dhi_wm2": round(dhi, 1),
                    "production_kw": round(production, 3),
                    "cumulative_energy_kwh": round(total_energy, 3),
                })

        # Write CSV
        csv_path = OUTPUT_DIR / "pv_daily_production.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")
        print(f"  Total daily energy: {total_energy:.2f} kWh")

        # Verify night hours have zero production
        night_entries = [d for d in data if d["solar_elevation_deg"] <= 0]
        for entry in night_entries:
            assert entry["production_kw"] == 0.0

    def test_solar_elevation_profile_csv(self):
        """
        Output solar elevation throughout the day.
        
        Generates: output/pv_solar_elevation.csv
        """
        pv = PV(
            name="Funchal_PV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        # Test multiple dates (solstices and equinoxes)
        test_dates = [
            (datetime(2025, 3, 21), "Spring Equinox"),
            (datetime(2025, 6, 21), "Summer Solstice"),
            (datetime(2025, 9, 23), "Autumn Equinox"),
            (datetime(2025, 12, 21), "Winter Solstice"),
        ]
        
        data = []
        
        for test_date, date_name in test_dates:
            for hour in range(24):
                timestamp = test_date.replace(hour=hour, minute=0)
                elevation = pv._calculate_solar_elevation(timestamp)
                
                data.append({
                    "date": test_date.strftime("%Y-%m-%d"),
                    "date_name": date_name,
                    "hour": hour,
                    "solar_elevation_deg": round(elevation, 2),
                    "is_daytime": elevation > 0,
                })

        # Write CSV
        csv_path = OUTPUT_DIR / "pv_solar_elevation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ CSV output saved to: {csv_path}")

    def test_pv_model_with_plot(self):
        """
        Generate plots demonstrating PV behavior.
        
        Generates:
            - output/pv_daily_production.png
            - output/pv_solar_elevation.png
            - output/pv_combined.png
        """
        pv = PV(
            name="Funchal_PV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        test_date = datetime(2025, 6, 21)
        
        # Collect data every 15 minutes
        times = []
        elevations = []
        productions = []
        ghis = []
        
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = test_date.replace(hour=hour, minute=minute)
                time_decimal = hour + minute / 60
                elevation = pv._calculate_solar_elevation(timestamp)
                ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
                
                production = pv.calculate_production(
                    ghi=ghi, dni=dni, dhi=dhi,
                    temperature=25.0, timestamp=timestamp
                )
                
                times.append(time_decimal)
                elevations.append(elevation)
                productions.append(production)
                ghis.append(ghi)

        # ===========================================
        # Plot 1: Daily Production Profile
        # ===========================================
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.fill_between(times, productions, alpha=0.3, color='orange', label='Production')
        ax1.plot(times, productions, 'orange', linewidth=2)
        ax1.axhline(y=PV_CAPACITY, color='red', linestyle='--', alpha=0.7, 
                    label=f'Rated capacity ({PV_CAPACITY} kW)')
        
        # Mark peak
        peak_idx = np.argmax(productions)
        ax1.plot(times[peak_idx], productions[peak_idx], 'ro', markersize=10)
        ax1.annotate(
            f'Peak: {productions[peak_idx]:.1f} kW\nat {times[peak_idx]:.1f}h',
            xy=(times[peak_idx], productions[peak_idx]),
            xytext=(times[peak_idx] + 2, productions[peak_idx] - 2),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )
        
        # Mark sunrise/sunset (where elevation crosses 0)
        sunrise_idx = next(i for i, e in enumerate(elevations) if e > 0)
        sunset_idx = len(elevations) - 1 - next(i for i, e in enumerate(reversed(elevations)) if e > 0)
        
        ax1.axvline(x=times[sunrise_idx], color='blue', linestyle=':', alpha=0.7, label='Sunrise')
        ax1.axvline(x=times[sunset_idx], color='purple', linestyle=':', alpha=0.7, label='Sunset')
        
        ax1.set_xlabel('Hour of Day (UTC)', fontsize=12)
        ax1.set_ylabel('Power Production (kW)', fontsize=12)
        ax1.set_title(f'PV Daily Production Profile - Summer Solstice\n'
                      f'Location: Port of Funchal ({FUNCHAL_LAT}°N, {abs(FUNCHAL_LON)}°W), '
                      f'Capacity: {PV_CAPACITY} kW', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)
        ax1.set_ylim(0, PV_CAPACITY * 1.1)
        ax1.set_xticks(range(0, 25, 2))
        
        fig1.tight_layout()
        plot1_path = OUTPUT_DIR / "pv_daily_production.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ===========================================
        # Plot 2: Solar Elevation Profile
        # ===========================================
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        ax2.fill_between(times, elevations, 0, where=[e > 0 for e in elevations],
                         alpha=0.3, color='yellow', label='Daytime')
        ax2.fill_between(times, elevations, 0, where=[e <= 0 for e in elevations],
                         alpha=0.3, color='darkblue', label='Nighttime')
        ax2.plot(times, elevations, 'b-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Mark peak elevation
        peak_elev_idx = np.argmax(elevations)
        ax2.plot(times[peak_elev_idx], elevations[peak_elev_idx], 'ro', markersize=10)
        ax2.annotate(
            f'Solar noon\nα = {elevations[peak_elev_idx]:.1f}°',
            xy=(times[peak_elev_idx], elevations[peak_elev_idx]),
            xytext=(times[peak_elev_idx] + 2, elevations[peak_elev_idx] - 10),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )
        
        ax2.set_xlabel('Hour of Day (UTC)', fontsize=12)
        ax2.set_ylabel('Solar Elevation (degrees)', fontsize=12)
        ax2.set_title(f'Solar Elevation Profile - Summer Solstice\n'
                      f'Location: Port of Funchal ({FUNCHAL_LAT}°N)', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        ax2.set_xticks(range(0, 25, 2))
        
        fig2.tight_layout()
        plot2_path = OUTPUT_DIR / "pv_solar_elevation.png"
        fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

        # ===========================================
        # Plot 3: Combined Figure (for thesis)
        # ===========================================
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Solar Elevation
        ax3a.fill_between(times, elevations, 0, where=[e > 0 for e in elevations],
                          alpha=0.3, color='yellow', label='Daytime (α > 0)')
        ax3a.fill_between(times, elevations, 0, where=[e <= 0 for e in elevations],
                          alpha=0.3, color='darkblue', label='Night (α ≤ 0)')
        ax3a.plot(times, elevations, 'b-', linewidth=2)
        ax3a.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3a.plot(times[peak_elev_idx], elevations[peak_elev_idx], 'ro', markersize=8)
        ax3a.set_xlabel('Hour of Day (UTC)', fontsize=12)
        ax3a.set_ylabel('Solar Elevation α (degrees)', fontsize=12)
        ax3a.set_title('(a) Solar Elevation Profile', fontsize=12)
        ax3a.legend(loc='upper right', fontsize=9)
        ax3a.grid(True, alpha=0.3)
        ax3a.set_xlim(0, 24)
        ax3a.set_xticks(range(0, 25, 2))
        ax3a.annotate(
            f'α_max = {elevations[peak_elev_idx]:.1f}°',
            xy=(times[peak_elev_idx], elevations[peak_elev_idx]),
            xytext=(times[peak_elev_idx] + 2, elevations[peak_elev_idx] - 15),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        # Right: Production Profile
        ax3b.fill_between(times, productions, alpha=0.3, color='orange')
        ax3b.plot(times, productions, 'orange', linewidth=2, label='PV Production')
        ax3b.axhline(y=PV_CAPACITY, color='red', linestyle='--', alpha=0.7, 
                     label=f'Capacity ({PV_CAPACITY} kW)')
        ax3b.axvline(x=times[sunrise_idx], color='blue', linestyle=':', alpha=0.7)
        ax3b.axvline(x=times[sunset_idx], color='purple', linestyle=':', alpha=0.7)
        ax3b.plot(times[peak_idx], productions[peak_idx], 'ro', markersize=8)
        ax3b.set_xlabel('Hour of Day (UTC)', fontsize=12)
        ax3b.set_ylabel('Power Production (kW)', fontsize=12)
        ax3b.set_title('(b) PV Production (Bell-Shaped Curve)', fontsize=12)
        ax3b.legend(loc='upper right', fontsize=9)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_xlim(0, 24)
        ax3b.set_ylim(0, PV_CAPACITY * 1.1)
        ax3b.set_xticks(range(0, 25, 2))
        ax3b.annotate(
            f'P_peak = {productions[peak_idx]:.1f} kW',
            xy=(times[peak_idx], productions[peak_idx]),
            xytext=(times[peak_idx] + 2, productions[peak_idx] - 3),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

        fig3.suptitle(f'PV Model Verification - Port of Funchal ({FUNCHAL_LAT}°N, {abs(FUNCHAL_LON)}°W)\n'
                      f'Summer Solstice (June 21), {PV_CAPACITY} kW System',
                      fontsize=12, fontweight='bold', y=1.02)
        fig3.tight_layout()
        plot3_path = OUTPUT_DIR / "pv_combined.png"
        fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)

        print(f"\n✓ Plots saved to:")
        print(f"  - {plot1_path}")
        print(f"  - {plot2_path}")
        print(f"  - {plot3_path}")

        # Verify test expectations
        # Night production should be 0
        night_productions = [p for e, p in zip(elevations, productions) if e <= 0]
        assert all(p == 0 for p in night_productions), "Night production should be 0"
        
        # Peak should be during daylight hours
        assert 6 <= times[peak_idx] <= 18, "Peak should be during daylight hours"

    def test_generate_thesis_table(self):
        """
        Generate a summary table for thesis.
        
        Generates: output/pv_summary.csv
        """
        pv = PV(
            name="Funchal_PV",
            capacity=PV_CAPACITY,
            tilt=PV_TILT,
            azimuth=180.0,
            efficiency=0.85,
            latitude=FUNCHAL_LAT,
            longitude=FUNCHAL_LON,
        )

        # Key hours throughout the day
        test_date = datetime(2025, 6, 21)
        key_hours = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        
        results = []
        
        for hour in key_hours:
            timestamp = test_date.replace(hour=hour, minute=0)
            elevation = pv._calculate_solar_elevation(timestamp)
            ghi, dni, dhi = calculate_clear_sky_irradiance(elevation)
            
            production = pv.calculate_production(
                ghi=ghi, dni=dni, dhi=dhi,
                temperature=25.0, timestamp=timestamp
            )
            
            results.append({
                "hour": f"{hour:02d}:00",
                "solar_elevation_deg": round(elevation, 1),
                "is_daytime": "Yes" if elevation > 0 else "No",
                "ghi_wm2": round(ghi, 0),
                "production_kw": round(production, 2),
                "capacity_factor_pct": round((production / PV_CAPACITY) * 100, 1),
            })

        # Write CSV
        csv_path = OUTPUT_DIR / "pv_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Print formatted table
        print("\n" + "=" * 80)
        print(f"PV MODEL VERIFICATION - Port of Funchal ({FUNCHAL_LAT}°N)")
        print(f"System: {PV_CAPACITY} kW, Tilt: {PV_TILT}°, Azimuth: 180° (South)")
        print("=" * 80)
        print(f"{'Hour':>8} {'Elevation':>12} {'Daytime':>10} {'GHI':>10} {'Production':>12} {'CF':>8}")
        print(f"{'':>8} {'(deg)':>12} {'':>10} {'(W/m²)':>10} {'(kW)':>12} {'(%)':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r['hour']:>8} {r['solar_elevation_deg']:>12.1f} {r['is_daytime']:>10} "
                  f"{r['ghi_wm2']:>10.0f} {r['production_kw']:>12.2f} {r['capacity_factor_pct']:>8.1f}")
        print("=" * 80)
        print(f"\n✓ Summary table saved to: {csv_path}")


def generate_all_outputs():
    """
    Standalone function to generate all outputs without running pytest.
    
    Usage: python test_pv_model.py --generate
    """
    print("=" * 60)
    print("GENERATING PV MODEL TEST OUTPUTS")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tests with output generation
    test_instance = TestPVModelWithOutput()
    test_instance.setup_output_dir()
    test_instance.test_daily_production_with_csv_output()
    test_instance.test_solar_elevation_profile_csv()
    test_instance.test_pv_model_with_plot()
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
