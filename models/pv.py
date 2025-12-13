"""PV (Photovoltaic) solar panel model."""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class PV:
    """
    Represents a photovoltaic (solar) system at the port.

    Attributes:
        name: Name/identifier of the PV system
        capacity: Rated capacity in kW (DC)
        tilt: Panel tilt angle in degrees (0=horizontal, 90=vertical)
        azimuth: Panel azimuth in degrees (0=North, 90=East, 180=South, 270=West)
        efficiency: System efficiency (0-1), accounting for inverter losses, etc.
        latitude: Installation latitude
        longitude: Installation longitude
        current_production: Current power production in kW
    """

    name: str
    capacity: float  # kW (DC rated)
    tilt: float = 30.0  # degrees
    azimuth: float = 180.0  # degrees (South-facing)
    efficiency: float = 0.85  # 85% system efficiency
    latitude: float = 0.0
    longitude: float = 0.0
    current_production: float = 0.0

    def __post_init__(self):
        """Validate PV attributes."""
        if self.capacity <= 0:
            raise ValueError("PV capacity must be positive")
        if not 0 <= self.tilt <= 90:
            raise ValueError("Tilt must be between 0 and 90 degrees")
        if not 0 <= self.azimuth < 360:
            raise ValueError("Azimuth must be between 0 and 360 degrees")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180")

    def calculate_production(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        temperature: float,
        timestamp,
    ) -> float:
        """
        Calculate PV power production using simplified model.

        Args:
            ghi: Global Horizontal Irradiance (W/m²)
            dni: Direct Normal Irradiance (W/m²)
            dhi: Diffuse Horizontal Irradiance (W/m²)
            temperature: Ambient temperature (°C)
            timestamp: Current datetime for sun position calculation

        Returns:
            Power production in kW
        """
        # Calculate sun position
        solar_elevation = self._calculate_solar_elevation(timestamp)

        if solar_elevation <= 0:
            # Sun is below horizon
            self.current_production = 0.0
            return 0.0

        # Calculate plane of array (POA) irradiance
        poa_irradiance = self._calculate_poa_irradiance(
            ghi, dni, dhi, solar_elevation, timestamp
        )

        # Temperature derating (power decreases ~0.4% per °C above 25°C)
        temp_coefficient = -0.004  # per °C
        cell_temperature = temperature + 25  # Simplified: cell temp ~25°C above ambient
        temp_factor = 1 + temp_coefficient * (cell_temperature - 25)

        # Calculate DC power (kW)
        # Standard test conditions: 1000 W/m², 25°C
        dc_power = (
            self.capacity * (poa_irradiance / 1000.0) * temp_factor * self.efficiency
        )

        # Ensure non-negative
        self.current_production = max(0.0, dc_power)
        return self.current_production

    def _calculate_solar_elevation(self, timestamp) -> float:
        """
        Calculate solar elevation angle using simplified formula.

        Args:
            timestamp: datetime object

        Returns:
            Solar elevation in degrees
        """
        # Day of year
        day_of_year = timestamp.timetuple().tm_yday

        # Solar declination (simplified)
        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

        # Hour angle
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = 15 * (hour - 12)  # degrees

        # Solar elevation
        lat_rad = math.radians(self.latitude)
        dec_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)

        sin_elevation = (
            math.sin(lat_rad) * math.sin(dec_rad)
            + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad)
        )

        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))

        return elevation

    def _calculate_poa_irradiance(
        self, ghi: float, dni: float, dhi: float, solar_elevation: float, timestamp
    ) -> float:
        """
        Calculate plane-of-array irradiance (simplified).

        Args:
            ghi: Global Horizontal Irradiance (W/m²)
            dni: Direct Normal Irradiance (W/m²)
            dhi: Diffuse Horizontal Irradiance (W/m²)
            solar_elevation: Solar elevation angle (degrees)
            timestamp: Current datetime

        Returns:
            POA irradiance in W/m²
        """
        # For now, use a simplified model
        # In a full implementation, this would use proper transposition models

        # Angle of incidence factor for south-facing panel
        tilt_rad = math.radians(self.tilt)
        elevation_rad = math.radians(solar_elevation)

        # Direct component on tilted surface
        # For a south-facing panel at solar noon: cos(θ) = sin(elevation + tilt)
        # This gives maximum when sun elevation + tilt = 90°
        cos_incident_angle = max(0, math.sin(elevation_rad + tilt_rad))
        direct_component = dni * cos_incident_angle

        # Diffuse component (isotropic sky model)
        diffuse_component = dhi * (1 + math.cos(tilt_rad)) / 2

        # Ground reflected component (albedo = 0.2)
        albedo = 0.2
        ground_component = ghi * albedo * (1 - math.cos(tilt_rad)) / 2

        poa = direct_component + diffuse_component + ground_component

        return max(0.0, poa)

    def __repr__(self) -> str:
        return (
            f"PV(name='{self.name}', capacity={self.capacity}kW, "
            f"tilt={self.tilt}°, azimuth={self.azimuth}°, "
            f"production={self.current_production:.2f}kW)"
        )

