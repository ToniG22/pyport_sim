"""PV system model using pvlib (POA irradiance, Sandia cell temperature, PVWatts DC)."""

from dataclasses import dataclass
import pvlib

REFERENCE_IRRADIANCE = 1000.0
REFERENCE_TEMP = 25.0
TEMP_COEFF_PDC = -0.004

SAPM_TEMP_PARAMS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_polymer"
]


@dataclass
class PV:
    """
    PV system with fixed tilt/azimuth; production from GHI/DNI/DHI and cell temperature.

    Attributes:
        name: System name.
        capacity: DC capacity at STC (kW).
        tilt: Panel tilt (degrees); default 30.
        azimuth: Panel azimuth (degrees); default 180.
        latitude: Site latitude.
        longitude: Site longitude.
        current_production: Last computed production (kW).
    """

    name: str
    capacity: float
    tilt: float = 30.0
    azimuth: float = 180.0
    latitude: float = 0.0
    longitude: float = 0.0
    current_production: float = 0.0

    def __post_init__(self):
        """Validate capacity, tilt, azimuth, and coordinates."""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not 0 <= self.tilt <= 90:
            raise ValueError("Tilt must be 0–90°")
        if not 0 <= self.azimuth < 360:
            raise ValueError("Azimuth must be 0–360°")
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be -90–90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be -180–180")

    def calculate_production(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        temperature: float,
        timestamp,
        wind_speed: float = 1.0,
    ) -> float:
        """
        Compute DC power (kW) at timestamp using POA irradiance, Sandia cell temperature, PVWatts DC.

        Args:
            ghi: Global horizontal irradiance (W/m²).
            dni: Direct normal irradiance (W/m²).
            dhi: Diffuse horizontal irradiance (W/m²).
            temperature: Air temperature (°C).
            timestamp: Time for solar position.
            wind_speed: Wind speed (m/s) for cell temperature; default 1.0.

        Returns:
            DC power in kW; 0 if sun below horizon. Also sets current_production.
        """
        solpos = pvlib.solarposition.get_solarposition(
            timestamp, self.latitude, self.longitude
        )

        if solpos["apparent_elevation"].iloc[0] <= 0:
            self.current_production = 0.0
            return 0.0

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith=solpos["zenith"].iloc[0],
            solar_azimuth=solpos["azimuth"].iloc[0],
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            albedo=0.2,
        )["poa_global"]

        cell_temperature = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=temperature,
            wind_speed=wind_speed,
            **SAPM_TEMP_PARAMS,
        )

        pdc = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=poa,
            temp_cell=cell_temperature,
            pdc0=self.capacity * REFERENCE_IRRADIANCE,
            gamma_pdc=TEMP_COEFF_PDC,
            temp_ref=REFERENCE_TEMP,
        )

        self.current_production = max(0.0, float(pdc) / 1000.0)
        return self.current_production

    def __repr__(self) -> str:
        return (
            f"PV(name='{self.name}', capacity={self.capacity}kW, "
            f"tilt={self.tilt}°, azimuth={self.azimuth}°, "
            f"production={self.current_production:.2f}kW)"
        )
