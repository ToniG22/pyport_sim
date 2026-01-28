from dataclasses import dataclass
import pvlib

# Constants for DC power model
REFERENCE_IRRADIANCE = 1000.0  # W/m² STC (standard test conditions)
REFERENCE_TEMP = 25.0  # °C STC (standard test conditions)
TEMP_COEFF_PDC = -0.004  # 1/°C, typical for crystalline silicon (PVWatts)

# Sandia temperature model parameters for open rack glass/polymer (PVWatts)
SAPM_TEMP_PARAMS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_polymer"
]


@dataclass
class PV:
    name: str
    capacity: float  # kW DC at STC (pdc0)
    tilt: float = 30.0
    azimuth: float = 180.0
    latitude: float = 0.0
    longitude: float = 0.0
    current_production: float = 0.0

    def __post_init__(self):
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
        wind_speed: float = 1.0,  # m/s (assumed to be 1 m/s)
    ) -> float:

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

        # Sandia cell temperature model
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
