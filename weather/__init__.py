"""
Weather and irradiance data for the simulator.

Exposes OpenMeteoClient for hourly weather and solar data (Open-Meteo forecast and archive APIs).
"""

from .openmeteo import OpenMeteoClient

__all__ = ["OpenMeteoClient"]
