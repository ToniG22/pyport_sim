"""Weather data fetcher using Open-Meteo API."""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class OpenMeteoClient:
    """Client for fetching weather data from Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float, longitude: float):
        """
        Initialize Open-Meteo client.

        Args:
            latitude: Location latitude
            longitude: Location longitude
        """
        self.latitude = latitude
        self.longitude = longitude

    def fetch_forecast(
        self, start_date: datetime, days: int = 7
    ) -> Optional[Dict[str, List]]:
        """
        Fetch weather forecast from Open-Meteo.

        Args:
            start_date: Start date for forecast
            days: Number of days to forecast (max 7 for free tier)

        Returns:
            Dictionary with weather data, or None on error
        """
        # Calculate end date
        end_date = start_date + timedelta(days=days)

        # Format dates as strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # API parameters
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "shortwave_radiation",  # GHI equivalent
                "direct_radiation",  # DNI equivalent
                "diffuse_radiation",  # DHI equivalent
                "direct_normal_irradiance",  # Actual DNI
            ],
            "timezone": "UTC",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return self._parse_response(data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _parse_response(self, data: dict) -> Dict[str, List]:
        """
        Parse Open-Meteo API response.

        Args:
            data: Raw API response

        Returns:
            Parsed weather data with timestamps and values
        """
        if "hourly" not in data:
            return {}

        hourly = data["hourly"]
        timestamps = hourly.get("time", [])

        # Parse timestamps
        parsed_timestamps = [
            datetime.strptime(ts, "%Y-%m-%dT%H:%M") for ts in timestamps
        ]

        # Extract all available metrics
        parsed_data = {"timestamps": parsed_timestamps}

        # Map API fields to our metric names
        metric_mapping = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "precipitation": "precipitation",
            "weather_code": "weather_code",
            "cloud_cover": "cloud_cover",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "shortwave_radiation": "ghi",  # Global Horizontal Irradiance
            "direct_radiation": "direct_radiation",
            "diffuse_radiation": "dhi",  # Diffuse Horizontal Irradiance
            "direct_normal_irradiance": "dni",  # Direct Normal Irradiance
        }

        for api_field, metric_name in metric_mapping.items():
            if api_field in hourly:
                parsed_data[metric_name] = hourly[api_field]

        return parsed_data

    def get_current_conditions(self, current_time: datetime) -> Optional[Dict]:
        """
        Get weather conditions for a specific time from forecast data.

        Args:
            current_time: Time to get conditions for

        Returns:
            Dictionary with weather conditions, or None if not available
        """
        # Fetch forecast if not cached
        forecast = self.fetch_forecast(current_time, days=1)

        if not forecast or "timestamps" not in forecast:
            return None

        # Find closest timestamp
        timestamps = forecast["timestamps"]
        closest_idx = min(
            range(len(timestamps)),
            key=lambda i: abs((timestamps[i] - current_time).total_seconds()),
        )

        # Extract conditions for that timestamp
        conditions = {"timestamp": timestamps[closest_idx]}

        for key, values in forecast.items():
            if key != "timestamps" and len(values) > closest_idx:
                conditions[key] = values[closest_idx]

        return conditions

    def __repr__(self) -> str:
        return f"OpenMeteoClient(lat={self.latitude}, lon={self.longitude})"

