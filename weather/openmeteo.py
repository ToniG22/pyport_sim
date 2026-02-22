"""Weather data via the Open-Meteo forecast and historical APIs."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests


class OpenMeteoClient:
    """Fetches hourly weather and irradiance from Open-Meteo for a fixed (lat, lon)."""

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float, longitude: float):
        """
        Initialize the client for a single location.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
        """
        self.latitude = latitude
        self.longitude = longitude

    def fetch_forecast(
        self, start_date: datetime, days: int = 7
    ) -> Optional[Dict[str, List]]:
        """
        Fetch hourly weather for the date range; uses historical API for past dates, forecast API otherwise.

        Args:
            start_date: Start date (timezone-aware or naive).
            days: Number of days to fetch (forecast free tier typically limited to 7).

        Returns:
            Parsed dict with 'timestamps' and metric lists, or None on request error.
        """
        end_date = start_date + timedelta(days=days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        is_historical = start_date.replace(tzinfo=None) < today

        if is_historical:
            return self._fetch_historical(start_str, end_str)
        else:
            return self._fetch_forecast(start_str, end_str)

    def _fetch_forecast(
        self, start_str: str, end_str: str
    ) -> Optional[Dict[str, List]]:
        """Request hourly data from the forecast API for current/future dates."""
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
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
            ],
            "timezone": "UTC",
        }

        try:
            response = requests.get(self.FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return None

    def _fetch_historical(
        self, start_str: str, end_str: str
    ) -> Optional[Dict[str, List]]:
        """Request hourly data from the archive API for past dates."""
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
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
            ],
            "timezone": "UTC",
        }

        try:
            print("Using historical weather API for past dates")
            response = requests.get(self.HISTORICAL_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data: {e}")
            return None

    def _parse_response(self, data: dict) -> Dict[str, List]:
        """
        Convert Open-Meteo hourly payload into a dict of timestamps and metric-name lists.

        Args:
            data: Raw JSON response (must contain "hourly" with "time" and metric arrays).

        Returns:
            Dict with "timestamps" (list of datetime) and keys like "temperature", "ghi", "dni", "dhi".
        """
        if "hourly" not in data:
            return {}

        hourly = data["hourly"]
        timestamps = hourly.get("time", [])
        parsed_timestamps = [
            datetime.strptime(ts, "%Y-%m-%dT%H:%M") for ts in timestamps
        ]
        parsed_data = {"timestamps": parsed_timestamps}
        metric_mapping = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "dew_point_2m": "dew_point",
            "precipitation": "precipitation",
            "weather_code": "weather_code",
            "cloud_cover": "cloud_cover",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "shortwave_radiation": "ghi",
            "direct_radiation": "direct_radiation",
            "diffuse_radiation": "dhi",
            "direct_normal_irradiance": "dni",
        }

        for api_field, metric_name in metric_mapping.items():
            if api_field in hourly:
                parsed_data[metric_name] = hourly[api_field]

        return parsed_data

    def get_current_conditions(self, current_time: datetime) -> Optional[Dict]:
        """
        Return conditions for the hour closest to current_time (fetches 1-day forecast).

        Args:
            current_time: Time to resolve (timezone-aware or naive).

        Returns:
            Dict with "timestamp" and one value per metric for that hour, or None if fetch fails.
        """
        forecast = self.fetch_forecast(current_time, days=1)

        if not forecast or "timestamps" not in forecast:
            return None

        timestamps = forecast["timestamps"]
        closest_idx = min(
            range(len(timestamps)),
            key=lambda i: abs((timestamps[i] - current_time).total_seconds()),
        )
        conditions = {"timestamp": timestamps[closest_idx]}

        for key, values in forecast.items():
            if key != "timestamps" and len(values) > closest_idx:
                conditions[key] = values[closest_idx]

        return conditions

    def __repr__(self) -> str:
        return f"OpenMeteoClient(lat={self.latitude}, lon={self.longitude})"
