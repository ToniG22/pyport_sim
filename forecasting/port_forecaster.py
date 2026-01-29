"""Port energy production and boat availability forecasting."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from database import DatabaseManager
from models import Boat, Port, Trip


@dataclass
class EnergyForecast:
    """Forecast for a single timestep."""

    timestamp: datetime
    # PV production in kW: per source and port sum
    power_active_production_kw_by_source: Dict[str, float]
    power_active_production_kw: float
    # Boat energy need until next departure (kWh)
    boat_required_energy_kwh: Dict[str, float]
    # Connectivity to charge: 0 = not available, 1 = available
    boat_available: Dict[str, int]


class PortForecaster:
    """Forecast PV production and boat availability for the port."""

    def __init__(
        self, port: Port, db_manager: DatabaseManager, timestep_seconds: int = 900
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds

    def generate_daily_forecast(
        self, forecast_date: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> List[EnergyForecast]:
        """
        Generate 24-hour forecast for the port.

        Forecasted variables:
        - power_active_production (kW) per PV and port sum
        - boat_required_energy (kWh) until next departure
        - boat_available (0 or 1) per boat
        """
        forecasts = []
        timesteps_per_day = int(24 * 3600 / self.timestep_seconds)
        weather_forecasts = self._get_weather_forecasts(forecast_date)

        for step in range(timesteps_per_day):
            timestamp = forecast_date + timedelta(
                seconds=step * self.timestep_seconds
            )

            by_source, total_kw = self._forecast_power_active_production(
                timestamp, weather_forecasts
            )
            boat_required_energy = self._forecast_boat_required_energy(
                timestamp, trip_assignments
            )
            boat_available = self._forecast_boat_available(
                timestamp, trip_assignments
            )

            forecast = EnergyForecast(
                timestamp=timestamp,
                power_active_production_kw_by_source=by_source,
                power_active_production_kw=total_kw,
                boat_required_energy_kwh=boat_required_energy,
                boat_available=boat_available,
            )
            forecasts.append(forecast)

        return forecasts

    def _forecast_power_active_production(
        self, timestamp: datetime, weather_forecasts: Dict[str, Dict[str, float]]
    ) -> tuple:
        """
        Forecast PV power production at timestamp (kW per source and total).
        """
        by_source: Dict[str, float] = {}
        total_kw = 0.0

        if not self.port.pv_systems:
            return by_source, total_kw

        ts_str = timestamp.strftime("%Y-%m-%d %H:00:00")
        conditions = weather_forecasts.get(
            ts_str, {"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temperature": 20.0}
        )

        for pv in self.port.pv_systems:
            production_kw = pv.calculate_production(
                ghi=conditions.get("ghi", 0.0),
                dni=conditions.get("dni", 0.0),
                dhi=conditions.get("dhi", 0.0),
                temperature=conditions.get("temperature", 20.0),
                timestamp=timestamp,
            )
            by_source[pv.name] = production_kw
            total_kw += production_kw

        return by_source, total_kw

    def _forecast_boat_required_energy(
        self, timestamp: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> Dict[str, float]:
        """Boat energy need (kWh) until next departure deadline."""
        out: Dict[str, float] = {}
        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            next_trip = self._next_departure_trip(timestamp, trips)
            if next_trip is None:
                out[boat.name] = 0.0
            else:
                out[boat.name] = next_trip.estimate_energy_required(boat.k)
        return out

    def _next_departure_trip(
        self, timestamp: datetime, trips: List[Trip]
    ) -> Optional[Trip]:
        """Next trip (by departure time) after timestamp; None if no more trips."""
        if not trips:
            return None
        t_trip0 = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
        t_trip1 = timestamp.replace(hour=14, minute=0, second=0, microsecond=0)

        if timestamp < t_trip0 and len(trips) >= 1:
            return trips[0]
        if len(trips) >= 2:
            dur0 = trips[0].duration
            t_trip0_end = t_trip0 + timedelta(seconds=dur0)
            if t_trip0_end <= timestamp < t_trip1:
                return trips[1]
            if timestamp >= t_trip1:
                return None  # after last departure
        else:
            dur0 = trips[0].duration
            t_trip0_end = t_trip0 + timedelta(seconds=dur0)
            if timestamp >= t_trip0_end:
                return None
        return None

    def _forecast_boat_available(
        self, timestamp: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> Dict[str, int]:
        """Boat availability to charge: 1 = available, 0 = not available."""
        out: Dict[str, int] = {}
        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            is_sailing = self._is_boat_sailing(timestamp, trips)
            in_window = self._in_charging_window(timestamp, trips, boat)
            out[boat.name] = 1 if (not is_sailing and in_window) else 0
        return out

    def _is_boat_sailing(self, timestamp: datetime, trips: List[Trip]) -> bool:
        """True if boat is sailing at timestamp (within a trip)."""
        if not trips:
            return False
        start_times = [
            timestamp.replace(hour=9, minute=0, second=0, microsecond=0),
            timestamp.replace(hour=14, minute=0, second=0, microsecond=0),
        ]
        if len(trips) >= 1:
            t0_end = start_times[0] + timedelta(seconds=trips[0].duration)
            if start_times[0] <= timestamp < t0_end:
                return True
        if len(trips) >= 2:
            t1_end = start_times[1] + timedelta(seconds=trips[1].duration)
            if start_times[1] <= timestamp < t1_end:
                return True
        return False

    def _in_charging_window(
        self, timestamp: datetime, trips: List[Trip], _boat: Boat
    ) -> bool:
        """True if timestamp is inside a charging availability window."""
        if not trips:
            return False
        t_day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        t_trip0 = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
        t_trip1 = timestamp.replace(hour=14, minute=0, second=0, microsecond=0)

        if t_day <= timestamp < t_trip0:
            return True
        if len(trips) >= 2:
            t_trip0_end = t_trip0 + timedelta(seconds=trips[0].duration)
            if t_trip0_end <= timestamp < t_trip1:
                return True
        if len(trips) >= 2:
            t_trip1_end = t_trip1 + timedelta(seconds=trips[1].duration)
            if timestamp >= t_trip1_end:
                return True
        else:
            t_trip0_end = t_trip0 + timedelta(seconds=trips[0].duration)
            if timestamp >= t_trip0_end:
                return True
        return False

    def _get_weather_forecasts(
        self, forecast_date: datetime
    ) -> Dict[str, Dict[str, float]]:
        """
        Get weather forecasts for a specific date from database.
        """
        weather_data = {}

        start_str = forecast_date.strftime("%Y-%m-%d 00:00:00")
        end_str = (forecast_date + timedelta(days=1)).strftime(
            "%Y-%m-%d 00:00:00"
        )

        openmeteo_src = self.db_manager.get_or_create_source(
            "openmeteo", "weather"
        )
        metrics = ["ghi", "dni", "dhi", "temperature"]

        for metric in metrics:
            metric_id = self.db_manager.get_metric_id(metric)
            forecast_rows = self.db_manager.get_records(
                "forecast",
                source_id=openmeteo_src,
                metric_id=metric_id,
                start_time=start_str,
                end_time=end_str,
            )

            for row in forecast_rows:
                ts_str = row["timestamp"]
                if ts_str not in weather_data:
                    weather_data[ts_str] = {}
                weather_data[ts_str][metric] = float(row["value"])

        return weather_data

    def save_forecasts_to_db(
        self, forecasts: List[EnergyForecast], forecast_type: str = "port_energy"  # pylint: disable=unused-argument
    ) -> None:
        """Save forecasts to DB: power_active_production (per PV + port), boat_required_energy, boat_available."""
        forecast_data = []
        port_src = self.db_manager.get_or_create_source(self.port.name, "port")
        power_met = self.db_manager.get_metric_id("power_active_production")
        boat_required_met = self.db_manager.get_metric_id(
            "boat_required_energy"
        )
        boat_available_met = self.db_manager.get_metric_id("boat_available")

        for forecast in forecasts:
            ts_str = forecast.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            for source_name, kw in forecast.power_active_production_kw_by_source.items():
                src = self.db_manager.get_or_create_source(
                    source_name, "pv"
                )
                forecast_data.append(
                    (ts_str, src, power_met, str(kw))
                )
            forecast_data.append(
                (
                    ts_str,
                    port_src,
                    power_met,
                    str(forecast.power_active_production_kw),
                )
            )

            for boat_name, kwh in forecast.boat_required_energy_kwh.items():
                boat_src = self.db_manager.get_or_create_source(
                    boat_name, "boat"
                )
                forecast_data.append(
                    (ts_str, boat_src, boat_required_met, str(kwh))
                )

            for boat_name, avail in forecast.boat_available.items():
                boat_src = self.db_manager.get_or_create_source(
                    boat_name, "boat"
                )
                forecast_data.append(
                    (ts_str, boat_src, boat_available_met, str(avail))
                )

        if forecast_data:
            self.db_manager.save_records_batch("forecast", forecast_data)

    def print_forecast_summary(self, forecasts: List[EnergyForecast]) -> None:
        """Print a short summary of the forecasts."""
        if not forecasts:
            print("  No forecasts to display")
            return

        total_production_kwh = sum(
            f.power_active_production_kw * (self.timestep_seconds / 3600)
            for f in forecasts
        )
        peak_f = max(forecasts, key=lambda x: x.power_active_production_kw)

        print("\n  📊 Forecast summary (24h):")
        print(f"     Total PV production: {total_production_kwh:.2f} kWh")
        print(
            f"     Peak PV power: {peak_f.power_active_production_kw:.2f} kW at {peak_f.timestamp.strftime('%H:%M')}"
        )
