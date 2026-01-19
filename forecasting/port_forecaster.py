"""Port energy consumption and production forecasting."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from database import DatabaseManager
from models import Boat, BoatState, Port, Trip


@dataclass
class EnergyForecast:
    """Energy forecast for a specific time period."""

    timestamp: datetime
    power_active_consumption_kwh: float  # Expected consumption in the timestep
    power_active_production_kwh: float  # Expected PV production in the timestep
    bess_available_kwh: float  # BESS energy available for discharge
    bess_capacity_kwh: float  # BESS capacity available for charging
    boat_states: Dict[str, BoatState]  # Forecasted state for each boat


class PortForecaster:
    """Forecast energy consumption and production for the port."""

    def __init__(
        self, port: Port, db_manager: DatabaseManager, timestep_seconds: int = 900
    ):
        """
        Initialize the port forecaster.

        Args:
            port: Port instance
            db_manager: Database manager for weather forecasts
            timestep_seconds: Simulation timestep in seconds
        """
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds

    def generate_daily_forecast(
        self, forecast_date: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> List[EnergyForecast]:
        """
        Generate 24-hour energy forecast for the port.

        Args:
            forecast_date: Date to forecast (should be at 00:00:00)
            trip_assignments: Dict mapping boat names to their assigned trips

        Returns:
            List of EnergyForecast objects for each timestep
        """
        forecasts = []

        # Number of timesteps in 24 hours
        timesteps_per_day = int(24 * 3600 / self.timestep_seconds)

        # Get weather forecast for the day (for PV production)
        weather_forecasts = self._get_weather_forecasts(forecast_date)

        # Track boat states and trip progress for forecasting
        boat_trip_progress = {}  # {boat_name: (trip, start_timestamp, elapsed_seconds)}

        # For each timestep in the day
        for step in range(timesteps_per_day):
            timestamp = forecast_date + timedelta(seconds=step * self.timestep_seconds)

            # Forecast boat states
            boat_states = self._forecast_boat_states(
                timestamp, trip_assignments, boat_trip_progress
            )

            # Forecast consumption (boats on trips + charging)
            power_active_consumption = self._forecast_power_active_consumption(
                timestamp, trip_assignments
            )

            # Forecast PV production
            power_active_production = self._forecast_power_active_production(
                timestamp, weather_forecasts
            )

            # Calculate BESS availability (assuming current SOC)
            bess_available, bess_capacity = self._calculate_bess_availability()

            forecast = EnergyForecast(
                timestamp=timestamp,
                power_active_consumption_kwh=power_active_consumption,
                power_active_production_kwh=power_active_production,
                bess_available_kwh=bess_available,
                bess_capacity_kwh=bess_capacity,
                boat_states=boat_states,
            )

            forecasts.append(forecast)

        return forecasts

    def _forecast_power_active_consumption(
        self, timestamp: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> float:
        """
        Forecast active power consumption at a specific timestamp.

        This forecasts the port's electrical consumption (chargers + BESS),
        NOT the boat's motor energy usage while sailing.

        Args:
            timestamp: Timestamp to forecast
            trip_assignments: Trip assignments for boats

        Returns:
            Expected energy consumption in kWh for the timestep
        """
        total_power_active_consumption = 0.0

        # Forecast charger consumption based on boats at port
        charger_consumption = self._forecast_charger_consumption(
            timestamp, trip_assignments
        )
        total_power_active_consumption += charger_consumption

        # BESS consumption (charging from grid/PV) is typically handled in optimization
        # For forecasting, we assume BESS may charge during low-demand/high-production periods
        # This is a simplified estimate - actual BESS behavior depends on control strategy

        return total_power_active_consumption

    def _forecast_boat_states(
        self,
        timestamp: datetime,
        trip_assignments: Dict[str, List[Trip]],
        boat_trip_progress: Dict[str, tuple],
    ) -> Dict[str, BoatState]:
        """
        Forecast boat states at a specific timestamp.

        Args:
            timestamp: Timestamp to forecast
            trip_assignments: Trip assignments for boats
            boat_trip_progress: Dict tracking active trips {boat_name: (trip, start_timestamp, elapsed_seconds)}

        Returns:
            Dictionary mapping boat names to their forecasted states
        """
        boat_states = {}
        current_hour = timestamp.hour
        current_minute = timestamp.minute

        # Trip schedule times: (start_hour, slot_number)
        trip_schedule = [(9, 0), (14, 1)]

        for boat in self.port.boats:
            boat_name = boat.name
            trips = trip_assignments.get(boat_name, [])

            # Check if boat is currently on a trip (from previous timesteps)
            if boat_name in boat_trip_progress:
                trip, start_timestamp, _ = boat_trip_progress[boat_name]

                # Calculate elapsed time from start timestamp
                elapsed = (timestamp - start_timestamp).total_seconds()

                # Check if trip is complete
                if elapsed >= trip.duration:
                    # Trip completed, boat returns to port
                    boat_states[boat_name] = BoatState.IDLE
                    del boat_trip_progress[boat_name]
                else:
                    # Still on trip
                    boat_states[boat_name] = BoatState.SAILING
                    boat_trip_progress[boat_name] = (trip, start_timestamp, elapsed)
                continue

            # Check if it's time to start a new trip
            trip_started = False
            for start_hour, slot in trip_schedule:
                # Start trip at the scheduled hour (check within the timestep window)
                if current_hour == start_hour and current_minute < (
                    self.timestep_seconds / 60
                ):
                    # Get trip for this slot
                    if slot < len(trips):
                        trip = trips[slot]

                        # Estimate energy required
                        estimated_energy = trip.estimate_energy_required(boat.k)
                        required_soc = estimated_energy / boat.battery_capacity

                        # Assume boat has enough charge (optimistic forecast)
                        # In reality, this might be delayed, but for forecasting we assume trips start on time
                        boat_states[boat_name] = BoatState.SAILING
                        boat_trip_progress[boat_name] = (trip, timestamp, 0.0)
                        trip_started = True
                        break

            if not trip_started:
                # Not on a trip - could be IDLE or CHARGING
                # For forecasting, we'll assume boats are IDLE when not sailing
                # (Charging state would require optimization knowledge)
                boat_states[boat_name] = BoatState.IDLE

        return boat_states

    def _forecast_charger_consumption(
        self, timestamp: datetime, trip_assignments: Dict[str, List[Trip]]
    ) -> float:
        if not self.port.chargers:
            return 0.0

        total_charger_consumption = 0.0

        boats_at_port = []
        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            is_sailing = self._is_boat_sailing(timestamp, trips)
            if not is_sailing:
                boats_at_port.append(boat)

        for boat in boats_at_port:
            trips = trip_assignments.get(boat.name, [])
            needs_charging = self._boat_needs_charging(timestamp, trips, boat)
            if needs_charging:
                charger_power = (
                    self.port.chargers[0].max_power if self.port.chargers else 0.0
                )
                energy_kwh = charger_power * (self.timestep_seconds / 3600)
                total_charger_consumption += energy_kwh

        return total_charger_consumption

    def _is_boat_sailing(self, timestamp: datetime, trips: List[Trip]) -> bool:
        """
        Check if a boat is sailing at a given timestamp using trip durations.

        Assumes slot 0 starts at 09:00, slot 1 starts at 14:00.
        """
        if not trips:
            return False

        # Define scheduled start times (same as engine)
        start_times = [
            timestamp.replace(hour=9, minute=0, second=0, microsecond=0),
            timestamp.replace(hour=14, minute=0, second=0, microsecond=0),
        ]

        # Slot 0
        if len(trips) >= 1:
            t0_start = start_times[0]
            t0_end = t0_start + timedelta(seconds=trips[0].duration)
            if t0_start <= timestamp < t0_end:
                return True

        # Slot 1
        if len(trips) >= 2:
            t1_start = start_times[1]
            t1_end = t1_start + timedelta(seconds=trips[1].duration)
            if t1_start <= timestamp < t1_end:
                return True

        return False

    def _boat_needs_charging(
        self, timestamp: datetime, trips: List[Trip], boat: Boat
    ) -> bool:
        """
        Estimate if a boat needs charging based on time windows derived from trip durations.
        This is still a heuristic, but now time-accurate.
        """
        if not trips:
            return False

        t_day_0000 = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        t_trip0_start = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
        t_trip1_start = timestamp.replace(hour=14, minute=0, second=0, microsecond=0)

        # Before first trip: allow charging from midnight until trip start
        if t_day_0000 <= timestamp < t_trip0_start:
            return True

        # Between trips: allow charging from end of trip 0 until trip 1 start
        if len(trips) >= 2:
            t_trip0_end = t_trip0_start + timedelta(seconds=trips[0].duration)
            if t_trip0_end <= timestamp < t_trip1_start:
                return True

        # After last trip: allow charging after trip 1 ends (or trip 0 ends if only one trip)
        if len(trips) >= 2:
            t_trip1_end = t_trip1_start + timedelta(seconds=trips[1].duration)
            if timestamp >= t_trip1_end:
                return True
        else:
            t_trip0_end = t_trip0_start + timedelta(seconds=trips[0].duration)
            if timestamp >= t_trip0_end:
                return True

        return False

    def _forecast_power_active_production(
        self, timestamp: datetime, weather_forecasts: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Forecast active power production at a specific timestamp.

        Args:
            timestamp: Timestamp to forecast
            weather_forecasts: Weather forecast data

        Returns:
            Expected power production in kWh for the timestep
        """
        if not self.port.pv_systems:
            return 0.0

        total_power_active_production = 0.0

        # Get weather conditions for this timestamp
        ts_str = timestamp.strftime("%Y-%m-%d %H:00:00")  # Round to hour
        conditions = weather_forecasts.get(
            ts_str, {"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temperature": 20.0}
        )

        # Calculate production for each PV system
        for pv in self.port.pv_systems:
            # Calculate production (same logic as PV model)
            production_kw = pv.calculate_production(
                ghi=conditions.get("ghi", 0.0),
                dni=conditions.get("dni", 0.0),
                dhi=conditions.get("dhi", 0.0),
                temperature=conditions.get("temperature", 20.0),
                timestamp=timestamp,
            )

            # Convert to energy for this timestep
            energy_kwh = production_kw * (self.timestep_seconds / 3600)
            total_power_active_production += energy_kwh

        return total_power_active_production

    def _calculate_bess_availability(self) -> Tuple[float, float]:
        """
        Calculate current BESS energy availability.

        Returns:
            Tuple of (available_for_discharge_kwh, available_for_charge_kwh)
        """
        if not self.port.bess_systems:
            return 0.0, 0.0

        total_available = 0.0
        total_capacity = 0.0

        for bess in self.port.bess_systems:
            # Energy available for discharge
            available = bess.get_available_energy()
            total_available += available

            # Capacity available for charging
            capacity = bess.get_available_charge_capacity()
            total_capacity += capacity

        return total_available, total_capacity

    def _get_weather_forecasts(
        self, forecast_date: datetime
    ) -> Dict[str, Dict[str, float]]:
        """
        Get weather forecasts for a specific date from database.

        Args:
            forecast_date: Date to get forecasts for

        Returns:
            Dictionary mapping timestamp strings to weather metrics
        """
        weather_data = {}

        # Get forecast for 24 hours
        start_str = forecast_date.strftime("%Y-%m-%d 00:00:00")
        end_str = (forecast_date + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")

        # Get source ID for openmeteo
        openmeteo_src = self.db_manager.get_or_create_source("openmeteo", "weather")

        # Fetch all weather metrics
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
        self, forecasts: List[EnergyForecast], forecast_type: str = "port_energy"
    ) -> None:
        """
        Save energy forecasts to the database.

        Args:
            forecasts: List of energy forecasts
            forecast_type: Type identifier for the forecasts (used as source_type)
        """
        forecast_data = []

        # Get source IDs - use port as source for port-level aggregated metrics
        port_src = self.db_manager.get_or_create_source(self.port.name, "port")

        # Get metric IDs (use same metric names as measurements for consistency)
        power_active_consumption_met = self.db_manager.get_metric_id(
            "power_active_consumption"
        )
        power_active_production_met = self.db_manager.get_metric_id(
            "power_active_production"
        )
        bess_available_met = self.db_manager.get_metric_id("bess_available")
        bess_capacity_met = self.db_manager.get_metric_id("bess_capacity")
        state_met = self.db_manager.get_metric_id("state")

        for forecast in forecasts:
            ts_str = forecast.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Save port-level aggregated metrics with port source
            forecast_data.append(
                (
                    ts_str,
                    port_src,
                    power_active_consumption_met,
                    str(forecast.power_active_consumption_kwh),
                )
            )
            forecast_data.append(
                (
                    ts_str,
                    port_src,
                    power_active_production_met,
                    str(forecast.power_active_production_kwh),
                )
            )
            forecast_data.append(
                (ts_str, port_src, bess_available_met, str(forecast.bess_available_kwh))
            )
            forecast_data.append(
                (ts_str, port_src, bess_capacity_met, str(forecast.bess_capacity_kwh))
            )

            # Save boat state forecasts (metric="state", source=boat_name)
            # Use same format as measurements: 1.0 for sailing, 0.0 otherwise
            for boat_name, boat_state in forecast.boat_states.items():
                boat_src = self.db_manager.get_or_create_source(boat_name, "boat")
                state_value = 1.0 if boat_state == BoatState.SAILING else 0.0
                forecast_data.append((ts_str, boat_src, state_met, str(state_value)))

        if forecast_data:
            self.db_manager.save_records_batch("forecast", forecast_data)

    def print_forecast_summary(self, forecasts: List[EnergyForecast]) -> None:
        """
        Print a summary of the forecasts.

        Args:
            forecasts: List of energy forecasts
        """
        if not forecasts:
            print("  No forecasts to display")
            return

        total_power_active_consumption = sum(
            f.power_active_consumption_kwh for f in forecasts
        )
        total_power_active_production = sum(
            f.power_active_production_kwh for f in forecasts
        )
        avg_bess_available = sum(f.bess_available_kwh for f in forecasts) / len(
            forecasts
        )

        print(f"\n  📊 Energy Forecast Summary (24h):")
        print(f"     Total Consumption: {total_power_active_consumption:.2f} kWh")
        print(f"     Total Production: {total_power_active_production:.2f} kWh")
        print(f"     Avg BESS Available: {avg_bess_available:.2f} kWh")

        # Peak consumption time
        peak_forecast = max(forecasts, key=lambda f: f.power_active_consumption_kwh)
        print(
            f"     Peak Consumption: {peak_forecast.power_active_consumption_kwh:.2f} kWh at {peak_forecast.timestamp.strftime('%H:%M')}"
        )

        # Peak production time
        peak_production = max(forecasts, key=lambda f: f.power_active_production_kwh)
        print(
            f"     Peak Production: {peak_production.power_active_production_kwh:.2f} kWh at {peak_production.timestamp.strftime('%H:%M')}"
        )
