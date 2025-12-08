"""Port energy consumption and production forecasting."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from models import Port, Boat, Trip, BoatState
from database import DatabaseManager


@dataclass
class EnergyForecast:
    """Energy forecast for a specific time period."""
    
    timestamp: datetime
    power_active_consumption_kwh: float  # Expected consumption in the timestep
    power_active_production_kwh: float  # Expected PV production in the timestep
    bess_available_kwh: float  # BESS energy available for discharge
    bess_capacity_kwh: float  # BESS capacity available for charging
    net_balance_kwh: float  # Net energy balance (production - consumption)
    boat_states: Dict[str, BoatState]  # Forecasted state for each boat


class PortForecaster:
    """Forecast energy consumption and production for the port."""
    
    def __init__(
        self,
        port: Port,
        db_manager: DatabaseManager,
        timestep_seconds: int = 900
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
        self,
        forecast_date: datetime,
        trip_assignments: Dict[str, List[Trip]]
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
            power_active_consumption = self._forecast_power_active_consumption(timestamp, trip_assignments)
            
            # Forecast PV production
            power_active_production = self._forecast_power_active_production(timestamp, weather_forecasts)
            
            # Calculate BESS availability (assuming current SOC)
            bess_available, bess_capacity = self._calculate_bess_availability()
            
            # Net balance
            net_balance = power_active_production - power_active_consumption
            
            forecast = EnergyForecast(
                timestamp=timestamp,
                power_active_consumption_kwh=power_active_consumption,
                power_active_production_kwh=power_active_production,
                bess_available_kwh=bess_available,
                bess_capacity_kwh=bess_capacity,
                net_balance_kwh=net_balance,
                boat_states=boat_states
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _forecast_power_active_consumption(
        self,
        timestamp: datetime,
        trip_assignments: Dict[str, List[Trip]]
    ) -> float:
        """
        Forecast active power consumption at a specific timestamp.
        
        Args:
            timestamp: Timestamp to forecast
            trip_assignments: Trip assignments for boats
            
        Returns:
            Expected energy consumption in kWh for the timestep
        """
        total_power_active_consumption = 0.0
        
        for boat in self.port.boats:
            # Check if boat is on a trip at this timestamp
            boat_power_active_consumption = self._forecast_boat_power_active_consumption(
                boat, timestamp, trip_assignments.get(boat.name, [])
            )
            total_power_active_consumption += boat_power_active_consumption
        
        # Note: Charger consumption is based on boat charging needs
        # This will be calculated in the optimization phase
        
        return total_power_active_consumption
    
    def _forecast_boat_states(
        self,
        timestamp: datetime,
        trip_assignments: Dict[str, List[Trip]],
        boat_trip_progress: Dict[str, tuple]
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
                if current_hour == start_hour and current_minute < (self.timestep_seconds / 60):
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
    
    def _forecast_boat_power_active_consumption(
        self,
        boat: Boat,
        timestamp: datetime,
        trips: List[Trip]
    ) -> float:
        """
        Forecast a single boat's active power consumption.
        
        Args:
            boat: Boat to forecast
            timestamp: Timestamp to check
            trips: Trips assigned to this boat
            
        Returns:
            Expected energy consumption in kWh for the timestep
        """
        # Check if boat is on a trip at this timestamp
        # Trips typically start at 9:00 and 14:00
        for trip in trips:
            # Check if this trip would be active at this timestamp
            # This is a simplified check - actual trip timing depends on schedule
            hour = timestamp.hour
            
            # Morning trip (9:00-13:00 approximately)
            if hour >= 9 and hour < 13:
                # Use average power consumption for the trip
                avg_power = trip.estimate_energy_required(boat.k) / (trip.duration / 3600)
                energy_kwh = avg_power * (self.timestep_seconds / 3600)
                return energy_kwh
            
            # Afternoon trip (14:00-18:00 approximately)
            elif hour >= 14 and hour < 18:
                # Second trip
                if len(trips) > 1:
                    trip = trips[1]
                avg_power = trip.estimate_energy_required(boat.k) / (trip.duration / 3600)
                energy_kwh = avg_power * (self.timestep_seconds / 3600)
                return energy_kwh
        
        # Not on a trip - no motor consumption
        return 0.0
    
    def _forecast_power_active_production(
        self,
        timestamp: datetime,
        weather_forecasts: Dict[str, Dict[str, float]]
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
        conditions = weather_forecasts.get(ts_str, {
            'ghi': 0.0,
            'dni': 0.0,
            'dhi': 0.0,
            'temperature': 20.0
        })
        
        # Calculate production for each PV system
        for pv in self.port.pv_systems:
            # Calculate production (same logic as PV model)
            production_kw = pv.calculate_production(
                ghi=conditions.get('ghi', 0.0),
                dni=conditions.get('dni', 0.0),
                dhi=conditions.get('dhi', 0.0),
                temperature=conditions.get('temperature', 20.0),
                timestamp=timestamp
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
        self,
        forecast_date: datetime
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
        metrics = ['ghi', 'dni', 'dhi', 'temperature']
        
        for metric in metrics:
            metric_id = self.db_manager.get_metric_id(metric)
            forecast_rows = self.db_manager.get_records(
                "forecast",
                source_id=openmeteo_src,
                metric_id=metric_id,
                start_time=start_str,
                end_time=end_str
            )
            
            for row in forecast_rows:
                ts_str = row['timestamp']
                if ts_str not in weather_data:
                    weather_data[ts_str] = {}
                weather_data[ts_str][metric] = float(row['value'])
        
        return weather_data
    
    def save_forecasts_to_db(
        self,
        forecasts: List[EnergyForecast],
        forecast_type: str = "port_energy"
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
        power_active_consumption_met = self.db_manager.get_metric_id("power_active_consumption")
        power_active_production_met = self.db_manager.get_metric_id("power_active_production")
        bess_available_met = self.db_manager.get_metric_id("bess_available")
        bess_capacity_met = self.db_manager.get_metric_id("bess_capacity")
        net_balance_met = self.db_manager.get_metric_id("net_balance")
        state_met = self.db_manager.get_metric_id("state")
        
        for forecast in forecasts:
            ts_str = forecast.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save port-level aggregated metrics with port source
            forecast_data.append((ts_str, port_src, power_active_consumption_met, str(forecast.power_active_consumption_kwh)))
            forecast_data.append((ts_str, port_src, power_active_production_met, str(forecast.power_active_production_kwh)))
            forecast_data.append((ts_str, port_src, bess_available_met, str(forecast.bess_available_kwh)))
            forecast_data.append((ts_str, port_src, bess_capacity_met, str(forecast.bess_capacity_kwh)))
            forecast_data.append((ts_str, port_src, net_balance_met, str(forecast.net_balance_kwh)))
            
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
        
        total_power_active_consumption = sum(f.power_active_consumption_kwh for f in forecasts)
        total_power_active_production = sum(f.power_active_production_kwh for f in forecasts)
        avg_bess_available = sum(f.bess_available_kwh for f in forecasts) / len(forecasts)
        
        print(f"\n  📊 Energy Forecast Summary (24h):")
        print(f"     Total Consumption: {total_power_active_consumption:.2f} kWh")
        print(f"     Total Production: {total_power_active_production:.2f} kWh")
        print(f"     Avg BESS Available: {avg_bess_available:.2f} kWh")
        print(f"     Net Balance: {total_power_active_production - total_power_active_consumption:.2f} kWh")
        
        # Peak consumption time
        peak_forecast = max(forecasts, key=lambda f: f.power_active_consumption_kwh)
        print(f"     Peak Consumption: {peak_forecast.power_active_consumption_kwh:.2f} kWh at {peak_forecast.timestamp.strftime('%H:%M')}")
        
        # Peak production time
        peak_production = max(forecasts, key=lambda f: f.power_active_production_kwh)
        print(f"     Peak Production: {peak_production.power_active_production_kwh:.2f} kWh at {peak_production.timestamp.strftime('%H:%M')}")

