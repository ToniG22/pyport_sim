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
    consumption_kwh: float  # Expected consumption in the timestep
    pv_production_kwh: float  # Expected PV production in the timestep
    bess_available_kwh: float  # BESS energy available for discharge
    bess_capacity_kwh: float  # BESS capacity available for charging
    net_balance_kwh: float  # Net energy balance (production - consumption)


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
        
        # For each timestep in the day
        for step in range(timesteps_per_day):
            timestamp = forecast_date + timedelta(seconds=step * self.timestep_seconds)
            
            # Forecast consumption (boats on trips + charging)
            consumption = self._forecast_consumption(timestamp, trip_assignments)
            
            # Forecast PV production
            pv_production = self._forecast_pv_production(timestamp, weather_forecasts)
            
            # Calculate BESS availability (assuming current SOC)
            bess_available, bess_capacity = self._calculate_bess_availability()
            
            # Net balance
            net_balance = pv_production - consumption
            
            forecast = EnergyForecast(
                timestamp=timestamp,
                consumption_kwh=consumption,
                pv_production_kwh=pv_production,
                bess_available_kwh=bess_available,
                bess_capacity_kwh=bess_capacity,
                net_balance_kwh=net_balance
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _forecast_consumption(
        self,
        timestamp: datetime,
        trip_assignments: Dict[str, List[Trip]]
    ) -> float:
        """
        Forecast energy consumption at a specific timestamp.
        
        Args:
            timestamp: Timestamp to forecast
            trip_assignments: Trip assignments for boats
            
        Returns:
            Expected energy consumption in kWh for the timestep
        """
        total_consumption = 0.0
        
        for boat in self.port.boats:
            # Check if boat is on a trip at this timestamp
            boat_consumption = self._forecast_boat_consumption(
                boat, timestamp, trip_assignments.get(boat.name, [])
            )
            total_consumption += boat_consumption
        
        # Note: Charger consumption is based on boat charging needs
        # This will be calculated in the optimization phase
        
        return total_consumption
    
    def _forecast_boat_consumption(
        self,
        boat: Boat,
        timestamp: datetime,
        trips: List[Trip]
    ) -> float:
        """
        Forecast a single boat's energy consumption.
        
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
                    trip = trips[1] if len(trips) > 1 else trips[0]
                avg_power = trip.estimate_energy_required(boat.k) / (trip.duration / 3600)
                energy_kwh = avg_power * (self.timestep_seconds / 3600)
                return energy_kwh
        
        # Not on a trip - no motor consumption
        return 0.0
    
    def _forecast_pv_production(
        self,
        timestamp: datetime,
        weather_forecasts: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Forecast PV production at a specific timestamp.
        
        Args:
            timestamp: Timestamp to forecast
            weather_forecasts: Weather forecast data
            
        Returns:
            Expected PV production in kWh for the timestep
        """
        if not self.port.pv_systems:
            return 0.0
        
        total_production = 0.0
        
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
            total_production += energy_kwh
        
        return total_production
    
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
        
        # Fetch all weather metrics
        metrics = ['ghi', 'dni', 'dhi', 'temperature']
        
        for metric in metrics:
            forecast_rows = self.db_manager.get_forecasts(
                source='openmeteo',
                metric=metric,
                start_time=start_str,
                end_time=end_str
            )
            
            for row in forecast_rows:
                ts_str = row['timestamp']
                if ts_str not in weather_data:
                    weather_data[ts_str] = {}
                weather_data[ts_str][metric] = row['value']
        
        return weather_data
    
    def save_forecasts_to_db(
        self,
        forecasts: List[EnergyForecast],
        forecast_type: str = "energy"
    ) -> None:
        """
        Save energy forecasts to the database.
        
        Args:
            forecasts: List of energy forecasts
            forecast_type: Type identifier for the forecasts
        """
        forecast_data = []
        
        for forecast in forecasts:
            ts_str = forecast.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save each forecast metric
            forecast_data.append((ts_str, forecast_type, "consumption", forecast.consumption_kwh))
            forecast_data.append((ts_str, forecast_type, "pv_production", forecast.pv_production_kwh))
            forecast_data.append((ts_str, forecast_type, "bess_available", forecast.bess_available_kwh))
            forecast_data.append((ts_str, forecast_type, "bess_capacity", forecast.bess_capacity_kwh))
            forecast_data.append((ts_str, forecast_type, "net_balance", forecast.net_balance_kwh))
        
        if forecast_data:
            self.db_manager.save_forecasts_batch(forecast_data)
    
    def print_forecast_summary(self, forecasts: List[EnergyForecast]) -> None:
        """
        Print a summary of the forecasts.
        
        Args:
            forecasts: List of energy forecasts
        """
        if not forecasts:
            print("  No forecasts to display")
            return
        
        total_consumption = sum(f.consumption_kwh for f in forecasts)
        total_production = sum(f.pv_production_kwh for f in forecasts)
        avg_bess_available = sum(f.bess_available_kwh for f in forecasts) / len(forecasts)
        
        print(f"\n  📊 Energy Forecast Summary (24h):")
        print(f"     Total Consumption: {total_consumption:.2f} kWh")
        print(f"     Total PV Production: {total_production:.2f} kWh")
        print(f"     Avg BESS Available: {avg_bess_available:.2f} kWh")
        print(f"     Net Balance: {total_production - total_consumption:.2f} kWh")
        
        # Peak consumption time
        peak_forecast = max(forecasts, key=lambda f: f.consumption_kwh)
        print(f"     Peak Consumption: {peak_forecast.consumption_kwh:.2f} kWh at {peak_forecast.timestamp.strftime('%H:%M')}")
        
        # Peak production time
        peak_production = max(forecasts, key=lambda f: f.pv_production_kwh)
        print(f"     Peak Production: {peak_production.pv_production_kwh:.2f} kWh at {peak_production.timestamp.strftime('%H:%M')}")

