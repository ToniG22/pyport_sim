"""Core simulation engine for the electric port simulator."""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from models import Port, BoatState, ChargerState, Trip
from database import DatabaseManager
from config import Settings, SimulationMode
from simulation.trip_manager import TripManager
from weather import OpenMeteoClient


class SimulationEngine:
    """Main simulation engine for the electric port."""

    def __init__(
        self,
        port: Port,
        settings: Settings,
        db_manager: DatabaseManager,
        start_date: Optional[datetime] = None,
        days: int = 1,
        trips_directory: str = "assets/trips",
    ):
        """
        Initialize the simulation engine.

        Args:
            port: Port instance (contains boats and chargers)
            settings: Simulation settings
            db_manager: Database manager
            start_date: Simulation start date (default: today at midnight UTC)
            days: Number of days to simulate (max 7)
            trips_directory: Directory containing trip CSV files
        """
        self.port = port
        self.settings = settings
        self.db_manager = db_manager

        # Set start date to today at midnight UTC if not provided
        if start_date is None:
            now = datetime.now(timezone.utc)
            self.start_date = datetime(now.year, now.month, now.day, 0, 0, 0)
        else:
            self.start_date = start_date

        # Cap days at 7
        self.days = min(days, 7)

        # Calculate total simulation time in seconds
        self.total_duration = self.days * 24 * 3600

        # Current simulation datetime
        self.current_datetime = self.start_date

        # Track boat-charger assignments
        self.boat_charger_map = {}  # {boat_name: charger_name}

        # Trip schedule times: (start_hour, slot_number)
        self.trip_schedule = [
            (9, 0),  # 9:00 AM, slot 0
            (14, 1),  # 2:00 PM, slot 1
        ]

        # Initialize trip manager
        print(f"\nLoading trips from {trips_directory}...")
        self.trip_manager = TripManager(trips_directory)

        # Track active trips: {boat_name: (trip, start_datetime, elapsed_seconds)}
        self.active_trips: Dict[str, tuple[Trip, datetime, float]] = {}

        # Track delayed trips (boats waiting for sufficient SOC): {boat_name: trip}
        self.delayed_trips: Dict[str, Trip] = {}

        # Track last date we assigned trips (to assign at midnight)
        self.last_assignment_date: Optional[str] = None

        # Initialize weather client and fetch forecast if PV systems present
        self.weather_client = None
        self.weather_forecast = {}  # {timestamp_str: {metric: value}}
        self.forecast_loaded = False

        if self.port.pv_systems:
            print(
                f"\nInitializing weather data for {len(self.port.pv_systems)} PV system(s)..."
            )
            self.weather_client = OpenMeteoClient(self.port.lat, self.port.lon)
            self._load_weather_forecast()

    def run(self):
        """Run the simulation based on the configured mode."""
        print("\n" + "=" * 60)
        print("Starting Simulation")
        print("=" * 60)
        print(f"Port: {self.port.name}")
        print(f"Start: {self.start_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Duration: {self.days} day(s)")
        print(f"Timestep: {self.settings.timestep}s")
        print(f"Mode: {self.settings.mode.value}")
        print(f"Boats: {len(self.port.boats)}")
        print(f"Chargers: {len(self.port.chargers)}")
        print("=" * 60 + "\n")

        if self.settings.mode == SimulationMode.BATCH:
            self._run_batch()
        else:
            self._run_realtime()

    def _run_batch(self):
        """Run simulation in batch mode (all timesteps at once)."""
        timestep_count = int(self.total_duration / self.settings.timestep)

        print(f"Running {timestep_count} timesteps...\n")

        for step in range(timestep_count):
            self._simulate_timestep()

            # Progress indicator every 100 steps or at midnight
            if step % 100 == 0 or self.current_datetime.hour == 0:
                print(
                    f"[{step}/{timestep_count}] {self.current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )

            # Advance time
            self.current_datetime += timedelta(seconds=self.settings.timestep)

        print(f"\n✓ Simulation completed: {timestep_count} timesteps")

    def _run_realtime(self):
        """Run simulation in real-time mode."""
        print("Running in real-time mode...\n")

        timestep_count = int(self.total_duration / self.settings.timestep)
        start_real_time = time.time()

        for step in range(timestep_count):
            step_start = time.time()

            self._simulate_timestep()

            # Print status
            if step % 10 == 0:
                print(
                    f"[{step}/{timestep_count}] {self.current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )

            # Advance time
            self.current_datetime += timedelta(seconds=self.settings.timestep)

            # Wait for the timestep duration
            elapsed = time.time() - step_start
            sleep_time = self.settings.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_real_time = time.time() - start_real_time
        print(
            f"\n✓ Simulation completed: {timestep_count} timesteps in {total_real_time:.1f}s"
        )

    def _simulate_timestep(self):
        """Simulate a single timestep."""
        # 0. Assign daily trips at midnight (00:00)
        self._assign_daily_trips_if_midnight()

        # 1. Update PV production based on weather
        self._update_pv_production()

        # 2. Check and handle trip schedules
        self._handle_trips()

        # 3. Check which boats need charging and assign to chargers
        self._assign_boats_to_chargers()

        # 4. Update BESS (after charger assignment to see correct load)
        self._update_bess()

        # 5. Update charging for boats
        self._update_charging()

        # 6. Save measurements to database
        self._save_measurements()

    def _assign_daily_trips_if_midnight(self):
        """Assign trips for all boats at midnight (00:00)."""
        current_date_str = self.current_datetime.strftime("%Y-%m-%d")

        # Check if it's midnight and we haven't assigned for this date yet
        if self.current_datetime.hour == 0 and self.current_datetime.minute == 0:
            if self.last_assignment_date != current_date_str:
                # Clear any delayed trips from previous day
                if self.delayed_trips:
                    print(f"\n  🗑️  Clearing {len(self.delayed_trips)} delayed trip(s) from previous day")
                    self.delayed_trips.clear()
                
                print(f"\n  📅 Assigning trips for {current_date_str}")
                weekday_name = self.current_datetime.strftime("%A")
                print(f"     Day: {weekday_name}")

                for boat in self.port.boats:
                    trips = self.trip_manager.assign_daily_trips(
                        boat.name, self.current_datetime
                    )
                    if trips:
                        trip_names = [t.route_name for t in trips]
                        print(f"     {boat.name}: {len(trips)} trip(s) - {trip_names}")
                    else:
                        print(f"     {boat.name}: No trips (rest day)")

                self.last_assignment_date = current_date_str
                print()

    def _handle_trips(self):
        """Handle boat trips based on schedule."""
        current_hour = self.current_datetime.hour
        current_minute = self.current_datetime.minute

        for boat in self.port.boats:
            boat_name = boat.name

            # Check if boat is currently on a trip
            if boat_name in self.active_trips:
                trip, start_time, elapsed = self.active_trips[boat_name]

                # Update elapsed time
                elapsed += self.settings.timestep

                # Check if trip is complete
                if elapsed >= trip.duration:
                    # Trip completed, return to port
                    boat.state = BoatState.IDLE
                    print(
                        f"  ← {boat.name} returned from {trip.route_name} at {self.current_datetime.strftime('%H:%M')}, SOC={boat.soc:.1%}"
                    )
                    del self.active_trips[boat_name]
                else:
                    # Still on trip, discharge battery based on current speed from CSV
                    self._discharge_boat_on_trip(boat, trip, elapsed)
                    self.active_trips[boat_name] = (trip, start_time, elapsed)
                continue

            # Check for delayed trips (boats waiting for sufficient charge)
            if boat_name in self.delayed_trips:
                trip = self.delayed_trips[boat_name]
                
                # Don't start trips after 6 PM (18:00)
                if current_hour >= 18:
                    # Cancel delayed trip if it's past 6 PM
                    if current_hour == 18 and current_minute < (self.settings.timestep / 60):
                        print(
                            f"  ❌ {boat.name} cancelled DELAYED {trip.route_name} - too late (after 6 PM)"
                        )
                        del self.delayed_trips[boat_name]
                    continue
                
                estimated_energy = trip.estimate_energy_required(boat.k)
                required_soc = estimated_energy / boat.battery_capacity

                # Check if boat now has enough charge
                if boat.soc >= required_soc:
                    # Disconnect from charger if connected
                    if boat_name in self.boat_charger_map:
                        charger_name = self.boat_charger_map[boat_name]
                        charger = next(
                            c for c in self.port.chargers if c.name == charger_name
                        )
                        charger.state = ChargerState.IDLE
                        charger.power = 0.0
                        charger.connected_boat = None
                        del self.boat_charger_map[boat_name]

                    # Start delayed trip
                    self.active_trips[boat_name] = (
                        trip,
                        self.current_datetime,
                        0.0,
                    )
                    boat.state = BoatState.SAILING
                    print(
                        f"  → {boat.name} starting DELAYED {trip.route_name} at {self.current_datetime.strftime('%H:%M')}, "
                        f"SOC={boat.soc:.1%} (needed {required_soc:.1%})"
                    )
                    del self.delayed_trips[boat_name]
                continue

            # Check if it's time to start a new trip
            for start_hour, slot in self.trip_schedule:
                # Start trip at the scheduled hour (check within the timestep window)
                if current_hour == start_hour and current_minute < (
                    self.settings.timestep / 60
                ):
                    # Get assigned trip for this slot
                    trip = self.trip_manager.get_trip_for_slot(
                        boat_name, self.current_datetime, slot
                    )

                    if trip is None:
                        continue  # No trip assigned for this slot

                    # Estimate energy required for trip
                    estimated_energy = trip.estimate_energy_required(boat.k)
                    required_soc = estimated_energy / boat.battery_capacity

                    # Check if boat has enough charge
                    if boat.soc >= required_soc:
                        # Disconnect from charger if connected
                        if boat_name in self.boat_charger_map:
                            charger_name = self.boat_charger_map[boat_name]
                            charger = next(
                                c for c in self.port.chargers if c.name == charger_name
                            )
                            charger.state = ChargerState.IDLE
                            charger.power = 0.0
                            charger.connected_boat = None
                            del self.boat_charger_map[boat_name]

                        # Start trip
                        self.active_trips[boat_name] = (
                            trip,
                            self.current_datetime,
                            0.0,
                        )
                        boat.state = BoatState.SAILING
                        print(
                            f"  → {boat.name} starting {trip.route_name} at {self.current_datetime.strftime('%H:%M')}, "
                            f"SOC={boat.soc:.1%} (need {required_soc:.1%})"
                        )
                        break
                    else:
                        # Not enough charge - delay the trip
                        self.delayed_trips[boat_name] = trip
                        print(
                            f"  ⏸️  {boat.name} delaying {trip.route_name} - insufficient charge "
                            f"(has {boat.soc:.1%}, needs {required_soc:.1%}). Will depart when ready."
                        )
                        break

    def _discharge_boat_on_trip(self, boat, trip: Trip, elapsed_seconds: float):
        """Discharge boat battery during trip based on CSV speed data."""
        # Get current point in the trip
        point = trip.get_point_at_elapsed_time(elapsed_seconds)

        if point is None:
            return

        # Calculate power consumption based on actual speed from CSV
        speed_knots = point.speed
        power_kw = boat.k * (speed_knots**3)

        # Energy consumed in this timestep (kWh)
        energy_consumed = (power_kw * self.settings.timestep) / 3600

        # Update SOC
        soc_decrease = energy_consumed / boat.battery_capacity
        boat.soc = max(0, boat.soc - soc_decrease)

    def _assign_boats_to_chargers(self):
        """Assign boats that need charging to available chargers."""
        # Get boats that need charging (not sailing, not fully charged)
        boats_needing_charge = [
            b
            for b in self.port.boats
            if b.state != BoatState.SAILING
            and b.soc < 0.99
            and b.name not in self.boat_charger_map
        ]

        # Sort by SOC (prioritize low SOC boats)
        boats_needing_charge.sort(key=lambda b: b.soc)

        # Get available chargers
        available_chargers = [
            c for c in self.port.chargers if c.state == ChargerState.IDLE
        ]

        # Calculate available power at port (including PV production and BESS discharge)
        used_power = sum(
            c.power for c in self.port.chargers if c.state == ChargerState.CHARGING
        )
        pv_production = sum(pv.current_production for pv in self.port.pv_systems)

        # BESS contribution (negative power means discharging, adds to available power)
        bess_discharge = sum(
            -bess.current_power
            for bess in self.port.bess_systems
            if bess.current_power < 0
        )
        
        # Calculate potential BESS discharge capacity (what BESS COULD provide)
        potential_bess_discharge = sum(
            bess.get_max_discharge_power_available(self.settings.timestep)
            for bess in self.port.bess_systems
        )

        available_power = (
            self.port.contracted_power + pv_production + bess_discharge - used_power
        )

        # Assign boats to chargers
        for boat in boats_needing_charge:
            if not available_chargers:
                break

            charger = available_chargers.pop(0)

            # Check if we have enough power to run this charger at max
            # If not enough immediate power, check if BESS can cover the deficit
            power_needed = charger.max_power
            
            if available_power >= power_needed:
                # Sufficient power available directly
                can_assign = True
            else:
                # Not enough power - check if BESS can cover the deficit
                deficit = power_needed - available_power
                can_assign = potential_bess_discharge >= deficit
            
            if can_assign:
                # Assign boat to charger
                self.boat_charger_map[boat.name] = charger.name
                boat.state = BoatState.CHARGING
                charger.state = ChargerState.CHARGING
                charger.power = charger.max_power
                charger.connected_boat = boat.name
                available_power -= charger.max_power
                
                # Reduce potential BESS discharge if we're counting on it
                if available_power < 0:
                    potential_bess_discharge -= abs(available_power)
                    available_power = 0

                # Log charging start
                if self.current_datetime.minute % 15 == 0:
                    print(
                        f"  ⚡ {boat.name} started charging at {charger.name}, SOC={boat.soc:.1%}"
                    )

    def _update_charging(self):
        """Update battery charge for boats that are charging."""
        for boat_name, charger_name in list(self.boat_charger_map.items()):
            boat = next(b for b in self.port.boats if b.name == boat_name)
            charger = next(c for c in self.port.chargers if c.name == charger_name)

            # Calculate energy delivered to battery in this timestep
            effective_power = charger.effective_power  # kW
            energy_delivered = (effective_power * self.settings.timestep) / 3600  # kWh

            # Update boat SOC
            soc_increase = energy_delivered / boat.battery_capacity
            boat.soc = min(1.0, boat.soc + soc_increase)

            # If boat is fully charged, disconnect
            if boat.soc >= 0.99:
                print(
                    f"  ✓ {boat.name} fully charged at {self.current_datetime.strftime('%H:%M')}"
                )
                boat.state = BoatState.IDLE
                charger.state = ChargerState.IDLE
                charger.power = 0.0
                charger.connected_boat = None
                del self.boat_charger_map[boat_name]

    def _load_weather_forecast(self):
        """Load weather forecast from Open-Meteo and save to database."""
        if not self.weather_client:
            return

        print("  Fetching weather forecast from Open-Meteo...")
        forecast_data = self.weather_client.fetch_forecast(self.start_date, self.days)

        if not forecast_data or "timestamps" not in forecast_data:
            print("  ⚠️  Failed to fetch weather forecast")
            return

        timestamps = forecast_data["timestamps"]
        print(f"  ✓ Received {len(timestamps)} hours of forecast data")

        # Save to database
        forecasts = []
        for i, ts in enumerate(timestamps):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            # Save each metric
            for metric, values in forecast_data.items():
                if metric == "timestamps":
                    continue

                if i < len(values) and values[i] is not None:
                    forecasts.append((ts_str, "openmeteo", metric, float(values[i])))

            # Store in memory for quick access
            if i < len(values):
                self.weather_forecast[ts_str] = {}
                for metric, values in forecast_data.items():
                    if (
                        metric != "timestamps"
                        and i < len(values)
                        and values[i] is not None
                    ):
                        self.weather_forecast[ts_str][metric] = float(values[i])

        # Save to database
        if forecasts:
            self.db_manager.save_forecasts_batch(forecasts)
            print(f"  ✓ Saved {len(forecasts)} forecast values to database")

        self.forecast_loaded = True

    def _get_weather_conditions(self, timestamp: datetime) -> Dict:
        """
        Get weather conditions for a specific timestamp.

        Args:
            timestamp: Datetime to get conditions for

        Returns:
            Dictionary with weather metrics
        """
        # Round to nearest hour for lookup
        rounded = timestamp.replace(minute=0, second=0, microsecond=0)
        ts_str = rounded.strftime("%Y-%m-%d %H:%M:%S")

        if ts_str in self.weather_forecast:
            return self.weather_forecast[ts_str]

        # Return default values if not found
        return {
            "ghi": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
            "temperature": 20.0,
        }

    def _update_pv_production(self):
        """Update PV production for all PV systems."""
        if not self.port.pv_systems:
            return

        # Get current weather conditions
        conditions = self._get_weather_conditions(self.current_datetime)

        # Update each PV system
        for pv in self.port.pv_systems:
            pv.calculate_production(
                ghi=conditions.get("ghi", 0.0),
                dni=conditions.get("dni", 0.0),
                dhi=conditions.get("dhi", 0.0),
                temperature=conditions.get("temperature", 20.0),
                timestamp=self.current_datetime,
            )

    def _update_bess(self):
        """
        Update BESS charge/discharge based on default control strategy.

        Default strategy:
        - Charge from surplus PV production (when PV > charger load)
        - Discharge when needed (when contracted + PV < charger load)
        - Respect SOC limits (10% min, 90% max by default)
        """
        if not self.port.bess_systems:
            return

        # Calculate current power flows
        pv_production = sum(pv.current_production for pv in self.port.pv_systems)
        charger_load = sum(
            c.power for c in self.port.chargers if c.state == ChargerState.CHARGING
        )

        # PV surplus = excess PV not used by chargers
        # Positive means excess PV available for charging BESS
        pv_surplus = pv_production - charger_load

        # Power deficit = chargers need more than grid + PV can provide
        # Positive means we need to discharge BESS
        power_deficit = charger_load - (self.port.contracted_power + pv_production)

        for bess in self.port.bess_systems:
            if pv_surplus > 0:
                # Excess PV available - charge BESS with the PV surplus ONLY
                max_charge = bess.get_max_charge_power_available(self.settings.timestep)
                charge_power = min(pv_surplus, max_charge)

                if charge_power > 0.1:  # Only charge if meaningful power available
                    bess.charge(charge_power, self.settings.timestep)
                    pv_surplus -= charge_power
                else:
                    bess.idle()

            elif power_deficit > 0:
                # Power deficit - discharge BESS to cover the shortfall
                max_discharge = bess.get_max_discharge_power_available(
                    self.settings.timestep
                )
                discharge_power = min(power_deficit, max_discharge)

                if discharge_power > 0.1:  # Only discharge if meaningful
                    bess.discharge(discharge_power, self.settings.timestep)
                    power_deficit -= discharge_power
                else:
                    bess.idle()
            else:
                # No PV surplus and no deficit - idle
                bess.idle()

    def _save_measurements(self):
        """Save current state to database."""
        measurements = []

        # Convert current datetime to ISO format string (UTC)
        timestamp_str = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate PV production
        total_pv_production = sum(pv.current_production for pv in self.port.pv_systems)

        # Calculate BESS contribution
        total_bess_power = sum(bess.current_power for bess in self.port.bess_systems)
        # Positive = charging (consuming power), Negative = discharging (providing power)
        bess_discharge = -total_bess_power if total_bess_power < 0 else 0
        bess_charge = total_bess_power if total_bess_power > 0 else 0

        # Calculate port metrics
        total_power_used = sum(
            c.power for c in self.port.chargers if c.state == ChargerState.CHARGING
        )

        # Available power = contracted power + PV production + BESS discharge - power used - BESS charge
        available_power = (
            self.port.contracted_power
            + total_pv_production
            + bess_discharge
            - total_power_used
            - bess_charge
        )

        # Port measurements
        measurements.append(
            (timestamp_str, "port", "total_power_used", total_power_used)
        )
        measurements.append(
            (timestamp_str, "port", "pv_production", total_pv_production)
        )
        measurements.append((timestamp_str, "port", "bess_discharge", bess_discharge))
        measurements.append((timestamp_str, "port", "bess_charge", bess_charge))
        measurements.append((timestamp_str, "port", "available_power", available_power))
        measurements.append(
            (timestamp_str, "port", "contracted_power", self.port.contracted_power)
        )

        # Boat measurements
        for boat in self.port.boats:
            measurements.append((timestamp_str, boat.name, "soc", boat.soc * 100))
            measurements.append(
                (
                    timestamp_str,
                    boat.name,
                    "state",
                    float(boat.state.value == "sailing"),
                )
            )

            # Calculate current motor power (only when sailing)
            if boat.state == BoatState.SAILING and boat.name in self.active_trips:
                trip, _, elapsed = self.active_trips[boat.name]
                point = trip.get_point_at_elapsed_time(elapsed)
                if point:
                    motor_power = boat.k * (point.speed**3)
                else:
                    motor_power = 0.0
            else:
                # Motor is off when charging or idle
                motor_power = 0.0

            measurements.append((timestamp_str, boat.name, "power", motor_power))

        # Charger measurements
        for charger in self.port.chargers:
            measurements.append((timestamp_str, charger.name, "power", charger.power))
            measurements.append(
                (
                    timestamp_str,
                    charger.name,
                    "state",
                    float(charger.state.value == "charging"),
                )
            )

        # PV system measurements
        for pv in self.port.pv_systems:
            measurements.append(
                (timestamp_str, pv.name, "production", pv.current_production)
            )

        # BESS measurements
        for bess in self.port.bess_systems:
            measurements.append(
                (timestamp_str, bess.name, "soc", bess.current_soc * 100)
            )
            measurements.append((timestamp_str, bess.name, "power", bess.current_power))
            measurements.append(
                (timestamp_str, bess.name, "energy_stored", bess.get_energy_stored())
            )

        # Save to database
        self.db_manager.save_measurements_batch(measurements)
