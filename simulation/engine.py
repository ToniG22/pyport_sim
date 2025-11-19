"""Core simulation engine for the electric port simulator."""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

from models import Port, BoatState, ChargerState, Trip
from database import DatabaseManager
from config import Settings, SimulationMode
from simulation.trip_manager import TripManager
from weather import OpenMeteoClient
from forecasting import PortForecaster
from optimization import PortOptimizer


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

        # Track last date we fetched weather forecast
        self.last_weather_fetch_date: Optional[str] = None

        # Track last date we generated energy forecast
        self.last_energy_forecast_date: Optional[str] = None

        # Initialize forecaster
        self.forecaster = PortForecaster(port, db_manager, settings.timestep)

        # Initialize optimizer (if enabled)
        self.optimizer = None
        self.use_optimizer = settings.use_optimizer
        if self.use_optimizer:
            self.optimizer = PortOptimizer(port, db_manager, settings.timestep)
            print(f"\n🔧 Optimizer enabled (SCIP)")

        # Store latest forecasts for optimizer
        self.latest_energy_forecasts = []

        # Track boats with energy shortfalls (for priority charging)
        self.boats_with_shortfalls = set()  # {boat_name}

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
            # Mark initial fetch date
            self.last_weather_fetch_date = self.start_date.strftime("%Y-%m-%d")

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
        # Check if it's midnight (00:00)
        is_midnight = (
            self.current_datetime.hour == 0 and self.current_datetime.minute == 0
        )

        if is_midnight:
            # 0. Fetch weather forecast daily at midnight (00:00)
            self._fetch_weather()
            # 1. Assign daily trips at midnight (00:00):
            self._assign_daily_trips()
            # 2. Generate energy forecast at midnight (00:00):
            self._generate_energy_forecast()

        # 3. Update PV production based on weather
        self._update_pv_production()

        # 4. Check and handle trip schedules
        self._handle_trips()

        # 5. Check which boats need charging and assign to chargers
        self._assign_boats_to_chargers()

        # 6. Update BESS (after charger assignment to see correct load)
        self._update_bess()

        # 7. Update charging for boats
        self._update_charging()

        # 8. Save measurements to database
        self._save_measurements()

    def _fetch_weather(self):
        """Fetch weather forecast daily at midnight (00:00)."""
        if not self.weather_client:
            return

        current_date_str = self.current_datetime.strftime("%Y-%m-%d")

        # Check if we haven't fetched for this date yet
        if self.last_weather_fetch_date != current_date_str:
            print(f"\n  🌤️  Fetching weather forecast for {current_date_str}")

            # Calculate remaining days in simulation from current date
            days_remaining = (
                self.start_date + timedelta(days=self.days) - self.current_datetime
            ).days
            days_to_fetch = min(days_remaining, 7)  # OpenMeteo provides up to 7 days

            if days_to_fetch > 0:
                # Fetch forecast from current date
                self._load_weather_forecast(
                    start_from=self.current_datetime, days=days_to_fetch
                )
                self.last_weather_fetch_date = current_date_str
                print(f"     ✓ Weather data updated for next {days_to_fetch} day(s)")
            print()

    def _generate_energy_forecast(self):
        """Generate energy consumption/production forecast at midnight (00:00)."""
        current_date_str = self.current_datetime.strftime("%Y-%m-%d")

        # Check if we haven't forecasted for this date yet
        if self.last_energy_forecast_date != current_date_str:
            print(f"  📊 Generating energy forecast for {current_date_str}")

            # Get trip assignments for today
            trip_assignments = {}
            for boat in self.port.boats:
                trips = self.trip_manager.get_trips_for_date(
                    boat.name, self.current_datetime
                )
                trip_assignments[boat.name] = trips

            # Generate 24-hour forecast
            forecasts = self.forecaster.generate_daily_forecast(
                self.current_datetime, trip_assignments
            )

            # Store forecasts for optimizer
            self.latest_energy_forecasts = forecasts

            # Save to database
            self.forecaster.save_forecasts_to_db(forecasts, forecast_type="port_energy")

            # Print summary
            self.forecaster.print_forecast_summary(forecasts)

            # Run optimization if enabled
            if self.use_optimizer and self.optimizer:
                self._run_optimization(trip_assignments)

            self.last_energy_forecast_date = current_date_str
            print()

    def _run_optimization(self, trip_assignments: Dict[str, List]):
        """
        Run optimization to create optimal schedules for chargers and BESS.

        Args:
            trip_assignments: Trip assignments per boat
        """
        print(f"  🎯 Running optimization...")

        # Run optimization
        result = self.optimizer.optimize_daily_schedule(
            self.current_datetime, self.latest_energy_forecasts, trip_assignments
        )

        # Handle energy shortfalls gracefully
        if result.energy_shortfalls:
            self._handle_energy_shortfalls(result, trip_assignments)

        # Save schedules to database
        self.optimizer.save_schedules_to_db(result)

        print(f"     ✓ Schedules saved to database")

    def _trigger_reoptimization(self):
        """Trigger re-optimization when boat state changes (arrive/depart)."""
        current_date_str = self.current_datetime.strftime("%Y-%m-%d")

        # Only re-optimize during daytime (after 6:00, before 20:00)
        if self.current_datetime.hour < 6 or self.current_datetime.hour >= 20:
            return

        print(f"  🔄 Re-optimizing from {self.current_datetime.strftime('%H:%M')}...")

        # Clear existing schedules for remaining day only (from current time onwards)
        # This preserves schedules for the early part of the day
        current_time_str = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        for charger in self.port.chargers:
            # Clear only schedules from current time onwards
            self.db_manager.clear_schedules(
                source=charger.name, from_time=current_time_str
            )
        for bess in self.port.bess_systems:
            self.db_manager.clear_schedules(
                source=bess.name, from_time=current_time_str
            )

        # Get trip assignments for today (remaining trips)
        trip_assignments = {}
        for boat in self.port.boats:
            trips = self.trip_manager.get_trips_for_date(
                boat.name, self.current_datetime
            )
            trip_assignments[boat.name] = trips

        # Generate forecast from current time to end of day
        remaining_forecasts = [
            f
            for f in self.latest_energy_forecasts
            if f.timestamp >= self.current_datetime
        ]

        if remaining_forecasts:
            # Run optimization with updated boat SOCs
            result = self.optimizer.optimize_daily_schedule(
                self.current_datetime, remaining_forecasts, trip_assignments
            )

            # Handle energy shortfalls gracefully
            if result.energy_shortfalls:
                self._handle_energy_shortfalls(result, trip_assignments)

            # Save new schedules
            self.optimizer.save_schedules_to_db(result)
            print(
                f"     ✓ Updated schedules from {self.current_datetime.strftime('%H:%M')} onwards"
            )

    def _handle_energy_shortfalls(self, result, trip_assignments: Dict[str, List]):
        """
        Handle cases where optimizer cannot meet energy requirements (graceful degradation).

        When energy shortfalls are detected, this function:
        1. Logs warnings about which boats are affected
        2. Attempts to maximize charging for affected boats
        3. Updates schedules to use maximum available power

        Args:
            result: OptimizationResult with energy_shortfalls
            trip_assignments: Trip assignments per boat
        """
        if not result.energy_shortfalls:
            return

        print(
            f"     ⚠️  Handling energy shortfalls for {len(result.energy_shortfalls)} boat(s)..."
        )

        # Update tracking of boats with shortfalls
        self.boats_with_shortfalls = set(result.energy_shortfalls.keys())

        # For each boat with shortfall, try to maximize charging
        for boat_name, shortfall_kwh in result.energy_shortfalls.items():
            boat = next(b for b in self.port.boats if b.name == boat_name)
            shortfall_pct = (shortfall_kwh / boat.battery_capacity) * 100

            print(
                f"       {boat_name}: {shortfall_kwh:.2f} kWh shortfall ({shortfall_pct:.1f}% of battery)"
            )

            # Calculate remaining trips and energy needed
            trips = trip_assignments.get(boat_name, [])
            if trips:
                total_trip_energy = sum(
                    trip.estimate_energy_required(boat.k) for trip in trips
                )
                current_energy = boat.soc * boat.battery_capacity
                energy_needed = total_trip_energy + (
                    boat.battery_capacity - current_energy
                )

                print(f"         Current SOC: {boat.soc:.1%}")
                print(f"         Energy needed: {energy_needed:.2f} kWh")
                print(f"         Energy available: {current_energy:.2f} kWh")
                print(f"         Shortfall: {shortfall_kwh:.2f} kWh")

                # Override schedules to use maximum power for this boat when available
                # This maximizes charging priority for boats with shortfalls
                self._override_schedules_for_shortfall_boat(boat_name, result)

                if trips:
                    print(
                        f"         ⚠️  Warning: {len(trips)} trip(s) may be delayed or cancelled"
                    )
                    print(
                        f"         Action: Maximizing charging priority for {boat_name}"
                    )

        # Note: The charger assignment logic will prioritize boats with lower SOC,
        # and boats with shortfalls are now marked for maximum power charging.

    def _assign_daily_trips(self):
        """Assign trips for all boats at midnight (00:00)."""
        current_date_str = self.current_datetime.strftime("%Y-%m-%d")

        # Check if we haven't assigned for this date yet
        if self.last_assignment_date != current_date_str:
            # Clear any delayed trips from previous day
            if self.delayed_trips:
                print(
                    f"  🗑️  Clearing {len(self.delayed_trips)} delayed trip(s) from previous day"
                )
                self.delayed_trips.clear()

            print(f"  📅 Assigning trips for {current_date_str}")
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

                    # Trigger re-optimization if optimizer is enabled
                    if self.use_optimizer and self.optimizer:
                        self._trigger_reoptimization()
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
                    if current_hour == 18 and current_minute < (
                        self.settings.timestep / 60
                    ):
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

                    # Trigger re-optimization if optimizer is enabled
                    if self.use_optimizer and self.optimizer:
                        self._trigger_reoptimization()
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

                        # Trigger re-optimization if optimizer is enabled
                        if self.use_optimizer and self.optimizer:
                            self._trigger_reoptimization()

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
        # Check if we're using optimizer schedules
        if self.use_optimizer:
            self._assign_boats_to_chargers_with_schedule()
        else:
            self._assign_boats_to_chargers_default()

    def _assign_boats_to_chargers_with_schedule(self):
        """Assign boats to chargers using optimized schedules from database."""
        timestamp_str = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Get charger power setpoints from schedule
        for charger in self.port.chargers:
            schedule = self.db_manager.get_schedules(
                source=charger.name,
                metric="power_setpoint",
                start_time=timestamp_str,
                end_time=timestamp_str,
            )

            if schedule:
                # Use scheduled power
                scheduled_power = schedule[0]["value"]

                if scheduled_power > 0.1 and charger.state == ChargerState.IDLE:
                    # Find a boat that needs charging
                    # Prioritize boats with delayed trips, then boats with energy shortfalls
                    boats_needing_charge = [
                        b
                        for b in self.port.boats
                        if (
                            b.state != BoatState.SAILING
                            and b.soc < 0.99
                            and b.name not in self.boat_charger_map
                        )
                    ]

                    # Sort: boats with delayed trips first, then boats with shortfalls, then by SOC (lowest first)
                    boats_needing_charge.sort(
                        key=lambda b: (
                            0 if b.name in self.delayed_trips else 1,
                            0 if b.name in self.boats_with_shortfalls else 1,
                            b.soc,
                        )
                    )

                    for boat in boats_needing_charge:
                        # For boats with delayed trips or shortfalls, use maximum power regardless of schedule
                        if (
                            boat.name in self.delayed_trips
                            or boat.name in self.boats_with_shortfalls
                        ):
                            power_to_use = charger.max_power
                        else:
                            power_to_use = min(scheduled_power, charger.max_power)

                        # Assign charger to boat
                        self.boat_charger_map[boat.name] = charger.name
                        boat.state = BoatState.CHARGING
                        charger.state = ChargerState.CHARGING
                        charger.power = power_to_use
                        charger.connected_boat = boat.name

                        if self.current_datetime.minute % 15 == 0:
                            if boat.name in self.delayed_trips:
                                priority_note = " (priority - delayed trip)"
                            elif boat.name in self.boats_with_shortfalls:
                                priority_note = " (priority - shortfall)"
                            else:
                                priority_note = " (scheduled)"
                            print(
                                f"  ⚡ {boat.name} started charging at {charger.name}{priority_note}, SOC={boat.soc:.1%}"
                            )
                        break
                elif scheduled_power < 0.1 and charger.state == ChargerState.CHARGING:
                    # Schedule says stop charging, but check if boat has delayed trip or shortfall
                    if charger.connected_boat in self.boat_charger_map:
                        boat_name = charger.connected_boat
                        boat = next(b for b in self.port.boats if b.name == boat_name)

                        # Don't stop charging if boat has delayed trip and still needs charge
                        if boat_name in self.delayed_trips and boat.soc < 0.99:
                            # Continue charging at max power for delayed trip
                            charger.power = charger.max_power
                            continue

                        # Don't stop charging if boat has shortfall and still needs charge
                        if boat_name in self.boats_with_shortfalls and boat.soc < 0.99:
                            # Continue charging at max power
                            charger.power = charger.max_power
                            continue

                        # Otherwise, stop charging as scheduled
                        boat.state = BoatState.IDLE
                        charger.state = ChargerState.IDLE
                        charger.power = 0.0
                        charger.connected_boat = None
                        del self.boat_charger_map[boat_name]
            else:
                # No schedule - check if any boat with delayed trip needs charging
                if charger.state == ChargerState.IDLE:
                    boats_with_delayed_trips = [
                        b
                        for b in self.port.boats
                        if (
                            b.name in self.delayed_trips
                            and b.state != BoatState.SAILING
                            and b.soc < 0.99
                            and b.name not in self.boat_charger_map
                        )
                    ]

                    if boats_with_delayed_trips:
                        # Prioritize boats with delayed trips - assign charger
                        boat = min(boats_with_delayed_trips, key=lambda b: b.soc)
                        self.boat_charger_map[boat.name] = charger.name
                        boat.state = BoatState.CHARGING
                        charger.state = ChargerState.CHARGING
                        charger.power = charger.max_power
                        charger.connected_boat = boat.name

                        if self.current_datetime.minute % 15 == 0:
                            print(
                                f"  ⚡ {boat.name} started charging at {charger.name} (priority - delayed trip, no schedule), SOC={boat.soc:.1%}"
                            )

    def _override_schedules_for_shortfall_boat(self, boat_name: str, result):
        """
        Override schedules to maximize charging for a boat with energy shortfall.

        This updates the schedules in the database to use maximum power
        when the boat is available (not sailing).

        Args:
            boat_name: Name of boat with shortfall
            result: OptimizationResult with schedules
        """
        # Find which charger is assigned to this boat (if any)
        assigned_charger = None
        if boat_name in self.boat_charger_map:
            charger_name = self.boat_charger_map[boat_name]
            assigned_charger = next(
                c for c in self.port.chargers if c.name == charger_name
            )

        # Update schedules to use maximum power for this boat's charger
        # when boat is available
        if assigned_charger and assigned_charger.name in result.charger_schedules:
            updated_schedules = []
            charger_schedule = result.charger_schedules[assigned_charger.name]

            for timestamp, power in charger_schedule:
                # Check if boat is available at this timestamp
                # Find corresponding forecast
                forecast = next(
                    (
                        f
                        for f in self.latest_energy_forecasts
                        if f.timestamp == timestamp
                    ),
                    None,
                )

                if (
                    forecast
                    and forecast.boat_states.get(boat_name, BoatState.IDLE)
                    != BoatState.SAILING
                ):
                    # Boat is available - use maximum power
                    updated_schedules.append((timestamp, assigned_charger.max_power))
                else:
                    # Boat is sailing or forecast not found - keep original schedule
                    updated_schedules.append((timestamp, power))

            # Update the schedule in result
            result.charger_schedules[assigned_charger.name] = updated_schedules

    def _assign_boats_to_chargers_default(self):
        """Assign boats to chargers using default rule-based logic."""
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

    def _load_weather_forecast(
        self, start_from: Optional[datetime] = None, days: Optional[int] = None
    ):
        """Load weather forecast from Open-Meteo and save to database.

        Args:
            start_from: Starting date for forecast (default: simulation start_date)
            days: Number of days to fetch (default: simulation days, max 7)
        """
        if not self.weather_client:
            return

        # Use provided values or defaults
        fetch_start = start_from if start_from else self.start_date
        fetch_days = min(days if days else self.days, 7)  # Cap at 7 days

        print("  Fetching weather forecast from Open-Meteo...")
        forecast_data = self.weather_client.fetch_forecast(fetch_start, fetch_days)

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
            if i < len(timestamps):
                self.weather_forecast[ts_str] = {}
                for metric, metric_values in forecast_data.items():
                    if (
                        metric != "timestamps"
                        and i < len(metric_values)
                        and metric_values[i] is not None
                    ):
                        self.weather_forecast[ts_str][metric] = float(metric_values[i])

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
        """Update BESS charge/discharge."""
        if not self.port.bess_systems:
            return

        # Check if we're using optimizer schedules
        if self.use_optimizer:
            self._update_bess_with_schedule()
        else:
            self._update_bess_default()

    def _update_bess_with_schedule(self):
        """Update BESS using optimized schedules from database."""
        timestamp_str = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        for bess in self.port.bess_systems:
            schedule = self.db_manager.get_schedules(
                source=bess.name,
                metric="power_setpoint",
                start_time=timestamp_str,
                end_time=timestamp_str,
            )

            if schedule:
                # Use scheduled power (positive = discharge, negative = charge)
                scheduled_power = schedule[0]["value"]

                if scheduled_power > 0.1:
                    # Discharge
                    bess.discharge(scheduled_power, self.settings.timestep)
                elif scheduled_power < -0.1:
                    # Charge
                    bess.charge(abs(scheduled_power), self.settings.timestep)
                else:
                    # Idle
                    bess.idle()
            else:
                # No schedule - fallback to default
                bess.idle()

    def _update_bess_default(self):
        """Update BESS using default control strategy."""
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
            (timestamp_str, "port", "power_active_consumption", total_power_used)
        )
        measurements.append(
            (timestamp_str, "port", "power_active_production", total_pv_production)
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

            measurements.append((timestamp_str, boat.name, "power_active", motor_power))

        # Charger measurements
        for charger in self.port.chargers:
            measurements.append(
                (timestamp_str, charger.name, "power_active", charger.power)
            )
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
                (
                    timestamp_str,
                    pv.name,
                    "power_active_production",
                    pv.current_production,
                )
            )

        # BESS measurements
        for bess in self.port.bess_systems:
            measurements.append(
                (timestamp_str, bess.name, "soc", bess.current_soc * 100)
            )
            measurements.append(
                (timestamp_str, bess.name, "power_active", bess.current_power)
            )
            measurements.append(
                (timestamp_str, bess.name, "energy_stored", bess.get_energy_stored())
            )

        # Save to database
        self.db_manager.save_measurements_batch(measurements)
