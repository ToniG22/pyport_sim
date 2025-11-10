"""Core simulation engine for the electric port simulator."""

import time
from datetime import datetime, timedelta
from typing import Optional

from models import Port, BoatState, ChargerState
from database import DatabaseManager
from config import Settings, SimulationMode


class SimulationEngine:
    """Main simulation engine for the electric port."""

    def __init__(
        self,
        port: Port,
        settings: Settings,
        db_manager: DatabaseManager,
        start_date: Optional[datetime] = None,
        days: int = 1,
    ):
        """
        Initialize the simulation engine.

        Args:
            port: Port instance (contains boats and chargers)
            settings: Simulation settings
            db_manager: Database manager
            start_date: Simulation start date (default: today at midnight UTC)
            days: Number of days to simulate (max 7)
        """
        self.port = port
        self.settings = settings
        self.db_manager = db_manager

        # Set start date to today at midnight UTC if not provided
        if start_date is None:
            now = datetime.utcnow()
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

        # Trip schedule: (start_hour, duration_hours)
        self.trip_schedule = [
            (9, 3),  # 9:00 AM, 3 hours
            (14, 3),  # 2:00 PM, 3 hours
        ]

        # Track which boats are on which trips
        self.boat_trip_status = {}  # {boat_name: (trip_start_datetime, trip_end_datetime)}

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
        # 1. Check and handle trip schedules
        self._handle_trips()

        # 2. Check which boats need charging and assign to chargers
        self._assign_boats_to_chargers()

        # 3. Update charging for boats
        self._update_charging()

        # 4. Save measurements to database
        self._save_measurements()

    def _handle_trips(self):
        """Handle boat trips based on schedule."""
        current_hour = self.current_datetime.hour
        current_minute = self.current_datetime.minute

        for boat in self.port.boats:
            boat_name = boat.name

            # Check if boat is currently on a trip
            if boat_name in self.boat_trip_status:
                trip_start, trip_end = self.boat_trip_status[boat_name]
                if self.current_datetime >= trip_end:
                    # Trip completed, return to port
                    boat.state = BoatState.IDLE
                    print(
                        f"  ← {boat.name} returned from trip at {self.current_datetime.strftime('%H:%M')}, SOC={boat.soc:.1%}"
                    )
                    del self.boat_trip_status[boat_name]
                else:
                    # Still on trip, discharge battery
                    self._discharge_boat(boat)
                continue

            # Check if it's time to start a new trip
            for start_hour, duration_hours in self.trip_schedule:
                # Start trip at the scheduled hour (check within the timestep window)
                if current_hour == start_hour and current_minute < (
                    self.settings.timestep / 60
                ):
                    # Check if boat has enough charge to make the trip (SOC > 20%)
                    if boat.soc > 0.2:
                        # Check if we haven't already started this trip
                        trip_already_started = False
                        if boat_name in self.boat_trip_status:
                            trip_start, _ = self.boat_trip_status[boat_name]
                            # If trip started within the last hour, don't start again
                            if (
                                self.current_datetime - trip_start
                            ).total_seconds() < 3600:
                                trip_already_started = True

                        if not trip_already_started:
                            # Disconnect from charger if connected
                            if boat_name in self.boat_charger_map:
                                charger_name = self.boat_charger_map[boat_name]
                                charger = next(
                                    c
                                    for c in self.port.chargers
                                    if c.name == charger_name
                                )
                                charger.state = ChargerState.IDLE
                                charger.power = 0.0
                                charger.connected_boat = None
                                del self.boat_charger_map[boat_name]

                            # Start trip
                            trip_end = self.current_datetime + timedelta(
                                hours=duration_hours
                            )
                            self.boat_trip_status[boat_name] = (
                                self.current_datetime,
                                trip_end,
                            )
                            boat.state = BoatState.SAILING
                            print(
                                f"  → {boat.name} starting trip at {self.current_datetime.strftime('%H:%M')}, SOC={boat.soc:.1%}"
                            )
                            break

    def _discharge_boat(self, boat):
        """Discharge boat battery during sailing."""
        # Calculate energy consumption for this timestep
        # Power consumption = k * speed^3
        speed = boat.range_speed  # knots
        power = boat.k * (speed**3)  # kW

        # Energy consumed in this timestep (kWh)
        energy_consumed = (power * self.settings.timestep) / 3600

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

        # Calculate available power at port
        used_power = sum(
            c.power for c in self.port.chargers if c.state == ChargerState.CHARGING
        )
        available_power = self.port.contracted_power - used_power

        # Assign boats to chargers
        for boat in boats_needing_charge:
            if not available_chargers:
                break

            charger = available_chargers.pop(0)

            # Check if we have enough power to run this charger at max
            if available_power >= charger.max_power:
                # Assign boat to charger
                self.boat_charger_map[boat.name] = charger.name
                boat.state = BoatState.CHARGING
                charger.state = ChargerState.CHARGING
                charger.power = charger.max_power
                charger.connected_boat = boat.name
                available_power -= charger.max_power

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

    def _save_measurements(self):
        """Save current state to database."""
        measurements = []
        
        # Convert current datetime to ISO format string (UTC)
        timestamp_str = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate port metrics
        total_power_used = sum(
            c.power for c in self.port.chargers if c.state == ChargerState.CHARGING
        )
        available_power = self.port.contracted_power - total_power_used

        # Port measurements
        measurements.append((timestamp_str, "port", "total_power_used", total_power_used))
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

            # Calculate current power draw
            if boat.state == BoatState.SAILING:
                power = boat.k * (boat.range_speed**3)
            elif boat.state == BoatState.CHARGING and boat.name in self.boat_charger_map:
                charger_name = self.boat_charger_map[boat.name]
                charger = next(c for c in self.port.chargers if c.name == charger_name)
                power = charger.effective_power
            else:
                power = 0.0

            measurements.append((timestamp_str, boat.name, "power", power))

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

        # Save to database
        self.db_manager.save_measurements_batch(measurements)
