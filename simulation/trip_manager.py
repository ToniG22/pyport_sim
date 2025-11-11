"""Trip management for the simulation."""

import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from models.trip import Trip


class TripManager:
    """Manages trip assignments and loading for boats."""

    def __init__(self, trips_directory: str = "assets/trips"):
        """
        Initialize the trip manager.

        Args:
            trips_directory: Directory containing trip CSV files
        """
        self.trips_directory = Path(trips_directory)
        self.available_trips: List[Trip] = []
        self._load_trips()

        # Track assigned trips per boat per day
        # {boat_name: {date_str: [trip1, trip2, ...]}}
        self.daily_assignments: Dict[str, Dict[str, List[Trip]]] = {}

    def _load_trips(self):
        """Load all available trip CSV files."""
        if not self.trips_directory.exists():
            print(f"Warning: Trips directory not found: {self.trips_directory}")
            return

        csv_files = list(self.trips_directory.glob("route_*.csv"))
        for csv_file in sorted(csv_files):
            try:
                trip = Trip(str(csv_file))
                if trip.points:  # Only add trips with valid data
                    self.available_trips.append(trip)
                    print(f"  Loaded {trip.route_name}: {len(trip.points)} points, {trip.duration/3600:.2f}h")
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")

        if self.available_trips:
            print(f"  Total trips loaded: {len(self.available_trips)}")
        else:
            print("  Warning: No trips loaded!")

    def assign_daily_trips(self, boat_name: str, current_date: datetime) -> List[Trip]:
        """
        Assign trips for a boat for the given day.

        Args:
            boat_name: Name of the boat
            current_date: Current simulation date

        Returns:
            List of trips assigned for this day
        """
        if not self.available_trips:
            return []

        date_str = current_date.strftime("%Y-%m-%d")
        weekday = current_date.weekday()  # 0=Monday, 6=Sunday

        # Check if already assigned for this day
        if boat_name in self.daily_assignments:
            if date_str in self.daily_assignments[boat_name]:
                return self.daily_assignments[boat_name][date_str]

        # Determine number of trips based on day of week
        if weekday < 5:  # Monday-Friday
            num_trips = 2
        elif weekday == 5:  # Saturday
            num_trips = 1
        else:  # Sunday
            num_trips = 0

        # Randomly select trips from available trips
        assigned_trips = []
        if num_trips > 0:
            # Allow repetition if we have fewer routes than needed trips
            if num_trips <= len(self.available_trips):
                assigned_trips = random.sample(self.available_trips, num_trips)
            else:
                assigned_trips = random.choices(self.available_trips, k=num_trips)

        # Store assignment
        if boat_name not in self.daily_assignments:
            self.daily_assignments[boat_name] = {}
        self.daily_assignments[boat_name][date_str] = assigned_trips

        return assigned_trips

    def get_trip_for_slot(
        self, boat_name: str, current_date: datetime, slot: int
    ) -> Optional[Trip]:
        """
        Get the assigned trip for a specific time slot.

        Args:
            boat_name: Name of the boat
            current_date: Current simulation date
            slot: Time slot (0 = morning 9AM, 1 = afternoon 2PM)

        Returns:
            The assigned trip, or None if no trip for this slot
        """
        date_str = current_date.strftime("%Y-%m-%d")

        if boat_name not in self.daily_assignments:
            return None

        if date_str not in self.daily_assignments[boat_name]:
            return None

        trips = self.daily_assignments[boat_name][date_str]
        if slot < len(trips):
            return trips[slot]

        return None

