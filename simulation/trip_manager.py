"""Trip management for the simulation."""

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from models.trip import Trip


class TripManager:
    """
    Manage trip loading and daily trip assignments for boats.

    Loads route CSV files once and then assigns trips per boat and day based on
    weekday rules, exposing helpers to query trips by slot or date.
    """

    def __init__(self, trips_directory: str = "assets/trips"):
        """
        Initialize the trip manager and load available trips from disk.

        Args:
            trips_directory: Directory containing trip CSV files
        """
        self.trips_directory = Path(trips_directory)
        self.available_trips: List[Trip] = []
        self._load_trips()
        self.daily_assignments: Dict[str, Dict[str, List[Trip]]] = {}

    def _load_trips(self):
        """
        Load all available trip CSV files into `available_trips`.

        Expects files named `route_*.csv`; only trips with at least one valid
        point are kept. Logs basic information about loaded routes.
        """
        if not self.trips_directory.exists():
            print(f"Warning: Trips directory not found: {self.trips_directory}")
            return

        csv_files = list(self.trips_directory.glob("route_*.csv"))
        for csv_file in sorted(csv_files):
            try:
                trip = Trip(str(csv_file))
                if trip.points:
                    self.available_trips.append(trip)
                    print(
                        f"  Loaded {trip.route_name}: {len(trip.points)} points, "
                        f"{trip.duration/3600:.2f}h"
                    )
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")

        if self.available_trips:
            print(f"  Total trips loaded: {len(self.available_trips)}")
        else:
            print("  Warning: No trips loaded!")

    def assign_daily_trips(self, boat_name: str, current_date: datetime) -> List[Trip]:
        """
        Assign trips for a boat for the given day if not already assigned.

        Args:
            boat_name: Name of the boat
            current_date: Current simulation date

        Returns:
            List of trips assigned for this day (may be empty).
        """
        if not self.available_trips:
            return []

        date_str = current_date.strftime("%Y-%m-%d")
        weekday = current_date.weekday()

        if boat_name in self.daily_assignments:
            if date_str in self.daily_assignments[boat_name]:
                return self.daily_assignments[boat_name][date_str]

        if weekday < 5:
            num_trips = 2
        elif weekday == 5:
            num_trips = 1
        else:
            num_trips = 0

        assigned_trips = []
        if num_trips > 0:
            if num_trips <= len(self.available_trips):
                assigned_trips = random.sample(self.available_trips, num_trips)
            else:
                assigned_trips = random.choices(self.available_trips, k=num_trips)

        if boat_name not in self.daily_assignments:
            self.daily_assignments[boat_name] = {}
        self.daily_assignments[boat_name][date_str] = assigned_trips

        return assigned_trips

    def get_trip_for_slot(
        self, boat_name: str, current_date: datetime, slot: int
    ) -> Optional[Trip]:
        """
        Get the assigned trip for a specific time slot on a given date.

        Args:
            boat_name: Name of the boat
            current_date: Current simulation date
            slot: Time slot index (e.g. 0 = morning, 1 = afternoon).

        Returns:
            The assigned trip, or None if no trip exists for this slot.
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

    def get_trips_for_date(self, boat_name: str, current_date: datetime) -> List[Trip]:
        """
        Get all assigned trips for a boat on a specific date.

        Args:
            boat_name: Name of the boat
            current_date: Date to get trips for

        Returns:
            List of trips assigned for this date (may be empty).
        """
        date_str = current_date.strftime("%Y-%m-%d")

        if boat_name not in self.daily_assignments:
            return []

        if date_str not in self.daily_assignments[boat_name]:
            return []

        return self.daily_assignments[boat_name][date_str]
