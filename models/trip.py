"""Trip model for boat routes."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


@dataclass
class TripPoint:
    """A single point in a trip route."""

    timestamp: datetime
    point_type: str  # Static, Dock, Terrestrial, Interpolated
    speed: float  # knots
    heading: float  # degrees
    latitude: float
    longitude: float


class Trip:
    """
    Represents a boat trip with waypoints.

    Attributes:
        route_name: Name of the route (e.g., 'route_1')
        points: List of trip waypoints
        duration: Total trip duration in seconds
    """

    def __init__(self, csv_path: str):
        """
        Load a trip from a CSV file.

        Args:
            csv_path: Path to the CSV file containing trip data
        """
        self.route_name = Path(csv_path).stem
        self.points: List[TripPoint] = []
        self._load_from_csv(csv_path)

        if self.points:
            # Calculate trip duration
            first_time = self.points[0].timestamp
            last_time = self.points[-1].timestamp
            self.duration = (last_time - first_time).total_seconds()
        else:
            self.duration = 0

    def _load_from_csv(self, csv_path: str):
        """Load trip points from CSV file."""
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is line 1)
                # Skip completely empty rows
                if not any(row.values()):
                    continue

                # Skip rows without timestamp
                if not row.get("timestamp") or not row.get("timestamp").strip():
                    continue

                try:
                    # Parse timestamp - handle both microseconds and nanoseconds
                    timestamp_str = row["timestamp"].strip()
                    try:
                        # Try with microseconds first (standard Python format)
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        # Handle nanoseconds by truncating to microseconds
                        if "." in timestamp_str:
                            base, frac = timestamp_str.split(".")
                            # Truncate to 6 digits (microseconds)
                            frac = frac[:6].ljust(6, "0")
                            timestamp_str = f"{base}.{frac}"
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                        else:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                    # Skip rows with empty data fields
                    if not row.get("speed") or not row.get("speed").strip():
                        continue
                    if not row.get("latitude") or not row.get("latitude").strip():
                        continue

                    point = TripPoint(
                        timestamp=timestamp,
                        point_type=row.get("type", ""),
                        speed=float(row["speed"]),
                        heading=float(row["heading"]) if row.get("heading") else 0.0,
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                    )
                    self.points.append(point)
                except (ValueError, KeyError) as e:
                    # Skip problematic rows silently
                    pass

    def get_point_at_elapsed_time(self, elapsed_seconds: float) -> Optional[TripPoint]:
        """
        Get the trip point at a given elapsed time from trip start.

        Args:
            elapsed_seconds: Seconds elapsed since trip start

        Returns:
            The closest trip point, or None if past the end
        """
        if not self.points or elapsed_seconds > self.duration:
            return None

        # Find the closest point by time
        target_time = self.points[0].timestamp + timedelta(seconds=elapsed_seconds)

        # Binary search would be more efficient, but linear is fine for our data size
        closest_point = self.points[0]
        min_diff = abs((target_time - closest_point.timestamp).total_seconds())

        for point in self.points[1:]:
            diff = abs((target_time - point.timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_point = point
            else:
                # Points are sorted, so we can stop once diff starts increasing
                break

        return closest_point

    def estimate_energy_required(self, boat_k_factor: float) -> float:
        """
        Estimate total energy required for this trip.

        Args:
            boat_k_factor: The boat's k factor (motor_power / range_speed^3)

        Returns:
            Estimated energy in kWh
        """
        if not self.points:
            return 0.0

        # Sum up energy for each segment based on speed
        total_energy = 0.0
        for i in range(len(self.points) - 1):
            point = self.points[i]
            next_point = self.points[i + 1]

            # Time for this segment
            segment_duration = (
                next_point.timestamp - point.timestamp
            ).total_seconds()  # seconds

            # Power consumption at this speed
            speed_knots = point.speed
            power_kw = boat_k_factor * (speed_knots**3)

            # Energy for this segment
            energy_kwh = (power_kw * segment_duration) / 3600

            total_energy += energy_kwh

        return total_energy

    def __repr__(self) -> str:
        return (
            f"Trip(route={self.route_name}, points={len(self.points)}, "
            f"duration={self.duration/3600:.2f}h)"
        )

