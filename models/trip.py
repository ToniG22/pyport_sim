"""Trip model for boat routes."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


@dataclass
class TripPoint:
    """
    One waypoint in a trip route.

    Attributes:
        timestamp: Time at this point.
        point_type: Type label (e.g. Static, Dock, Terrestrial, Interpolated).
        speed: Speed in knots.
        heading: Heading in degrees.
        latitude: Latitude.
        longitude: Longitude.
    """

    timestamp: datetime
    point_type: str
    speed: float
    heading: float
    latitude: float
    longitude: float


class Trip:
    """
    Boat trip with waypoints loaded from CSV.

    Attributes:
        route_name: Route name (from CSV filename stem).
        points: List of TripPoint waypoints.
        duration: Total trip duration in seconds.
    """

    def __init__(self, csv_path: str):
        """Load trip from a CSV file (columns: timestamp, type, speed, heading, latitude, longitude)."""
        self.route_name = Path(csv_path).stem
        self.points: List[TripPoint] = []
        self._load_from_csv(csv_path)

        if self.points:
            first_time = self.points[0].timestamp
            last_time = self.points[-1].timestamp
            self.duration = (last_time - first_time).total_seconds()
        else:
            self.duration = 0

    def _load_from_csv(self, csv_path: str):
        """Parse CSV into TripPoint list; skips empty rows and rows missing timestamp/speed/latitude."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for _row_num, row in enumerate(reader, start=2):
                if not any(row.values()):
                    continue

                if not row.get("timestamp") or not row.get("timestamp").strip():
                    continue

                try:
                    timestamp_str = row["timestamp"].strip()
                    try:
                        timestamp = datetime.strptime(
                            timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                        )
                    except ValueError:
                        if "." in timestamp_str:
                            base, frac = timestamp_str.split(".")
                            frac = frac[:6].ljust(6, "0")
                            timestamp_str = f"{base}.{frac}"
                            timestamp = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                            )
                        else:
                            timestamp = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S"
                            )

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
                    raise ValueError(f"Error parsing CSV: {csv_path}") from e

    def get_point_at_elapsed_time(self, elapsed_seconds: float) -> Optional[TripPoint]:
        """
        Return the waypoint closest to the given elapsed time from trip start.

        Args:
            elapsed_seconds: Seconds since trip start.

        Returns:
            Closest TripPoint, or None if elapsed_seconds exceeds trip duration.
        """
        if not self.points or elapsed_seconds > self.duration:
            return None

        target_time = self.points[0].timestamp + timedelta(seconds=elapsed_seconds)
        closest_point = self.points[0]
        min_diff = abs((target_time - closest_point.timestamp).total_seconds())

        for point in self.points[1:]:
            diff = abs((target_time - point.timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_point = point
            else:
                break

        return closest_point

    def estimate_energy_required(self, boat_k_factor: float) -> float:
        """
        Estimate total energy (kWh) for the full trip using cube-law power per segment.

        Args:
            boat_k_factor: Boat k-factor (motor_power / range_speed^3).

        Returns:
            Total energy in kWh.
        """
        if not self.points:
            return 0.0

        total_energy = 0.0
        for i in range(len(self.points) - 1):
            point = self.points[i]
            next_point = self.points[i + 1]
            segment_duration = (next_point.timestamp - point.timestamp).total_seconds()
            power_kw = boat_k_factor * (point.speed**3)
            energy_kwh = (power_kw * segment_duration) / 3600
            total_energy += energy_kwh

        return total_energy

    def get_energy_between(
        self,
        start_elapsed_seconds: float,
        end_elapsed_seconds: float,
        boat_k_factor: float,
    ) -> float:
        """
        Energy (kWh) consumed in [start_elapsed, end_elapsed] using same segment integration as estimate_energy_required.

        Args:
            start_elapsed_seconds: Window start (s from trip start).
            end_elapsed_seconds: Window end (s from trip start).
            boat_k_factor: Boat k-factor (motor_power / range_speed^3).

        Returns:
            Energy in kWh for the window.
        """
        if not self.points or start_elapsed_seconds >= end_elapsed_seconds:
            return 0.0

        t0 = self.points[0].timestamp
        total_energy = 0.0
        for i in range(len(self.points) - 1):
            point = self.points[i]
            next_point = self.points[i + 1]
            seg_start = (point.timestamp - t0).total_seconds()
            seg_end = (next_point.timestamp - t0).total_seconds()

            overlap_start = max(seg_start, start_elapsed_seconds)
            overlap_end = min(seg_end, end_elapsed_seconds)
            duration = max(0.0, overlap_end - overlap_start)
            if duration <= 0:
                continue

            power_kw = boat_k_factor * (point.speed**3)
            total_energy += (power_kw * duration) / 3600

        return total_energy

    def __repr__(self) -> str:
        return (
            f"Trip(route={self.route_name}, points={len(self.points)}, "
            f"duration={self.duration/3600:.2f}h)"
        )
