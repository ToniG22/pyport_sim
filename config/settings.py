"""Settings and configuration for the simulator."""

from dataclasses import dataclass
from enum import Enum


class SimulationMode(Enum):
    """Simulation execution modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"


@dataclass
class Settings:
    """
    Global settings for the port simulator.

    Attributes:
        timestep: Simulation timestep in seconds
        mode: Simulation mode (real-time or batch)
        db_path: Path to SQLite database file
        use_optimizer: Whether to use optimization for scheduling
        power_limit_mode: Whether to enforce contracted power limit without optimization (baseline mode)
        trip_schedule: Trip departure times as (hour_utc, slot_index) per day, e.g. ((9, 0), (14, 1))
    """

    timestep: int = 900  # Default: 15 minutes
    mode: SimulationMode = SimulationMode.BATCH
    db_path: str = "port_simulation.db"
    use_optimizer: bool = False  # Default: use rule-based control
    power_limit_mode: bool = False  # Default: no power limiting (unlimited charging)
    # Trip schedule: list of (hour_utc, slot_index), e.g. 9:00 slot 0, 14:00 slot 1
    trip_schedule: tuple = ((9, 0), (14, 1))

    def __post_init__(self):
        """Validate settings."""
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
