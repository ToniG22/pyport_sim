"""Settings and configuration for the port simulator."""

from dataclasses import dataclass
from enum import Enum


class SimulationMode(Enum):
    """Execution mode for the simulation (real-time or batch)."""

    REAL_TIME = "real_time"
    BATCH = "batch"


@dataclass
class Settings:
    """
    Global settings for the port simulator.

    Attributes:
        timestep: Simulation timestep in seconds (default 900, i.e. 15 minutes).
        mode: Simulation mode (real-time or batch).
        db_path: Path to the SQLite database file.
        use_optimizer: If True, use optimization for scheduling; if False, use rule-based control.
        power_limit_mode: If True, enforce contracted power limit without optimization (baseline);
            if False, charging is unlimited.
        trip_schedule: Trip departure times per day as a tuple of (hour_utc, slot_index);
            e.g. ((9, 0), (14, 1)) for 09:00 slot 0 and 14:00 slot 1.
    """

    timestep: int = 900
    mode: SimulationMode = SimulationMode.BATCH
    db_path: str = "port_simulation.db"
    use_optimizer: bool = False
    power_limit_mode: bool = False
    trip_schedule: tuple = ((9, 0), (14, 1))

    def __post_init__(self):
        """Validate settings; raises ValueError if timestep is not positive."""
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
