"""Settings and configuration for the simulator."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
    """

    timestep: int = 900  # Default: 15 minute
    mode: SimulationMode = SimulationMode.BATCH
    db_path: str = "port_simulation.db"

    def __post_init__(self):
        """Validate settings."""
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
