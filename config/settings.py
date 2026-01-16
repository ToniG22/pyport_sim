"""Settings and configuration for the simulator."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SimulationMode(Enum):
    """Simulation execution modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"


class OptimizerType(Enum):
    """Optimizer types for charging schedule optimization."""

    COST = "cost"  # Cost-minimization optimizer (original PortOptimizer)
    RELIABILITY = "reliability"  # Reliability-focused optimizer (ReliabilityOptimizer)
    RELIABILITY_FIRST = (
        "reliability_first"  # Reliability-first optimizer (ReliabilityFirstOptimizer)
    )


@dataclass
class Settings:
    """
    Global settings for the port simulator.

    Attributes:
        timestep: Simulation timestep in seconds
        mode: Simulation mode (real-time or batch)
        db_path: Path to SQLite database file
        use_optimizer: Whether to use optimization for scheduling
        optimizer_type: Type of optimizer to use (cost or reliability)
    """

    timestep: int = 900  # Default: 15 minutes
    mode: SimulationMode = SimulationMode.BATCH
    db_path: str = "port_simulation.db"
    use_optimizer: bool = False  # Default: use rule-based control
    optimizer_type: OptimizerType = (
        OptimizerType.RELIABILITY_FIRST
    )  # Default: reliability-focused

    def __post_init__(self):
        """Validate settings."""
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
