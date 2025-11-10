"""Charger model for the electric recreational port simulator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChargerState(Enum):
    """Possible states for a charger."""

    IDLE = "idle"
    CHARGING = "charging"


@dataclass
class Charger:
    """
    Represents a charging station in the port.

    Attributes:
        name: Name/identifier of the charger
        max_power: Maximum power output in kW
        efficiency: Charging efficiency (0-1, where 1 = 100%)
        power: Current power output in kW
        _state: Current state of the charger
        connected_boat: Reference to connected boat (if any)
    """

    max_power: int  # kW
    efficiency: float = 0.95  # Default 95% efficiency (0-1)
    power: float = 0.0  # Current power output in kW
    name: str = ""  # Will be set in __post_init__ if empty
    _state: ChargerState = field(default=ChargerState.IDLE, init=False)
    connected_boat: Optional[str] = field(
        default=None, init=False
    )  # Boat name/id if connected

    # Class variable to track charger count for default naming
    _charger_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate charger attributes and set default name if needed."""
        # Set default name if not provided
        if not self.name:
            Charger._charger_count = getattr(Charger, "_charger_count", 0) + 1
            self.name = f"Charger_{Charger._charger_count}"

        # Validation
        if self.max_power <= 0:
            raise ValueError("Max power must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if self.power < 0:
            raise ValueError("Power cannot be negative")
        if self.power > self.max_power:
            raise ValueError("Power cannot exceed max_power")

    @property
    def state(self) -> ChargerState:
        """Get the current state of the charger."""
        return self._state

    @state.setter
    def state(self, new_state: ChargerState):
        """
        Set the state of the charger.

        Args:
            new_state: New state for the charger
        """
        if not isinstance(new_state, ChargerState):
            raise ValueError(
                f"State must be a ChargerState enum, got {type(new_state)}"
            )
        self._state = new_state

        # Reset power to 0 when going idle
        if new_state == ChargerState.IDLE:
            self.power = 0.0
            self.connected_boat = None

    @property
    def effective_power(self) -> float:
        """
        Calculate the effective power delivered to the battery after efficiency losses.

        Returns:
            Effective power in kW
        """
        return self.power * self.efficiency

    def __repr__(self) -> str:
        boat_info = f", connected to '{self.connected_boat}'" if self.connected_boat else ""
        return (
            f"Charger(name='{self.name}', max_power={self.max_power}kW, "
            f"efficiency={self.efficiency:.1%}, power={self.power}kW, "
            f"state={self.state.value}{boat_info})"
        )

