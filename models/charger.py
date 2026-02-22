"""Charger model for the electric recreational port simulator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChargerState(Enum):
    """Charger state: idle or charging."""

    IDLE = "idle"
    CHARGING = "charging"


@dataclass
class Charger:
    """
    Charging station at the port.

    Attributes:
        max_power: Maximum output power (kW).
        efficiency: Charging efficiency in (0, 1]; default 0.95.
        power: Current output power (kW).
        name: Identifier; auto-generated as Charger_N if empty.
        _state: Internal state; use .state property.
        connected_boat: Name of connected boat, or None.
    """

    max_power: int
    efficiency: float = 0.95
    power: float = 0.0
    name: str = ""
    _state: ChargerState = field(default=ChargerState.IDLE, init=False)
    connected_boat: Optional[str] = field(default=None, init=False)
    _charger_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate attributes and set default name if empty."""
        if not self.name:
            Charger._charger_count = getattr(Charger, "_charger_count", 0) + 1
            self.name = f"Charger_{Charger._charger_count}"

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
        """Current charger state."""
        return self._state

    @state.setter
    def state(self, new_state: ChargerState):
        """Set state; must be ChargerState. If IDLE, power and connected_boat are reset."""
        if not isinstance(new_state, ChargerState):
            raise ValueError(
                f"State must be a ChargerState enum, got {type(new_state)}"
            )
        self._state = new_state

        if new_state == ChargerState.IDLE:
            self.power = 0.0
            self.connected_boat = None

    @property
    def effective_power(self) -> float:
        """Power delivered to the battery after efficiency (kW)."""
        return self.power * self.efficiency

    def __repr__(self) -> str:
        boat_info = (
            f", connected to '{self.connected_boat}'" if self.connected_boat else ""
        )
        return (
            f"Charger(name='{self.name}', max_power={self.max_power}kW, "
            f"efficiency={self.efficiency:.1%}, power={self.power}kW, "
            f"state={self.state.value}{boat_info})"
        )
