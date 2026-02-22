"""Boat model for the electric recreational port simulator."""

from dataclasses import dataclass, field
from enum import Enum


class BoatState(Enum):
    """Boat state: idle, charging, or sailing."""

    IDLE = "idle"
    CHARGING = "charging"
    SAILING = "sailing"


@dataclass
class Boat:
    """
    Electric boat at the port.

    Attributes:
        motor_power: Motor power (kW).
        weight: Weight (kg).
        length: Length (m).
        battery_capacity: Battery capacity (kWh).
        range_speed: Range speed (knots).
        soc: State of charge in [0, 1]; default 1.0 (full).
        name: Identifier; auto-generated as Boat_N if empty.
        _state: Internal state (BoatState); use .state property.
    """

    motor_power: int
    weight: float
    length: float
    battery_capacity: float
    range_speed: float
    soc: float = 1.0
    name: str = ""
    _state: BoatState = field(default=BoatState.IDLE, init=False)
    _boat_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate attributes and set default name if empty."""
        if not self.name:
            Boat._boat_count = getattr(Boat, "_boat_count", 0) + 1
            self.name = f"Boat_{Boat._boat_count}"

        if self.motor_power <= 0:
            raise ValueError("Motor power must be positive")
        if self.weight <= 0:
            raise ValueError("Weight must be positive")
        if self.length <= 0:
            raise ValueError("Length must be positive")
        if self.battery_capacity <= 0:
            raise ValueError("Battery capacity must be positive")
        if self.range_speed <= 0:
            raise ValueError("Range speed must be positive")
        if not 0 <= self.soc <= 1:
            raise ValueError("SOC must be between 0 and 1")

    @property
    def k(self) -> float:
        """Return k-factor for cube law: motor_power / range_speed^3 (used in trip energy estimates)."""
        return self.motor_power / (self.range_speed**3)

    @property
    def state(self) -> BoatState:
        """Current boat state."""
        return self._state

    @state.setter
    def state(self, new_state: BoatState):
        """Set boat state; new_state must be a BoatState enum."""
        if not isinstance(new_state, BoatState):
            raise ValueError(f"State must be a BoatState enum, got {type(new_state)}")
        self._state = new_state

    def __repr__(self) -> str:
        return (
            f"Boat(name='{self.name}', motor_power={self.motor_power}kW, "
            f"battery={self.battery_capacity}kWh, soc={self.soc:.1%}, "
            f"state={self.state.value})"
        )
