"""Boat model for the electric recreational port simulator."""

from dataclasses import dataclass, field
from enum import Enum


class BoatState(Enum):
    """Possible states for a boat."""

    IDLE = "idle"
    CHARGING = "charging"
    SAILING = "sailing"


@dataclass
class Boat:
    """
    Represents an electric boat in the port.

    Attributes:
        name: Name/identifier of the boat
        motor_power: Motor power in kW
        weight: Boat weight in kg
        length: Boat length in meters
        battery_capacity: Battery capacity in kWh
        range_speed: Range speed in knots
        soc: State of charge (0-1, where 1 = 100%)
        _state: Current state of the boat
    """

    motor_power: int  # kW
    weight: float  # kg
    length: float  # m
    battery_capacity: float  # kWh
    range_speed: float  # knots
    soc: float = 1.0  # Default fully charged (0-1)
    name: str = ""  # Will be set in __post_init__ if empty
    _state: BoatState = field(default=BoatState.IDLE, init=False)

    # Class variable to track boat count for default naming
    _boat_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate boat attributes and set default name if needed."""
        # Set default name if not provided
        if not self.name:
            Boat._boat_count = getattr(Boat, "_boat_count", 0) + 1
            self.name = f"Boat_{Boat._boat_count}"

        # Validation
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
        """
        Calculate k-factor for cube law: motor_power / range_speed^3.

        Returns:
            K-factor for power consumption calculations
        """
        return self.motor_power / (self.range_speed**3)

    @property
    def state(self) -> BoatState:
        """Get the current state of the boat."""
        return self._state

    @state.setter
    def state(self, new_state: BoatState):
        """
        Set the state of the boat.

        Args:
            new_state: New state for the boat
        """
        if not isinstance(new_state, BoatState):
            raise ValueError(f"State must be a BoatState enum, got {type(new_state)}")
        self._state = new_state

    def __repr__(self) -> str:
        return (
            f"Boat(name='{self.name}', motor_power={self.motor_power}kW, "
            f"battery={self.battery_capacity}kWh, soc={self.soc:.1%}, "
            f"state={self.state.value})"
        )

