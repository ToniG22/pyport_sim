"""Port model for the electric recreational port simulator."""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .boat import Boat
    from .charger import Charger
    from .pv import PV


@dataclass
class Port:
    """
    Represents an electric recreational port.

    Attributes:
        name: Name of the port
        contracted_power: Maximum contracted power in kW
        lat: Latitude coordinate
        lon: Longitude coordinate
        boats: List of boats at this port
        chargers: List of chargers at this port
        pv_systems: List of PV systems at this port
    """

    name: str
    contracted_power: int  # in kW
    lat: float
    lon: float
    boats: List["Boat"] = field(default_factory=list)
    chargers: List["Charger"] = field(default_factory=list)
    pv_systems: List["PV"] = field(default_factory=list)

    def __post_init__(self):
        """Validate port attributes."""
        if self.contracted_power <= 0:
            raise ValueError("Contracted power must be positive")
        if not -90 <= self.lat <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= self.lon <= 180:
            raise ValueError("Longitude must be between -180 and 180")

    def add_boat(self, boat: "Boat") -> None:
        """Add a boat to the port."""
        self.boats.append(boat)

    def add_charger(self, charger: "Charger") -> None:
        """Add a charger to the port."""
        self.chargers.append(charger)

    def add_pv(self, pv: "PV") -> None:
        """Add a PV system to the port."""
        self.pv_systems.append(pv)

    def __repr__(self) -> str:
        return (
            f"Port(name='{self.name}', contracted_power={self.contracted_power}kW, "
            f"coordinates=({self.lat}, {self.lon}), "
            f"boats={len(self.boats)}, chargers={len(self.chargers)}, "
            f"pv={len(self.pv_systems)})"
        )
