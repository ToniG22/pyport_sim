"""Port model for the electric recreational port simulator."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .bess import BESS
    from .boat import Boat
    from .charger import Charger
    from .pv import PV


@dataclass
class Port:
    """
    Electric recreational port with boats, chargers, PV, BESS, and optional tariff.

    Attributes:
        name: Port name.
        contracted_power: Contracted power limit (kW).
        lat: Latitude.
        lon: Longitude.
        boats: List of boats.
        chargers: List of chargers.
        pv_systems: List of PV systems.
        bess_systems: List of BESS systems.
        tariff_path: Optional path to tariff JSON; if None, default_tariff.json is tried.
    """

    name: str
    contracted_power: int
    lat: float
    lon: float
    boats: List["Boat"] = field(default_factory=list)
    chargers: List["Charger"] = field(default_factory=list)
    pv_systems: List["PV"] = field(default_factory=list)
    bess_systems: List["BESS"] = field(default_factory=list)
    tariff_path: Optional[str] = None
    _tariff_data: Optional[Dict] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate attributes and load tariff from tariff_path or default path if present."""
        if self.contracted_power <= 0:
            raise ValueError("Contracted power must be positive")
        if not -90 <= self.lat <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= self.lon <= 180:
            raise ValueError("Longitude must be between -180 and 180")

        if self.tariff_path:
            self._load_tariff()
        else:
            default_path = (
                Path(__file__).parent.parent
                / "assets"
                / "tariff"
                / "default_tariff.json"
            )
            if default_path.exists():
                self.tariff_path = str(default_path)
                self._load_tariff()

    def _load_tariff(self) -> None:
        """Load tariff JSON from tariff_path into _tariff_data."""
        if not self.tariff_path:
            return

        tariff_file = Path(self.tariff_path)
        if not tariff_file.exists():
            raise FileNotFoundError(f"Tariff file not found: {self.tariff_path}")

        try:
            with open(tariff_file, "r", encoding="utf-8") as f:
                self._tariff_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tariff file: {tariff_file}") from e

    @property
    def tariff(self) -> Optional[Dict]:
        """Loaded tariff data (from tariff_path); None if no tariff loaded."""
        return self._tariff_data

    def get_tariff_price(self, timestamp: datetime) -> float:
        """
        Return electricity price per kWh for the given timestamp (15-minute resolution).

        Args:
            timestamp: Time to look up.

        Returns:
            Price per kWh, or 0.0 if no tariff or slot not found.
        """
        if not self._tariff_data or "tariff" not in self._tariff_data:
            return 0.0

        weekday = timestamp.weekday()
        day_key = str(weekday)
        if day_key not in self._tariff_data["tariff"]:
            return 0.0

        day_pricing = self._tariff_data["tariff"][day_key]["pricing"]
        minute = timestamp.minute
        rounded_minute = (minute // 15) * 15
        time_str = timestamp.strftime(f"%H:{rounded_minute:02d}")
        return day_pricing.get(time_str, 0.0)

    def add_boat(self, boat: "Boat") -> None:
        """Append a boat to the port's boat list."""
        self.boats.append(boat)

    def add_charger(self, charger: "Charger") -> None:
        """Append a charger to the port's charger list."""
        self.chargers.append(charger)

    def add_pv(self, pv: "PV") -> None:
        """Append a PV system to the port's pv_systems list."""
        self.pv_systems.append(pv)

    def add_bess(self, bess: "BESS") -> None:
        """Append a BESS to the port's bess_systems list."""
        self.bess_systems.append(bess)

    def __repr__(self) -> str:
        return (
            f"Port(name='{self.name}', contracted_power={self.contracted_power}kW, "
            f"coordinates=({self.lat}, {self.lon}), "
            f"boats={len(self.boats)}, chargers={len(self.chargers)}, "
            f"pv={len(self.pv_systems)}, bess={len(self.bess_systems)})"
        )
