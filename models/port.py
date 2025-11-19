"""Port model for the electric recreational port simulator."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .boat import Boat
    from .charger import Charger
    from .pv import PV
    from .bess import BESS


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
        bess_systems: List of BESS (battery storage) systems at this port
    """

    name: str
    contracted_power: int  # in kW
    lat: float
    lon: float
    boats: List["Boat"] = field(default_factory=list)
    chargers: List["Charger"] = field(default_factory=list)
    pv_systems: List["PV"] = field(default_factory=list)
    bess_systems: List["BESS"] = field(default_factory=list)
    tariff_path: Optional[str] = None  # Path to tariff JSON file
    _tariff_data: Optional[Dict] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate port attributes."""
        if self.contracted_power <= 0:
            raise ValueError("Contracted power must be positive")
        if not -90 <= self.lat <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= self.lon <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        
        # Load tariff if path is provided
        if self.tariff_path:
            self._load_tariff()
        else:
            # Try default tariff path
            default_path = Path(__file__).parent.parent / "assets" / "tariff" / "default_tariff.json"
            if default_path.exists():
                self.tariff_path = str(default_path)
                self._load_tariff()
    
    def _load_tariff(self) -> None:
        """Load tariff data from JSON file."""
        if not self.tariff_path:
            return
        
        tariff_file = Path(self.tariff_path)
        if not tariff_file.exists():
            raise FileNotFoundError(f"Tariff file not found: {self.tariff_path}")
        
        try:
            with open(tariff_file, 'r') as f:
                self._tariff_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tariff file: {e}")
    
    @property
    def tariff(self) -> Optional[Dict]:
        """Get tariff data."""
        return self._tariff_data
    
    def get_tariff_price(self, timestamp: datetime) -> float:
        """
        Get electricity price for a specific timestamp.
        
        Args:
            timestamp: Datetime to get price for
            
        Returns:
            Price per kWh in the tariff currency (default: 0.0 if no tariff)
        """
        if not self._tariff_data or "tariff" not in self._tariff_data:
            return 0.0
        
        # Get day of week (0=Monday, 6=Sunday)
        weekday = timestamp.weekday()
        day_key = str(weekday)
        
        # Get pricing for this day
        if day_key not in self._tariff_data["tariff"]:
            return 0.0
        
        day_pricing = self._tariff_data["tariff"][day_key]["pricing"]
        
        # Format time as HH:MM (15-minute intervals)
        time_str = timestamp.strftime("%H:%M")
        
        # Round to nearest 15 minutes
        minute = timestamp.minute
        rounded_minute = (minute // 15) * 15
        time_str = timestamp.strftime(f"%H:{rounded_minute:02d}")
        
        # Get price for this time slot
        return day_pricing.get(time_str, 0.0)

    def add_boat(self, boat: "Boat") -> None:
        """Add a boat to the port."""
        self.boats.append(boat)

    def add_charger(self, charger: "Charger") -> None:
        """Add a charger to the port."""
        self.chargers.append(charger)

    def add_pv(self, pv: "PV") -> None:
        """Add a PV system to the port."""
        self.pv_systems.append(pv)

    def add_bess(self, bess: "BESS") -> None:
        """Add a BESS to the port."""
        self.bess_systems.append(bess)

    def __repr__(self) -> str:
        return (
            f"Port(name='{self.name}', contracted_power={self.contracted_power}kW, "
            f"coordinates=({self.lat}, {self.lon}), "
            f"boats={len(self.boats)}, chargers={len(self.chargers)}, "
            f"pv={len(self.pv_systems)}, bess={len(self.bess_systems)})"
        )
