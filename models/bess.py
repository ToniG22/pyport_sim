"""BESS (Battery Energy Storage System) model."""

from dataclasses import dataclass
from enum import Enum


class BESSControlStrategy(Enum):
    """BESS control strategy; DEFAULT = charge from PV surplus, discharge when needed."""

    DEFAULT = "default"


@dataclass
class BESS:
    """
    Battery energy storage system with SOC limits and round-trip efficiency.

    Attributes:
        name: BESS identifier.
        capacity: Total energy capacity (kWh).
        max_charge_power: Maximum charge power (kW).
        max_discharge_power: Maximum discharge power (kW).
        efficiency: Round-trip efficiency in (0, 1]; default 0.90.
        soc_min: Minimum SOC in [0, 1]; default 0.10.
        soc_max: Maximum SOC in [0, 1]; default 0.90.
        initial_soc: Initial SOC; default 0.50.
        control_strategy: Control strategy enum.
        current_soc: Current SOC (updated by charge/discharge).
        current_power: Current power (kW); positive = charging, negative = discharging.
    """

    name: str
    capacity: float
    max_charge_power: float
    max_discharge_power: float
    efficiency: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50
    control_strategy: BESSControlStrategy = BESSControlStrategy.DEFAULT
    current_soc: float = 0.50
    current_power: float = 0.0

    def __post_init__(self):
        """Validate attributes and set current_soc from initial_soc."""
        if self.capacity <= 0:
            raise ValueError("BESS capacity must be positive")
        if self.max_charge_power <= 0:
            raise ValueError("Max charge power must be positive")
        if self.max_discharge_power <= 0:
            raise ValueError("Max discharge power must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("SOC limits must satisfy: 0 ≤ soc_min < soc_max ≤ 1")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("Initial SOC must be between soc_min and soc_max")

        self.current_soc = self.initial_soc

    def charge(self, power: float, timestep_seconds: float) -> float:
        """
        Charge for the given power and timestep; SOC is capped at soc_max.

        Args:
            power: Requested charging power (kW), must be positive.
            timestep_seconds: Timestep duration (s).

        Returns:
            Actual power charged (kW); may be less than requested if SOC limit reached.
        """
        if power < 0:
            raise ValueError("Charging power must be positive")

        actual_power = min(power, self.max_charge_power)
        energy_added = (actual_power * timestep_seconds / 3600.0) * self.efficiency
        new_soc = self.current_soc + (energy_added / self.capacity)

        if new_soc > self.soc_max:
            allowed_energy = (self.soc_max - self.current_soc) * self.capacity
            actual_power = (allowed_energy / self.efficiency) / (
                timestep_seconds / 3600.0
            )
            self.current_soc = self.soc_max
        else:
            self.current_soc = new_soc

        self.current_power = actual_power
        return actual_power

    def discharge(self, power: float, timestep_seconds: float) -> float:
        """
        Discharge for the given power and timestep; SOC is floored at soc_min.

        Args:
            power: Requested discharge power (kW), must be positive.
            timestep_seconds: Timestep duration (s).

        Returns:
            Actual power discharged (kW); may be less than requested if SOC limit reached.
        """
        if power < 0:
            raise ValueError("Discharging power must be positive")

        actual_power = min(power, self.max_discharge_power)
        energy_removed = actual_power * timestep_seconds / 3600.0
        energy_from_battery = energy_removed / self.efficiency
        new_soc = self.current_soc - (energy_from_battery / self.capacity)

        if new_soc < self.soc_min:
            allowed_energy = (self.current_soc - self.soc_min) * self.capacity
            actual_power = (allowed_energy * self.efficiency) / (
                timestep_seconds / 3600.0
            )
            self.current_soc = self.soc_min
        else:
            self.current_soc = new_soc

        self.current_power = -actual_power
        return actual_power

    def get_available_energy(self) -> float:
        """Energy (kWh) that can still be discharged (from current_soc down to soc_min, after efficiency)."""
        return (self.current_soc - self.soc_min) * self.capacity * self.efficiency

    def get_available_charge_capacity(self) -> float:
        """Capacity (kWh) available for charging (from current_soc to soc_max, accounting for efficiency)."""
        return (self.soc_max - self.current_soc) * self.capacity / self.efficiency

    def get_max_discharge_power_available(self, timestep_seconds: float) -> float:
        """Maximum discharge power (kW) in this timestep (min of max_discharge_power and energy-limited power)."""
        available_energy = self.get_available_energy()
        energy_limited_power = available_energy / (timestep_seconds / 3600.0)
        return min(self.max_discharge_power, energy_limited_power)

    def get_max_charge_power_available(self, timestep_seconds: float) -> float:
        """Maximum charge power (kW) in this timestep (min of max_charge_power and capacity-limited power)."""
        available_capacity = self.get_available_charge_capacity()
        capacity_limited_power = available_capacity / (timestep_seconds / 3600.0)
        return min(self.max_charge_power, capacity_limited_power)

    def idle(self) -> None:
        """Set current_power to 0 (no charge or discharge)."""
        self.current_power = 0.0

    def get_energy_stored(self) -> float:
        """Current energy stored in the battery (kWh)."""
        return self.current_soc * self.capacity

    def __repr__(self) -> str:
        return (
            f"BESS(name='{self.name}', capacity={self.capacity}kWh, "
            f"SOC={self.current_soc*100:.1f}%, "
            f"power={self.current_power:+.2f}kW, "
            f"strategy={self.control_strategy.value})"
        )
