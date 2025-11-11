"""BESS (Battery Energy Storage System) model."""

from dataclasses import dataclass
from enum import Enum


class BESSControlStrategy(Enum):
    """Control strategies for BESS operation."""
    DEFAULT = "default"  # Charge from PV surplus, discharge when needed
    # Future strategies can be added here (e.g., PEAK_SHAVING, ARBITRAGE, etc.)


@dataclass
class BESS:
    """
    Represents a Battery Energy Storage System at the port.
    
    Attributes:
        name: Name/identifier of the BESS
        capacity: Total energy capacity in kWh
        max_charge_power: Maximum charging power in kW
        max_discharge_power: Maximum discharging power in kW
        efficiency: Round-trip efficiency (0-1)
        soc_min: Minimum state of charge (0-1), default 0.10 (10%)
        soc_max: Maximum state of charge (0-1), default 0.90 (90%)
        initial_soc: Initial state of charge (0-1), default 0.50 (50%)
        control_strategy: Control strategy for operation
        current_soc: Current state of charge (0-1)
        current_power: Current power (kW) - positive=charging, negative=discharging
    """
    
    name: str
    capacity: float  # kWh
    max_charge_power: float  # kW
    max_discharge_power: float  # kW
    efficiency: float = 0.90  # 90% round-trip efficiency
    soc_min: float = 0.10  # 10% minimum SOC
    soc_max: float = 0.90  # 90% maximum SOC
    initial_soc: float = 0.50  # Start at 50%
    control_strategy: BESSControlStrategy = BESSControlStrategy.DEFAULT
    current_soc: float = 0.50
    current_power: float = 0.0  # kW (+ charging, - discharging)
    
    def __post_init__(self):
        """Validate BESS attributes and initialize state."""
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
        
        # Initialize current SOC
        self.current_soc = self.initial_soc
    
    def charge(self, power: float, timestep_seconds: float) -> float:
        """
        Charge the battery with given power for a timestep.
        
        Args:
            power: Charging power in kW (must be positive)
            timestep_seconds: Duration of the timestep in seconds
            
        Returns:
            Actual power charged (may be less than requested if SOC limit reached)
        """
        if power < 0:
            raise ValueError("Charging power must be positive")
        
        # Limit to max charge power
        actual_power = min(power, self.max_charge_power)
        
        # Calculate energy to be added (with efficiency loss during charging)
        energy_added = (actual_power * timestep_seconds / 3600.0) * self.efficiency  # kWh
        
        # Calculate new SOC
        new_soc = self.current_soc + (energy_added / self.capacity)
        
        # Check SOC limit
        if new_soc > self.soc_max:
            # Reduce charging to hit exactly soc_max
            allowed_energy = (self.soc_max - self.current_soc) * self.capacity
            actual_power = (allowed_energy / self.efficiency) / (timestep_seconds / 3600.0)
            self.current_soc = self.soc_max
        else:
            self.current_soc = new_soc
        
        self.current_power = actual_power
        return actual_power
    
    def discharge(self, power: float, timestep_seconds: float) -> float:
        """
        Discharge the battery with given power for a timestep.
        
        Args:
            power: Discharging power in kW (must be positive)
            timestep_seconds: Duration of the timestep in seconds
            
        Returns:
            Actual power discharged (may be less than requested if SOC limit reached)
        """
        if power < 0:
            raise ValueError("Discharging power must be positive")
        
        # Limit to max discharge power
        actual_power = min(power, self.max_discharge_power)
        
        # Calculate energy to be removed (with efficiency loss during discharging)
        # When discharging, we lose energy due to efficiency
        energy_removed = actual_power * timestep_seconds / 3600.0  # kWh delivered
        energy_from_battery = energy_removed / self.efficiency  # kWh removed from battery
        
        # Calculate new SOC
        new_soc = self.current_soc - (energy_from_battery / self.capacity)
        
        # Check SOC limit
        if new_soc < self.soc_min:
            # Reduce discharging to hit exactly soc_min
            allowed_energy = (self.current_soc - self.soc_min) * self.capacity
            actual_power = (allowed_energy * self.efficiency) / (timestep_seconds / 3600.0)
            self.current_soc = self.soc_min
        else:
            self.current_soc = new_soc
        
        self.current_power = -actual_power  # Negative for discharging
        return actual_power
    
    def get_available_energy(self) -> float:
        """
        Get available energy that can be discharged (kWh).
        
        Returns:
            Available energy in kWh
        """
        return (self.current_soc - self.soc_min) * self.capacity * self.efficiency
    
    def get_available_charge_capacity(self) -> float:
        """
        Get available capacity for charging (kWh).
        
        Returns:
            Available charge capacity in kWh
        """
        return (self.soc_max - self.current_soc) * self.capacity / self.efficiency
    
    def get_max_discharge_power_available(self, timestep_seconds: float) -> float:
        """
        Get maximum power that can be discharged in the current timestep.
        
        Args:
            timestep_seconds: Duration of the timestep in seconds
            
        Returns:
            Maximum discharge power in kW
        """
        # Limited by available energy
        available_energy = self.get_available_energy()
        energy_limited_power = available_energy / (timestep_seconds / 3600.0)
        
        # Limited by max discharge power
        return min(self.max_discharge_power, energy_limited_power)
    
    def get_max_charge_power_available(self, timestep_seconds: float) -> float:
        """
        Get maximum power that can be charged in the current timestep.
        
        Args:
            timestep_seconds: Duration of the timestep in seconds
            
        Returns:
            Maximum charge power in kW
        """
        # Limited by available capacity
        available_capacity = self.get_available_charge_capacity()
        capacity_limited_power = available_capacity / (timestep_seconds / 3600.0)
        
        # Limited by max charge power
        return min(self.max_charge_power, capacity_limited_power)
    
    def idle(self) -> None:
        """Set battery to idle state (no charging or discharging)."""
        self.current_power = 0.0
    
    def get_energy_stored(self) -> float:
        """
        Get current energy stored in the battery.
        
        Returns:
            Energy stored in kWh
        """
        return self.current_soc * self.capacity
    
    def __repr__(self) -> str:
        return (
            f"BESS(name='{self.name}', capacity={self.capacity}kWh, "
            f"SOC={self.current_soc*100:.1f}%, "
            f"power={self.current_power:+.2f}kW, "
            f"strategy={self.control_strategy.value})"
        )

