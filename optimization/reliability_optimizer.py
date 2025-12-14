"""Reliability-focused port energy optimization using SCIP.

This optimizer uses mixed-integer linear programming to maximize charging power
while respecting contracted power limits.

Key features:
1. SCIP-based optimization
2. Objective: Maximize charging power when boats are available
3. Hard constraint: Grid import <= contracted power
4. Use PV and BESS to enable more charging
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pyscipopt import Model, quicksum
from models import Port, Boat, BoatState
from database import DatabaseManager
from forecasting import EnergyForecast


# ===========================================================================
# OPTIMIZATION PARAMETERS
# ===========================================================================

# Objective weights
WEIGHT_CHARGING = 100.0  # Maximize charging power
WEIGHT_EARLY = 10.0  # Prefer charging earlier
WEIGHT_PRIORITY = 50.0  # Extra weight for boats with low SOC


@dataclass
class ReliabilityOptimizationResult:
    """Result from reliability optimization."""
    
    status: str
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]
    bess_schedules: Dict[str, List[Tuple[datetime, float]]]
    boats_ready_on_time: List[str]
    boats_delayed: List[str]
    boats_cancelled: List[str]
    peak_power_kw: float
    total_energy_kwh: float


class ReliabilityOptimizer:
    """SCIP-based optimizer for maximum charging power."""
    
    def __init__(
        self,
        port: Port,
        db_manager: DatabaseManager,
        timestep_seconds: int = 900
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0
    
    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
        trip_assignments: Dict[str, List]
    ) -> ReliabilityOptimizationResult:
        """
        Optimize charging schedule using SCIP.
        
        Simple model:
        - Maximize total charging power
        - Subject to: grid <= contracted power
        - Use PV to enable more charging
        """
        print("     🎯 Running SCIP reliability optimization...")
        
        T = len(energy_forecasts)
        timesteps = list(range(T))
        
        # Create SCIP model
        model = Model("reliability_charging")
        model.hideOutput()
        model.setRealParam('limits/time', 30.0)
        
        num_chargers = len(self.port.chargers)
        charger_max = self.port.chargers[0].max_power if self.port.chargers else 22.0
        
        print(f"        {num_chargers} chargers, {T} timesteps")
        
        # ===================================================================
        # DECISION VARIABLES
        # ===================================================================
        
        # Charger power at each timestep
        charger_power = {}
        for c_idx in range(num_chargers):
            charger_power[c_idx] = {}
            for t in timesteps:
                charger_power[c_idx][t] = model.addVar(
                    name=f"p_{c_idx}_{t}",
                    vtype="C",
                    lb=0,
                    ub=self.port.chargers[c_idx].max_power
                )
        
        # Grid import at each timestep
        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_{t}",
                vtype="C",
                lb=0,
                ub=self.port.contracted_power
            )
        
        # BESS discharge at each timestep (if BESS available)
        bess_discharge = {}
        if self.port.bess_systems:
            for t in timesteps:
                max_discharge = sum(b.max_discharge_power for b in self.port.bess_systems)
                bess_discharge[t] = model.addVar(
                    name=f"bess_{t}",
                    vtype="C",
                    lb=0,
                    ub=max_discharge
                )
        
        # ===================================================================
        # PRE-COMPUTE
        # ===================================================================
        
        # Count available boats at each timestep
        boats_available = {}
        for t in timesteps:
            forecast = energy_forecasts[t]
            count = 0
            for boat in self.port.boats:
                state = forecast.boat_states.get(boat.name, BoatState.IDLE)
                if state != BoatState.SAILING:
                    count += 1
            boats_available[t] = count
        
        # PV power at each timestep
        pv_power = {}
        for t in timesteps:
            forecast = energy_forecasts[t]
            pv_power[t] = forecast.power_active_production_kwh / self.timestep_hours if self.timestep_hours > 0 else 0
        
        # ===================================================================
        # CONSTRAINTS
        # ===================================================================
        
        for t in timesteps:
            # Total charger power
            total_charger = quicksum(charger_power[c][t] for c in range(num_chargers))
            
            # CONSTRAINT 1: Power balance
            # Grid + PV + BESS >= Chargers
            if self.port.bess_systems:
                model.addCons(
                    grid_import[t] + pv_power[t] + bess_discharge[t] >= total_charger,
                    name=f"balance_{t}"
                )
            else:
                model.addCons(
                    grid_import[t] + pv_power[t] >= total_charger,
                    name=f"balance_{t}"
                )
            
            # CONSTRAINT 2: Grid import limit (already in bounds, but explicit)
            model.addCons(
                grid_import[t] <= self.port.contracted_power,
                name=f"grid_limit_{t}"
            )
            
            # CONSTRAINT 3: Only use chargers if boats are available
            # Limit active chargers to number of available boats
            max_active = min(num_chargers, boats_available[t])
            if max_active < num_chargers:
                # Turn off excess chargers
                for c in range(max_active, num_chargers):
                    model.addCons(
                        charger_power[c][t] == 0,
                        name=f"no_boat_{c}_{t}"
                    )
        
        # CONSTRAINT 4: BESS energy limit (simplified)
        if self.port.bess_systems:
            total_bess_energy = sum(b.capacity * b.current_soc for b in self.port.bess_systems)
            total_discharge = quicksum(bess_discharge[t] * self.timestep_hours for t in timesteps)
            model.addCons(
                total_discharge <= total_bess_energy * 0.8,  # Use up to 80% of BESS
                name="bess_energy_limit"
            )
        
        # ===================================================================
        # OBJECTIVE: Maximize total charging, prefer early charging
        # ===================================================================
        
        total_charging = quicksum(
            charger_power[c][t] * (1 + (T - t) / T * 0.1)  # Small bonus for earlier
            for c in range(num_chargers)
            for t in timesteps
        )
        
        model.setObjective(total_charging, "maximize")
        
        # ===================================================================
        # SOLVE
        # ===================================================================
        
        model.optimize()
        status = model.getStatus()
        print(f"        SCIP status: {status}")
        
        # ===================================================================
        # EXTRACT RESULTS
        # ===================================================================
        
        charger_schedules = {c.name: [] for c in self.port.chargers}
        bess_schedules = {b.name: [] for b in self.port.bess_systems}
        
        peak_power = 0.0
        total_energy = 0.0
        
        if status in ['optimal', 'bestsollimit', 'timelimit']:
            try:
                for t in timesteps:
                    timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                    power_this_t = 0.0
                    
                    for c_idx, charger in enumerate(self.port.chargers):
                        power_val = max(0, model.getVal(charger_power[c_idx][t]))
                        charger_schedules[charger.name].append((timestamp, power_val))
                        power_this_t += power_val
                    
                    peak_power = max(peak_power, power_this_t)
                    total_energy += power_this_t * self.timestep_hours
                
                # BESS schedules
                for bess in self.port.bess_systems:
                    for t in timesteps:
                        timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                        if self.port.bess_systems:
                            bess_val = model.getVal(bess_discharge[t]) if t in bess_discharge else 0
                        else:
                            bess_val = 0
                        bess_schedules[bess.name].append((timestamp, bess_val))
                
                print(f"     ✓ SCIP optimization complete")
                print(f"       Peak power: {peak_power:.1f} kW, Energy: {total_energy:.1f} kWh")
                
            except Exception as e:
                print(f"     ⚠️ Error: {e}, using fallback")
                return self._create_aggressive_fallback(forecast_date, energy_forecasts)
        else:
            print(f"     ⚠️ SCIP failed ({status}), using fallback")
            return self._create_aggressive_fallback(forecast_date, energy_forecasts)
        
        return ReliabilityOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            boats_ready_on_time=[],
            boats_delayed=[],
            boats_cancelled=[],
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy
        )
    
    def _create_aggressive_fallback(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast]
    ) -> ReliabilityOptimizationResult:
        """Aggressive fallback if SCIP fails."""
        T = len(energy_forecasts)
        charger_power = self.port.chargers[0].max_power if self.port.chargers else 22.0
        max_chargers = min(len(self.port.chargers), int(self.port.contracted_power / charger_power))
        
        charger_schedules = {}
        for c_idx, charger in enumerate(self.port.chargers):
            charger_schedules[charger.name] = []
            for t in range(T):
                timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                power = charger.max_power if c_idx < max_chargers else 0.0
                charger_schedules[charger.name].append((timestamp, power))
        
        bess_schedules = {}
        for bess in self.port.bess_systems:
            bess_schedules[bess.name] = [(
                forecast_date + timedelta(seconds=t * self.timestep_seconds), 0.0
            ) for t in range(T)]
        
        return ReliabilityOptimizationResult(
            status="fallback",
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            boats_ready_on_time=[],
            boats_delayed=[],
            boats_cancelled=[],
            peak_power_kw=max_chargers * charger_power,
            total_energy_kwh=max_chargers * charger_power * T * self.timestep_hours
        )
    
    def save_schedules_to_db(self, result: ReliabilityOptimizationResult) -> None:
        """Save schedules to database."""
        schedules = []
        power_setpoint_met = self.db_manager.get_metric_id("power_setpoint")
        
        for charger_name, schedule in result.charger_schedules.items():
            charger_src = self.db_manager.get_or_create_source(charger_name, "charger")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, charger_src, power_setpoint_met, str(power)))
        
        for bess_name, schedule in result.bess_schedules.items():
            bess_src = self.db_manager.get_or_create_source(bess_name, "bess")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, bess_src, power_setpoint_met, str(power)))
        
        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     ✓ Saved {len(schedules)} schedule entries")
