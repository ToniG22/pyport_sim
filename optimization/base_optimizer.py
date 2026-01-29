"""Base optimizer with single constraint: do not exceed contracted_power.

This optimizer maximizes charging power while respecting the contracted power limit.
No other constraints are applied.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pyscipopt import Model, quicksum
from models import Port
from database import DatabaseManager
from forecasting import EnergyForecast


@dataclass
class BaseOptimizationResult:
    """Result from base optimization."""

    status: str
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]
    bess_schedules: Dict[str, List[Tuple[datetime, float]]]
    peak_power_kw: float
    total_energy_kwh: float


class BaseOptimizer:
    """Base optimizer with single constraint: contracted power limit."""

    def __init__(
        self, port: Port, db_manager: DatabaseManager, timestep_seconds: int = 900
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0

    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
    ) -> BaseOptimizationResult:
        """
        Optimize charging schedule with single constraint: do not exceed contracted_power.

        Args:
            forecast_date: Date to optimize for
            energy_forecasts: Energy forecasts for the day
            trip_assignments: Trip assignments per boat (not used in base optimizer)

        Returns:
            BaseOptimizationResult with optimal schedules
        """
        print("     🎯 Running base optimization (contracted power constraint only)...")

        T = len(energy_forecasts)
        timesteps = list(range(T))

        # Create SCIP model
        model = Model("base_optimizer")
        model.hideOutput()
        model.setRealParam("limits/time", 30.0)

        num_chargers = len(self.port.chargers)

        print(
            f"        {num_chargers} chargers, {T} timesteps, contracted_power={self.port.contracted_power} kW"
        )

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
                    ub=self.port.chargers[c_idx].max_power,
                )

        # Grid import at each timestep
        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_{t}", vtype="C", lb=0, ub=self.port.contracted_power
            )

        # BESS discharge at each timestep (if BESS available)
        bess_discharge = {}
        if self.port.bess_systems:
            for t in timesteps:
                max_discharge = sum(
                    b.max_discharge_power for b in self.port.bess_systems
                )
                bess_discharge[t] = model.addVar(
                    name=f"bess_{t}", vtype="C", lb=0, ub=max_discharge
                )

        # ===================================================================
        # PRE-COMPUTE
        # ===================================================================

        # PV power at each timestep (kW) from forecaster
        pv_power = {}
        for t in timesteps:
            pv_power[t] = energy_forecasts[t].power_active_production_kw

        # ===================================================================
        # CONSTRAINTS
        # ===================================================================

        for t in timesteps:
            # Total charger power
            total_charger = quicksum(charger_power[c][t] for c in range(num_chargers))

            # CONSTRAINT: Power balance
            # Grid + PV + BESS >= Chargers
            # If PV is greater add a relaxation
            if self.port.bess_systems:
                model.addCons(
                    grid_import[t] + pv_power[t] + bess_discharge[t] >= total_charger,
                    name=f"balance_{t}",
                )
            else:
                model.addCons(
                    grid_import[t] + pv_power[t] == total_charger, name=f"balance_{t}"
                )

            # CONSTRAINT: Do not exceed contracted_power
            # This is the ONLY constraint (besides power balance)
            model.addCons(
                grid_import[t] <= self.port.contracted_power,
                name=f"contracted_power_limit_{t}",
            )

        # ===================================================================
        # OBJECTIVE: Maximize total charging power
        # ===================================================================

        total_charging = quicksum(
            charger_power[c][t] for c in range(num_chargers) for t in timesteps
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

        if status in ["optimal", "bestsollimit", "timelimit"]:
            try:
                for t in timesteps:
                    timestamp = forecast_date + timedelta(
                        seconds=t * self.timestep_seconds
                    )
                    power_this_t = 0.0

                    for c_idx, charger in enumerate(self.port.chargers):
                        power_val = max(0, model.getVal(charger_power[c_idx][t]))
                        charger_schedules[charger.name].append((timestamp, power_val))
                        power_this_t += power_val

                    peak_power = max(peak_power, power_this_t)
                    total_energy += power_this_t * self.timestep_hours

                # BESS schedules
                if self.port.bess_systems:
                    for bess in self.port.bess_systems:
                        for t in timesteps:
                            timestamp = forecast_date + timedelta(
                                seconds=t * self.timestep_seconds
                            )
                            bess_val = (
                                model.getVal(bess_discharge[t])
                                if t in bess_discharge
                                else 0
                            )
                            bess_schedules[bess.name].append((timestamp, bess_val))

                print("     ✓ Base optimization complete")
                print(
                    f"       Peak power: {peak_power:.1f} kW, Energy: {total_energy:.1f} kWh"
                )

            except Exception as e:
                print(f"     ⚠️ Error extracting results: {e}, using fallback")
                return self._create_fallback(forecast_date, energy_forecasts)
        else:
            print(f"     ⚠️ SCIP failed ({status}), using fallback")
            return self._create_fallback(forecast_date, energy_forecasts)

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
        )

    def _create_fallback(
        self, forecast_date: datetime, energy_forecasts: List[EnergyForecast]
    ) -> BaseOptimizationResult:
        """Fallback if SCIP fails - use max power up to contracted limit."""
        T = len(energy_forecasts)
        charger_power = self.port.chargers[0].max_power if self.port.chargers else 22.0
        max_chargers = min(
            len(self.port.chargers), int(self.port.contracted_power / charger_power)
        )

        charger_schedules = {}
        for c_idx, charger in enumerate(self.port.chargers):
            charger_schedules[charger.name] = []
            for t in range(T):
                timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                power = charger.max_power if c_idx < max_chargers else 0.0
                charger_schedules[charger.name].append((timestamp, power))

        bess_schedules = {}
        for bess in self.port.bess_systems:
            bess_schedules[bess.name] = [
                (forecast_date + timedelta(seconds=t * self.timestep_seconds), 0.0)
                for t in range(T)
            ]

        return BaseOptimizationResult(
            status="fallback",
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            peak_power_kw=max_chargers * charger_power,
            total_energy_kwh=max_chargers * charger_power * T * self.timestep_hours,
        )

    def save_schedules_to_db(self, result: BaseOptimizationResult) -> None:
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
