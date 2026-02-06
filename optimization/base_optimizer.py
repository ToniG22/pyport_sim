"""Base optimizer: minimize cost subject to import <= contracted_power."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from pyscipopt import Model, quicksum

from database import DatabaseManager
from forecasting import EnergyForecast
from models import Port


@dataclass
class BaseOptimizationResult:
    status: str
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]
    boat_schedules: Dict[str, List[Tuple[datetime, float]]]
    peak_power_kw: float
    total_energy_kwh: float
    total_cost: float
    bess_schedules: Dict[str, List[Tuple[datetime, float]]] = field(
        default_factory=dict
    )  # power: + charge, - discharge


class BaseOptimizer:
    """Minimize cost. Single constraint: grid import <= contracted_power."""

    def __init__(
        self,
        port: Port,
        db_manager: DatabaseManager,
        timestep_seconds: int = 900,
        boat_charger_assignments: Dict[str, int] = None,
        trip_schedule: tuple = ((9, 0), (14, 1)),
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0
        self.boat_charger_assignments = boat_charger_assignments or {}
        self.trip_schedule = trip_schedule

    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
    ) -> BaseOptimizationResult:
        """Minimize cost (grid x tariff) s.t. grid_import <= contracted_power."""
        print("     Running base optimization (minimize cost)...")

        T = len(energy_forecasts)
        timesteps = list(range(T))
        num_chargers = len(self.port.chargers)
        print(
            f"        {num_chargers} chargers, {T} timesteps, contracted_power={self.port.contracted_power} kW"
        )

        model = Model("base_optimizer")
        model.hideOutput()
        # model.setRealParam("limits/time", 30.0)

        # Grid import
        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_{t}", vtype="C", lb=0, ub=self.port.contracted_power
            )

        # Charger power
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

        # Pre-compute PV forecast
        pv_power = {}
        for t in timesteps:
            pv_power[t] = (
                energy_forecasts[t].power_active_production_kw
                if self.port.pv_systems
                else 0.0
            )

        # PV usage variables (allows curtailment)
        pv_used = {}
        for t in timesteps:
            pv_used[t] = model.addVar(
                name=f"pv_used_{t}", vtype="C", lb=0, ub=pv_power[t]
            )

        # BESS power: positive = charging (load), negative = discharging (supply)
        bess_power = {}
        if self.port.bess_systems:
            for b_idx, bess in enumerate(self.port.bess_systems):
                bess_power[b_idx] = {}
                for t in timesteps:
                    bess_power[b_idx][t] = model.addVar(
                        name=f"bess_{b_idx}_{t}",
                        vtype="C",
                        lb=-bess.max_discharge_power,
                        ub=bess.max_charge_power,
                    )

        # Tariffs
        tariff_price = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff_price[t] = self.port.get_tariff_price(ts)

        # Power Balance: grid + PV + BESS discharge = chargers + BESS charge
        # (bess_power > 0 = charge = load, bess_power < 0 = discharge = supply)
        for t in timesteps:
            load = quicksum(charger_power[c][t] for c in range(num_chargers))
            if self.port.bess_systems:
                load = load + quicksum(
                    bess_power[b][t] for b in range(len(self.port.bess_systems))
                )
            model.addCons(
                grid_import[t] + pv_used[t] == load,
                name=f"power_balance_{t}",
            )

        # Objective: minimize cost
        total_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )
        model.setObjective(total_cost, "minimize")

        model.optimize()
        status = model.getStatus()
        print(f"        SCIP status: {status}")

        # Extract results
        charger_schedules = {c.name: [] for c in self.port.chargers}
        bess_schedules = {b.name: [] for b in self.port.bess_systems}
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        if status == "optimal":
            try:
                for t in timesteps:
                    timestamp = forecast_date + timedelta(
                        seconds=t * self.timestep_seconds
                    )
                    power_this_t = 0.0
                    for c_idx, charger in enumerate(self.port.chargers):
                        p = max(0, model.getVal(charger_power[c_idx][t]))
                        charger_schedules[charger.name].append((timestamp, p))
                        power_this_t += p
                    if self.port.bess_systems:
                        for b_idx, bess in enumerate(self.port.bess_systems):
                            pb = model.getVal(bess_power[b_idx][t])
                            bess_schedules.setdefault(bess.name, []).append(
                                (timestamp, pb)
                            )
                    peak_power = max(peak_power, power_this_t)
                    total_energy += power_this_t * self.timestep_hours
                    g = max(0, model.getVal(grid_import[t]))
                    total_cost_val += g * tariff_price[t] * self.timestep_hours

                print("     Base optimization complete")
                print(
                    f"       Peak: {peak_power:.1f} kW, Energy: {total_energy:.1f} kWh, Cost: {total_cost_val:.2f}"
                )
                # Build boat_schedules from charger_schedules (1:1 mapping)
                boat_schedules = {}
                for boat_name, c_idx in self.boat_charger_assignments.items():
                    charger_name = self.port.chargers[c_idx].name
                    boat_schedules[boat_name] = list(charger_schedules[charger_name])
                return BaseOptimizationResult(
                    status=status,
                    charger_schedules=charger_schedules,
                    boat_schedules=boat_schedules,
                    peak_power_kw=peak_power,
                    total_energy_kwh=total_energy,
                    total_cost=total_cost_val,
                    bess_schedules=bess_schedules,
                )
            except Exception as e:
                print(f"     Error extracting results: {e}, using fallback")
                raise e
        else:
            print(f"     SCIP failed ({status}), using fallback")
            raise RuntimeError(f"SCIP failed ({status})")

    def save_schedules_to_db(self, result: BaseOptimizationResult) -> None:
        """Save schedules to database."""
        schedules = []
        power_setpoint_met = self.db_manager.get_metric_id("power_setpoint")

        for charger_name, schedule in result.charger_schedules.items():
            charger_src = self.db_manager.get_or_create_source(charger_name, "charger")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, charger_src, power_setpoint_met, str(power)))

        # BESS: store as (positive=discharge, negative=charge) for engine
        for bess_name, schedule in result.bess_schedules.items():
            bess_src = self.db_manager.get_or_create_source(bess_name, "bess")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, bess_src, power_setpoint_met, str(-power)))

        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} schedule entries")
