"""Base optimizer: minimize cost subject to import <= contracted_power."""

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
    peak_power_kw: float
    total_energy_kwh: float
    total_cost: float


class BaseOptimizer:
    """Minimize cost. Single constraint: grid import <= contracted_power."""

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
        """Minimize cost (grid × tariff) s.t. grid_import <= contracted_power."""
        print("     Running base optimization (minimize cost)...")

        T = len(energy_forecasts)
        timesteps = list(range(T))
        num_chargers = len(self.port.chargers)
        print(
            f"        {num_chargers} chargers, {T} timesteps, contracted_power={self.port.contracted_power} kW"
        )

        model = Model("base_optimizer")
        model.hideOutput()
        model.setRealParam("limits/time", 30.0)

        # Decision variables
        charger_power = {}
        for c_idx in range(num_chargers):
            charger_power[c_idx] = {}
            ub = self.port.chargers[c_idx].max_power
            for t in timesteps:
                charger_power[c_idx][t] = model.addVar(
                    name=f"p_{c_idx}_{t}", vtype="C", lb=0, ub=ub
                )

        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_{t}", vtype="C", lb=0, ub=self.port.contracted_power
            )

        # Per boat (source): max required energy over timesteps; sum those, then × 2
        per_boat_kwh = {}
        for t in timesteps:
            for boat, kwh in energy_forecasts[t].boat_required_energy_kwh.items():
                per_boat_kwh[boat] = max(per_boat_kwh.get(boat, 0), kwh)
        energy_required_kwh = sum(per_boat_kwh.values()) * 2

        energy_required_boat_kwh = sum(b.battery_capacity * 2 for b in self.port.boats)

        print(f"Energy required: {energy_required_boat_kwh} kWh")

        # at least 2× (sum of energy required per boat)
        model.addCons(
            quicksum(
                charger_power[c_idx][t] * self.timestep_hours
                for c_idx in range(num_chargers)
                for t in timesteps
            )
            >= energy_required_boat_kwh,
            name="min_total_energy",
        )

        # Pre-compute
        pv_power = {}
        if self.port.pv_systems:
            for t in timesteps:
                pv_power[t] = energy_forecasts[t].power_active_production_kw
        else:
            for t in timesteps:
                pv_power[t] = 0.0

        tariff_price = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff_price[t] = self.port.get_tariff_price(ts)

        # Power balance: grid + PV == sum(charger_power)
        for t in timesteps:
            charger_demand = quicksum(charger_power[c][t] for c in range(num_chargers))
            model.addCons(
                grid_import[t] + pv_power[t] == charger_demand, name=f"balance_{t}"
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
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        if status in ["optimal", "bestsollimit", "timelimit"]:
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
                    peak_power = max(peak_power, power_this_t)
                    total_energy += power_this_t * self.timestep_hours
                    g = max(0, model.getVal(grid_import[t]))
                    total_cost_val += g * tariff_price[t] * self.timestep_hours

                print("     Base optimization complete")
                print(
                    f"       Peak: {peak_power:.1f} kW, Energy: {total_energy:.1f} kWh, Cost: {total_cost_val:.2f}"
                )
            except Exception as e:
                print(f"     Error extracting results: {e}, using fallback")
                return self._create_fallback(forecast_date, energy_forecasts)
        else:
            print(f"     SCIP failed ({status}), using fallback")
            return self._create_fallback(forecast_date, energy_forecasts)

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
            total_cost=total_cost_val,
        )

    def _create_fallback(
        self, forecast_date: datetime, energy_forecasts: List[EnergyForecast]
    ) -> BaseOptimizationResult:
        """Fallback if SCIP fails."""
        T = len(energy_forecasts)
        pwr = self.port.chargers[0].max_power if self.port.chargers else 22.0
        max_chargers = min(
            len(self.port.chargers),
            int(self.port.contracted_power / pwr),
        )
        total_pwr = max_chargers * pwr

        charger_schedules = {}
        for c_idx, charger in enumerate(self.port.chargers):
            charger_schedules[charger.name] = []
            for t in range(T):
                ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                charger_schedules[charger.name].append(
                    (ts, charger.max_power if c_idx < max_chargers else 0.0)
                )

        cost = 0.0
        for t in range(T):
            pv = energy_forecasts[t].power_active_production_kw
            grid = max(0, total_pwr - pv)
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            cost += grid * self.port.get_tariff_price(ts) * self.timestep_hours

        return BaseOptimizationResult(
            status="fallback",
            charger_schedules=charger_schedules,
            peak_power_kw=total_pwr,
            total_energy_kwh=total_pwr * T * self.timestep_hours,
            total_cost=cost,
        )

    def _v2gStateEq(self, boat_available, boat, t):
        if boat_available[boat, t] == 0:  # If vehicle is not scheduled
            return boat.soc[t] == 0 # valor que tinha - custo da viagem
        elif t > 1:  # If not the first time step
            if (boat_available[boat, t - 1] == 1) & (
                boat_available[boat, t] == 1
            ):  # If was and is currently connected
                return (
                    boat.soc[t]
                    == boat.soc[t - 1]
                    + boat.charge[t] * boat.charger.efficiency
                    - boat.discharge[t] / boat.charger.efficiency
                )
            elif (boat_available[boat, t - 1] == 0) & (
                boat_available[boat, t] == 1
            ):  # If became connected
                return (
                    boat.soc[t]
                    == # soc do departure - custo da viagem
                    boat.soc[t - 1] - boat.discharge[t] / boat.charger.efficiency
                    + boat.charge[t] * boat.charger.efficiency
                    - boat.discharge[t] / boat.charger.efficiency
                )
        return None

    def save_schedules_to_db(self, result: BaseOptimizationResult) -> None:
        """Save schedules to database."""
        schedules = []
        power_setpoint_met = self.db_manager.get_metric_id("power_setpoint")

        for charger_name, schedule in result.charger_schedules.items():
            charger_src = self.db_manager.get_or_create_source(charger_name, "charger")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, charger_src, power_setpoint_met, str(power)))

        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} schedule entries")
