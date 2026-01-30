"""Base optimizer: cost vs SOC tradeoff with contracted_power limit.

Minimizes cost minus reward for fuller boat SOC (using port tariff).
Subject to grid <= contracted_power.
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
    total_cost: float


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
        Minimize cost minus reward for fuller boat SOC, subject to grid <= contracted_power.

        Objective: cost (grid × tariff) - soc_reward_per_kwh × total_energy_delivered.
        Rewards charging boats closer to 100% SOC. Single constraint: grid <= contracted_power.

        Args:
            forecast_date: Date to optimize for
            energy_forecasts: Energy forecasts for the day

        Returns:
            BaseOptimizationResult with optimal schedules
        """
        print(
            "     🎯 Running base optimization (cost min + SOC reward)..."
        )

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
        charger_on = {}  # binary: 1 if charger has power > 0
        max_charger_power = (
            max(c.max_power for c in self.port.chargers) if self.port.chargers else 22.0
        )
        for c_idx in range(num_chargers):
            charger_power[c_idx] = {}
            charger_on[c_idx] = {}
            for t in timesteps:
                charger_power[c_idx][t] = model.addVar(
                    name=f"p_{c_idx}_{t}",
                    vtype="C",
                    lb=0,
                    ub=self.port.chargers[c_idx].max_power,
                )
                charger_on[c_idx][t] = model.addVar(
                    name=f"on_{c_idx}_{t}", vtype="B", lb=0, ub=1
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

        # Tariff price per kWh at each timestep (from port)
        tariff_price = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff_price[t] = self.port.get_tariff_price(ts)

        # Boats needing charge at each t (available + have required energy > 0)
        def _boats_needing_at(t_idx: int) -> int:
            if not energy_forecasts or t_idx >= len(energy_forecasts):
                return 0
            fc = energy_forecasts[t_idx]
            return sum(
                1
                for b, need in fc.boat_required_energy_kwh.items()
                if need > 0 and fc.boat_available.get(b, 0) == 1
            )

        # ===================================================================
        # CONSTRAINTS
        # ===================================================================

        for t in timesteps:
            total_charger = quicksum(charger_power[c][t] for c in range(num_chargers))

            # Power balance: grid + PV + BESS >= charger demand
            if self.port.bess_systems:
                model.addCons(
                    grid_import[t] + pv_power[t] + bess_discharge[t] >= total_charger,
                    name=f"balance_{t}",
                )
            else:
                model.addCons(
                    grid_import[t] + pv_power[t] >= total_charger, name=f"balance_{t}"
                )

            # Single constraint: grid must not exceed contracted_power
            model.addCons(
                grid_import[t] <= self.port.contracted_power,
                name=f"contracted_power_{t}",
            )

            # Link charger_on to charger_power (0.01 kW min when on)
            for c in range(num_chargers):
                model.addCons(
                    charger_power[c][t] <= max_charger_power * charger_on[c][t],
                    name=f"link_ub_{c}_{t}",
                )
                model.addCons(
                    charger_power[c][t] >= 0.01 * charger_on[c][t],
                    name=f"link_lb_{c}_{t}",
                )

            # At least N chargers on when N boats need charge (enables parallel charging)
            n_needing = _boats_needing_at(t)
            if n_needing > 0:
                model.addCons(
                    quicksum(charger_on[c][t] for c in range(num_chargers))
                    >= min(n_needing, num_chargers),
                    name=f"parallel_{t}",
                )

        # Must meet boat energy demand (otherwise cost min → all zeros, no charging)
        demand_from_forecast = (
            sum(energy_forecasts[0].boat_required_energy_kwh.values())
            if energy_forecasts
            else 0.0
        )
        if demand_from_forecast > 0:
            model.addCons(
                quicksum(
                    charger_power[c][t]
                    for c in range(num_chargers)
                    for t in timesteps
                )
                * self.timestep_hours
                >= demand_from_forecast,
                name="min_demand",
            )

        # ===================================================================
        # OBJECTIVE: Minimize cost - reward for fuller SOC
        # ===================================================================

        # Cost = sum over t of (grid_import[t] kW × price[t] €/kWh × dt h)
        total_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours
            for t in timesteps
        )
        # Reward: value assigned to each kWh delivered to boats (fuller SOC)
        total_energy = quicksum(
            charger_power[c][t] for c in range(num_chargers) for t in timesteps
        ) * self.timestep_hours
        soc_reward = 0.15 * total_energy  # built-in: reward fuller SOC (€/kWh)
        # Minimize cost minus reward → prefers cheaper charging AND fuller boats
        model.setObjective(total_cost - soc_reward, "minimize")

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
        total_cost = 0.0

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

                    grid_val = max(0, model.getVal(grid_import[t]))
                    total_cost += grid_val * tariff_price[t] * self.timestep_hours

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
                    f"       Peak power: {peak_power:.1f} kW, Energy: {total_energy:.1f} kWh, "
                    f"Cost: {total_cost:.2f}"
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
            total_cost=total_cost,
        )

    def _create_fallback(
        self, forecast_date: datetime, energy_forecasts: List[EnergyForecast]
    ) -> BaseOptimizationResult:
        """Fallback if SCIP fails - use max power up to contracted limit."""
        T = len(energy_forecasts)
        pwr_per_charger = (
            self.port.chargers[0].max_power if self.port.chargers else 22.0
        )
        max_chargers = min(
            len(self.port.chargers),
            int(self.port.contracted_power / pwr_per_charger),
        )
        total_charger_pwr = max_chargers * pwr_per_charger

        charger_schedules = {}
        fallback_cost = 0.0
        for c_idx, charger in enumerate(self.port.chargers):
            charger_schedules[charger.name] = []
            for t in range(T):
                timestamp = forecast_date + timedelta(
                    seconds=t * self.timestep_seconds
                )
                power = charger.max_power if c_idx < max_chargers else 0.0
                charger_schedules[charger.name].append((timestamp, power))

        for t in range(T):
            pv = energy_forecasts[t].power_active_production_kw
            grid = max(0, total_charger_pwr - pv)
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            fallback_cost += (
                grid * self.port.get_tariff_price(ts) * self.timestep_hours
            )

        bess_schedules = {}
        for bess in self.port.bess_systems:
            bess_schedules[bess.name] = [
                (
                    forecast_date + timedelta(seconds=t * self.timestep_seconds),
                    0.0,
                )
                for t in range(T)
            ]

        return BaseOptimizationResult(
            status="fallback",
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            peak_power_kw=total_charger_pwr,
            total_energy_kwh=total_charger_pwr * T * self.timestep_hours,
            total_cost=fallback_cost,
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
