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

        T = len(energy_forecasts)
        timesteps = list(range(T))

        model = Model("base_optimizer")
        model.hideOutput()

        # --------------------------------------------------
        # PLACEHOLDER TRIPS (per boat)
        # --------------------------------------------------
        TRIPS = [
            {"t": 35, "energy": 64.0},  # trip 1
            {"t": 55, "energy": 64.0},  # trip 2
        ]

        # --------------------------------------------------
        # Grid import
        # --------------------------------------------------
        grid_import = {
            t: model.addVar(lb=0, ub=self.port.contracted_power, name=f"grid_{t}")
            for t in timesteps
        }

        # --------------------------------------------------
        # Charger power variables
        # --------------------------------------------------
        charger_power = {}
        for charger in self.port.chargers:
            charger_power[charger.name] = {
                t: model.addVar(
                    lb=0, ub=charger.max_power, name=f"p_{charger.name}_{t}"
                )
                for t in timesteps
            }

        # --------------------------------------------------
        # Power balance
        # --------------------------------------------------
        for t in timesteps:
            model.addCons(
                grid_import[t]
                == quicksum(charger_power[c.name][t] for c in self.port.chargers),
                name=f"power_balance_{t}",
            )

        # --------------------------------------------------
        # Boat ↔ charger mapping
        # --------------------------------------------------
        charger_to_boat = {
            self.port.chargers[idx].name: boat
            for boat, idx in self.boat_charger_assignments.items()
        }

        # --------------------------------------------------
        # Trip decision binaries
        # --------------------------------------------------
        go_trip = {}  # go_trip[(boat, trip_idx)]

        for boat in self.boat_charger_assignments:
            for i in range(len(TRIPS)):
                go_trip[(boat, i)] = model.addVar(
                    vtype="B", name=f"go_{boat}_trip{i+1}"
                )

        BIG_M = 1e4  # kWh, safely large

        # --------------------------------------------------
        # Trip energy constraints (cumulative)
        # --------------------------------------------------
        for charger in self.port.chargers:
            charger_name = charger.name
            boat = charger_to_boat[charger_name]

            cumulative_energy = 0.0

            for i, trip in enumerate(TRIPS):
                cumulative_energy += trip["energy"]
                t_dep = trip["t"]

                model.addCons(
                    quicksum(
                        charger_power[charger_name][t] * self.timestep_hours
                        for t in range(t_dep)
                    )
                    >= cumulative_energy * go_trip[(boat, i)]
                    - BIG_M * (1 - go_trip[(boat, i)]),
                    name=f"trip_{boat}_{i+1}",
                )

        # --------------------------------------------------
        # Tariffs
        # --------------------------------------------------
        tariff_price = {
            t: self.port.get_tariff_price(
                forecast_date + timedelta(seconds=t * self.timestep_seconds)
            )
            for t in timesteps
        }

        energy_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )

        # --------------------------------------------------
        # Objective: maximize trips, then minimize cost
        # --------------------------------------------------
        trip_reward = quicksum(go_trip.values())

        model.setObjective(
            -1000 * trip_reward + energy_cost,  # lexicographic via scaling
            "minimize",
        )

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------
        model.optimize()
        status = model.getStatus()

        if status != "optimal":
            raise RuntimeError(f"SCIP failed ({status})")

        # --------------------------------------------------
        # Extract schedules (unchanged from your code)
        # --------------------------------------------------
        charger_schedules = {c.name: [] for c in self.port.chargers}
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            power_t = 0.0
            for c in self.port.chargers:
                p = max(0, model.getVal(charger_power[c.name][t]))
                charger_schedules[c.name].append((ts, p))
                power_t += p
            peak_power = max(peak_power, power_t)
            total_energy += power_t * self.timestep_hours
            total_cost_val += (
                model.getVal(grid_import[t]) * tariff_price[t] * self.timestep_hours
            )

        boat_schedules = {}
        for boat, idx in self.boat_charger_assignments.items():
            charger_name = self.port.chargers[idx].name
            boat_schedules[boat] = list(charger_schedules[charger_name])

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            boat_schedules=boat_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
            total_cost=total_cost_val,
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

        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} schedule entries")
