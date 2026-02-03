"""Base optimizer: minimize cost subject to import <= contracted_power."""

from collections import defaultdict
from dataclasses import dataclass
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
    peak_power_kw: float
    total_energy_kwh: float
    total_cost: float


class BaseOptimizer:
    """Minimize cost. Grid import constrained by contracted power."""

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
        print("     Running base optimization (minimize cost)...")

        T = len(energy_forecasts)
        timesteps = list(range(T))
        chargers = self.port.chargers
        boats = list(energy_forecasts[0].boat_required_energy_kwh.keys())

        num_chargers = len(chargers)
        num_boats = len(boats)

        print(
            f"        {num_chargers} chargers, {num_boats} boats, "
            f"{T} timesteps, contracted_power={self.port.contracted_power} kW"
        )

        model = Model("base_optimizer")
        model.hideOutput()
        model.setRealParam("limits/time", 30.0)

        # ----------------------------------------------------
        # Extract deadlines and trip durations
        # ----------------------------------------------------
        deadlines = self._extract_deadlines(energy_forecasts)
        print(f"        Deadlines: {deadlines}")

        trip_durations = self._extract_trip_durations(energy_forecasts)
        print(f"        Trip durations: {trip_durations}")

        # ----------------------------------------------------
        # Decision variables
        # p[c][b][t] = power (kW) delivered from charger c to boat b at time t
        # ----------------------------------------------------
        p = {}
        for c_idx, charger in enumerate(chargers):
            p[c_idx] = {}
            for b in boats:
                p[c_idx][b] = {}
                for t in timesteps:
                    p[c_idx][b][t] = model.addVar(
                        name=f"p_{c_idx}_{b}_{t}",
                        vtype="C",
                        lb=0.0,
                        ub=charger.max_power,
                    )

        # ----------------------------------------------------
        # Battery SOC variables (in kWh)
        # ----------------------------------------------------
        soc = {}
        boat_objects = {}  # Store boat objects for easy access

        for b in boats:
            boat_obj = next(boat for boat in self.port.boats if boat.name == b)
            boat_objects[b] = boat_obj
            battery_capacity = boat_obj.battery_capacity  # in kWh

            soc[b] = {}
            for t in timesteps:
                soc[b][t] = model.addVar(
                    name=f"soc_{b}_{t}",
                    vtype="C",
                    lb=0.0,
                    ub=battery_capacity,
                )

        # ----------------------------------------------------
        # Grid import
        # ----------------------------------------------------
        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_{t}",
                vtype="C",
                lb=0.0,
                ub=self.port.contracted_power,
            )

        # ----------------------------------------------------
        # PV production
        # ----------------------------------------------------
        pv_power = {}
        for t in timesteps:
            pv_power[t] = (
                energy_forecasts[t].power_active_production_kw
                if self.port.pv_systems
                else 0.0
            )

        pv_used = {}
        for t in timesteps:
            pv_used[t] = model.addVar(
                name=f"pv_used_{t}",
                vtype="C",
                lb=0.0,
                ub=pv_power[t],
            )

        # ----------------------------------------------------
        # Tariffs
        # ----------------------------------------------------
        tariff_price = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff_price[t] = self.port.get_tariff_price(ts)

        # ----------------------------------------------------
        # CONSTRAINTS
        # ----------------------------------------------------

        # Charger capacity constraints
        for c_idx, charger in enumerate(chargers):
            for t in timesteps:
                model.addCons(
                    quicksum(p[c_idx][b][t] for b in boats) <= charger.max_power,
                    name=f"charger_cap_{c_idx}_{t}",
                )

        # Power balance
        for t in timesteps:
            total_charger_power = quicksum(
                p[c_idx][b][t] for c_idx in range(num_chargers) for b in boats
            )

            model.addCons(
                grid_import[t] + pv_used[t] == total_charger_power,
                name=f"balance_{t}",
            )

        # ----------------------------------------------------
        # NEW: SOC CONSTRAINTS
        # ----------------------------------------------------

        # 1. Initial SOC constraints (use current boat SOC)
        for b in boats:
            boat_obj = boat_objects[b]
            initial_soc_kwh = boat_obj.soc * boat_obj.battery_capacity
            model.addCons(soc[b][0] == initial_soc_kwh, name=f"initial_soc_{b}")
            print(
                f"        {b}: Initial SOC = {initial_soc_kwh:.1f} kWh ({boat_obj.soc:.1%})"
            )

        # 2. SOC dynamics (battery balance over time)
        charging_efficiency = 0.95  # Typical charging efficiency

        for b in boats:
            for t in range(1, T):
                # Check if boat is available (at port, not on trip)
                available = energy_forecasts[t - 1].boat_available.get(b, 1)

                if available == 1:
                    # Boat is at port, can charge
                    total_charging = quicksum(
                        p[c_idx][b][t - 1] for c_idx in range(num_chargers)
                    )

                    model.addCons(
                        soc[b][t]
                        == soc[b][t - 1]
                        + total_charging * charging_efficiency * self.timestep_hours,
                        name=f"soc_charge_{b}_{t}",
                    )
                else:
                    # Boat is on trip - SOC remains constant during this timestep
                    # (energy consumption happens at trip start, handled by deadline constraints)
                    model.addCons(soc[b][t] == soc[b][t - 1], name=f"soc_trip_{b}_{t}")

        # 3. Deadline constraints (boats must have required energy before trips)
        for boat, deadline_list in deadlines.items():
            for deadline_t, required_energy_kwh in deadline_list:
                if deadline_t < T:
                    model.addCons(
                        soc[boat][deadline_t] >= required_energy_kwh,
                        name=f"deadline_{boat}_{deadline_t}",
                    )
                    print(
                        f"        Deadline: {boat} must have {required_energy_kwh:.1f} kWh by t={deadline_t}"
                    )

        # 4. No charging while boat is on trip
        for b in boats:
            for t in timesteps:
                available = energy_forecasts[t].boat_available.get(b, 1)
                if available == 0:
                    # Boat is sailing, cannot charge
                    for c_idx in range(num_chargers):
                        model.addCons(
                            p[c_idx][b][t] == 0.0,
                            name=f"no_charge_sailing_{c_idx}_{b}_{t}",
                        )

        # ----------------------------------------------------
        # Objective: minimize cost
        # ----------------------------------------------------
        total_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )
        model.setObjective(total_cost, "minimize")

        # ----------------------------------------------------
        # Solve
        # ----------------------------------------------------
        print("        Solving optimization problem...")
        model.optimize()
        status = model.getStatus()
        print(f"        SCIP status: {status}")

        if status not in ["optimal", "bestsollimit"]:
            print(f"        WARNING: Optimization status is {status}")
            # Still try to extract solution if one exists
            if model.getNSols() == 0:
                raise Exception(f"Optimization failed with status: {status}")

        # ----------------------------------------------------
        # Extract results
        # ----------------------------------------------------
        charger_schedules = {c.name: [] for c in chargers}
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        for t in timesteps:
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)

            power_this_t = 0.0
            for c_idx, charger in enumerate(chargers):
                p_c_t = sum(max(0.0, model.getVal(p[c_idx][b][t])) for b in boats)
                charger_schedules[charger.name].append((timestamp, p_c_t))
                power_this_t += p_c_t

            peak_power = max(peak_power, power_this_t)
            total_energy += power_this_t * self.timestep_hours

            g = max(0.0, model.getVal(grid_import[t]))
            total_cost_val += g * tariff_price[t] * self.timestep_hours

        print("     Base optimization complete")
        print(
            f"       Peak: {peak_power:.1f} kW, "
            f"Energy: {total_energy:.1f} kWh, "
            f"Cost: {total_cost_val:.2f}"
        )

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
            total_cost=total_cost_val,
        )

    # --------------------------------------------------------
    # Deadline extraction
    # --------------------------------------------------------
    def _extract_deadlines(self, energy_forecasts: List[EnergyForecast]):
        """
        Extract deadline constraints from energy forecasts.

        A deadline occurs when boat_required_energy_kwh drops from >0 to 0,
        indicating the boat is about to depart on a trip.

        Returns:
            Dict[str, List[Tuple[int, float]]]
            {
                boat_name: [
                    (timestep, required_energy_kwh),
                    ...
                ]
            }
        """
        deadlines = defaultdict(list)
        T = len(energy_forecasts)
        boats = energy_forecasts[0].boat_required_energy_kwh.keys()

        for boat in boats:
            for t in range(T - 1):
                req_now = energy_forecasts[t].boat_required_energy_kwh.get(boat, 0.0)
                req_next = energy_forecasts[t + 1].boat_required_energy_kwh.get(
                    boat, 0.0
                )

                # Deadline: boat needs energy now, but won't need it next timestep
                # (meaning it's about to leave on a trip)
                if req_now > 0 and req_next == 0:
                    deadlines[boat].append((t, req_now))

        return deadlines

    # --------------------------------------------------------
    # Extract trip durations from forecast
    # --------------------------------------------------------
    def _extract_trip_durations(self, energy_forecasts: List[EnergyForecast]):
        """
        Extract trip start times and durations (in timesteps) from boat_available.

        A trip is detected when boat_available changes from 1 -> 0.
        The trip duration is the number of consecutive timesteps with value 0.

        Returns:
            Dict[str, List[Tuple[int, int]]]
            {
                boat_name: [
                    (trip_start_t, duration_in_timesteps),
                    ...
                ]
            }
        """
        trip_durations: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        T = len(energy_forecasts)
        boats = energy_forecasts[0].boat_available.keys()

        for boat in boats:
            t = 0
            while t < T:
                prev_available = (
                    energy_forecasts[t - 1].boat_available.get(boat, 1) if t > 0 else 1
                )
                curr_available = energy_forecasts[t].boat_available.get(boat, 1)

                if prev_available == 1 and curr_available == 0:
                    start_t = t
                    duration = 0

                    while (
                        t < T and energy_forecasts[t].boat_available.get(boat, 1) == 0
                    ):
                        duration += 1
                        t += 1

                    trip_durations[boat].append((start_t - 1, duration))
                else:
                    t += 1

        return trip_durations

    # --------------------------------------------------------
    # Save schedules
    # --------------------------------------------------------
    def save_schedules_to_db(self, result: BaseOptimizationResult) -> None:
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
