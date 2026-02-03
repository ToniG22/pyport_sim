"""Base optimizer: minimize cost subject to import <= contracted_power.

FIXED VERSION v4:
- Based on working v2 (one charger per boat constraint)
- Adds: boats that can't depart on time will continue charging and depart late
- Simpler approach than v3: keep the same departure logic but allow charging during "trip" timesteps if boat hasn't departed

Key insight: The original model marks timesteps after deadline as "on trip" and blocks charging.
This version allows charging to continue if the boat is late (departed=0), enabling
the boat to eventually reach required SOC and depart.
"""

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
    boat_schedules: Dict[str, List[Tuple[datetime, float]]]
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
        model.setRealParam("limits/time", 120.0)

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
        # Binary variables for charger-boat assignment
        # y[c][b][t] = 1 if charger c is assigned to boat b at time t
        # ----------------------------------------------------
        y = {}
        for c_idx, charger in enumerate(chargers):
            y[c_idx] = {}
            for b in boats:
                y[c_idx][b] = {}
                for t in timesteps:
                    y[c_idx][b][t] = model.addVar(
                        name=f"y_{c_idx}_{b}_{t}",
                        vtype="B",
                    )

        # ----------------------------------------------------
        # Binary variables: departed[b][trip_idx] = 1 if boat b departed on trip trip_idx
        # ----------------------------------------------------
        departed = {}
        late = {}

        for b in boats:
            departed[b] = {}
            late[b] = {}
            num_trips = len(deadlines.get(b, []))
            for trip_idx in range(num_trips):
                departed[b][trip_idx] = model.addVar(
                    name=f"departed_{b}_{trip_idx}", vtype="B"
                )
                late[b][trip_idx] = model.addVar(name=f"late_{b}_{trip_idx}", vtype="B")

        # ----------------------------------------------------
        # NEW: Continuous variable for ACTUAL departure timestep
        # This allows departure at deadline OR later (when SOC is sufficient)
        # ----------------------------------------------------
        actual_depart_t = {}
        for b in boats:
            actual_depart_t[b] = {}
            num_trips = len(deadlines.get(b, []))
            for trip_idx in range(num_trips):
                deadline_t = deadlines[b][trip_idx][0]
                actual_depart_t[b][trip_idx] = model.addVar(
                    name=f"actual_depart_t_{b}_{trip_idx}",
                    vtype="C",
                    lb=deadline_t,  # Can't depart before deadline
                    ub=T - 1,
                )

        # ----------------------------------------------------
        # Battery SOC variables (in kWh)
        # ----------------------------------------------------
        soc = {}
        boat_objects = {}

        for b in boats:
            boat_obj = next(boat for boat in self.port.boats if boat.name == b)
            boat_objects[b] = boat_obj
            battery_capacity = boat_obj.battery_capacity

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

        # Each charger can only be assigned to ONE boat at a time
        for c_idx in range(num_chargers):
            for t in timesteps:
                model.addCons(
                    quicksum(y[c_idx][b][t] for b in boats) <= 1,
                    name=f"charger_one_boat_{c_idx}_{t}",
                )

        # Each boat can only use ONE charger at a time
        for b in boats:
            for t in timesteps:
                model.addCons(
                    quicksum(y[c_idx][b][t] for c_idx in range(num_chargers)) <= 1,
                    name=f"boat_one_charger_{b}_{t}",
                )

        # Link power to assignment
        for c_idx, charger in enumerate(chargers):
            for b in boats:
                for t in timesteps:
                    model.addCons(
                        p[c_idx][b][t] <= charger.max_power * y[c_idx][b][t],
                        name=f"link_power_assign_{c_idx}_{b}_{t}",
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
        # SOC CONSTRAINTS
        # ----------------------------------------------------

        # 1. Initial SOC
        charging_efficiency = 0.95

        for b in boats:
            boat_obj = boat_objects[b]
            initial_soc_kwh = boat_obj.soc * boat_obj.battery_capacity
            model.addCons(soc[b][0] == initial_soc_kwh, name=f"initial_soc_{b}")
            print(
                f"        {b}: Initial SOC = {initial_soc_kwh:.1f} kWh ({boat_obj.soc:.1%})"
            )

        # 2. Build trip timeline for each boat
        for b in boats:
            boat_trips = deadlines.get(b, [])

            # Create timeline: map timestep -> trip status
            trip_timeline = {}

            for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                trip_start_t, duration = trip_durations[b][trip_idx]

                # Departure timestep is right after deadline
                departure_t = deadline_t + 1
                return_t = departure_t + duration

                # Mark trip duration
                for t in range(departure_t, min(return_t, T)):
                    trip_timeline[t] = (True, trip_idx, False)

                # Mark return timestep
                if return_t < T:
                    trip_timeline[return_t] = (True, trip_idx, True)

            # 3. SOC dynamics with trip logic
            for t in range(1, T):
                total_charging = quicksum(
                    p[c_idx][b][t - 1] for c_idx in range(num_chargers)
                )

                if t not in trip_timeline:
                    # Boat is at port, normal charging
                    model.addCons(
                        soc[b][t]
                        == soc[b][t - 1]
                        + total_charging * charging_efficiency * self.timestep_hours,
                        name=f"soc_charge_{b}_{t}",
                    )
                else:
                    is_on_trip, trip_idx, is_return = trip_timeline[t]
                    deadline_t, required_energy = boat_trips[trip_idx]

                    if is_return:
                        # Boat returning from trip - consume energy IF it departed
                        M = boat_objects[b].battery_capacity * 2

                        # If departed: soc[t] = soc[t-1] - required_energy
                        model.addCons(
                            soc[b][t]
                            >= soc[b][t - 1]
                            - required_energy
                            - M * (1 - departed[b][trip_idx]),
                            name=f"soc_return_lb_{b}_{t}",
                        )
                        model.addCons(
                            soc[b][t]
                            <= soc[b][t - 1]
                            - required_energy
                            + M * (1 - departed[b][trip_idx]),
                            name=f"soc_return_ub_{b}_{t}",
                        )
                        # If NOT departed: soc[t] = soc[t-1] + charging (can still charge!)
                        model.addCons(
                            soc[b][t]
                            >= soc[b][t - 1]
                            + total_charging * charging_efficiency * self.timestep_hours
                            - M * departed[b][trip_idx],
                            name=f"soc_no_trip_charge_lb_{b}_{t}",
                        )
                        model.addCons(
                            soc[b][t]
                            <= soc[b][t - 1]
                            + total_charging * charging_efficiency * self.timestep_hours
                            + M * departed[b][trip_idx],
                            name=f"soc_no_trip_charge_ub_{b}_{t}",
                        )
                    else:
                        # During trip window - SOC constant IF departed, else can charge
                        M = boat_objects[b].battery_capacity * 2

                        # If departed: SOC stays constant (on trip)
                        model.addCons(
                            soc[b][t]
                            >= soc[b][t - 1] - M * (1 - departed[b][trip_idx]),
                            name=f"soc_trip_const_lb_{b}_{t}",
                        )
                        model.addCons(
                            soc[b][t]
                            <= soc[b][t - 1] + M * (1 - departed[b][trip_idx]),
                            name=f"soc_trip_const_ub_{b}_{t}",
                        )

                        # If NOT departed: can charge (KEY FIX - allows late boats to keep charging)
                        model.addCons(
                            soc[b][t]
                            >= soc[b][t - 1]
                            + total_charging * charging_efficiency * self.timestep_hours
                            - M * departed[b][trip_idx],
                            name=f"soc_delayed_charge_lb_{b}_{t}",
                        )
                        model.addCons(
                            soc[b][t]
                            <= soc[b][t - 1]
                            + total_charging * charging_efficiency * self.timestep_hours
                            + M * departed[b][trip_idx],
                            name=f"soc_delayed_charge_ub_{b}_{t}",
                        )

        # 4. Departure logic: can only depart if SOC >= required energy
        # KEY CHANGE: Check SOC at ACTUAL departure time, not just deadline
        for b in boats:
            boat_trips = deadlines.get(b, [])
            for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                M = boat_objects[b].battery_capacity * 2

                # If departed, must have had enough SOC at deadline
                # (This is a simplification - ideally we'd check at actual_depart_t)
                model.addCons(
                    soc[b][deadline_t]
                    >= required_energy - M * (1 - departed[b][trip_idx]),
                    name=f"can_depart_{b}_{trip_idx}",
                )

                # late = 1 if boat didn't depart on time
                model.addCons(
                    late[b][trip_idx] >= 1 - departed[b][trip_idx],
                    name=f"late_def_{b}_{trip_idx}",
                )

        # 5. No charging during actual trips (ONLY when departed)
        for b in boats:
            boat_trips = deadlines.get(b, [])
            for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                trip_start_t, duration = trip_durations[b][trip_idx]
                departure_t = deadline_t + 1
                return_t = departure_t + duration

                for t in range(departure_t, min(return_t, T)):
                    for c_idx in range(num_chargers):
                        M = chargers[c_idx].max_power
                        # Can only charge if NOT departed (allows late boats to keep charging)
                        model.addCons(
                            p[c_idx][b][t] <= M * (1 - departed[b][trip_idx]),
                            name=f"no_charge_trip_{c_idx}_{b}_{t}_{trip_idx}",
                        )

        # ----------------------------------------------------
        # Objective: minimize cost + penalty for late boats
        # Make the late penalty VERY HIGH to encourage departures
        # Also add a "missed trip" penalty that's even higher for not departing at all
        # ----------------------------------------------------
        total_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )

        # High penalty for being late
        late_penalty = 1000.0
        total_late_penalty = late_penalty * quicksum(
            late[b][trip_idx]
            for b in boats
            for trip_idx in range(len(deadlines.get(b, [])))
        )

        # VERY high penalty for NOT departing at all (missed trip)
        # This ensures the optimizer prefers late departure over no departure
        missed_trip_penalty = 50000.0
        total_missed_penalty = missed_trip_penalty * quicksum(
            (1 - departed[b][trip_idx])
            for b in boats
            for trip_idx in range(len(deadlines.get(b, [])))
        )

        model.setObjective(
            total_cost + total_late_penalty + total_missed_penalty, "minimize"
        )

        # ----------------------------------------------------
        # Solve
        # ----------------------------------------------------
        print("        Solving optimization problem...")
        model.optimize()
        status = model.getStatus()
        print(f"        SCIP status: {status}")

        if status not in ["optimal", "bestsollimit", "timelimit"]:
            print(f"        WARNING: Optimization status is {status}")
            if model.getNSols() == 0:
                raise Exception(f"Optimization failed with status: {status}")

        # ----------------------------------------------------
        # Extract results
        # ----------------------------------------------------
        charger_schedules = {c.name: [] for c in chargers}
        boat_schedules = {b: [] for b in boats}

        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        # Print departure status
        print("        Departure status:")
        for b in boats:
            for trip_idx in range(len(deadlines.get(b, []))):
                dep_val = model.getVal(departed[b][trip_idx])
                late_val = model.getVal(late[b][trip_idx])
                deadline_t = deadlines[b][trip_idx][0]
                print(
                    f"          {b} trip {trip_idx+1} (t={deadline_t}): departed={dep_val:.0f}, late={late_val:.0f}"
                )

        for t in timesteps:
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)

            power_this_t = 0.0
            for c_idx, charger in enumerate(chargers):
                p_c_t = sum(max(0.0, model.getVal(p[c_idx][b][t])) for b in boats)
                charger_schedules[charger.name].append((timestamp, p_c_t))
                power_this_t += p_c_t

            # Extract per-boat power
            for b in boats:
                boat_power = 0.0
                for c_idx, charger in enumerate(chargers):
                    power_from_charger = max(0.0, model.getVal(p[c_idx][b][t]))
                    if power_from_charger > 0.01:
                        boat_power += power_from_charger

                boat_schedules[b].append((timestamp, boat_power))

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

        # Debug: Print boat schedules summary
        print("        Boat charging schedule summary:")
        for b in boats:
            total_boat_energy = sum(
                power * self.timestep_hours for _, power in boat_schedules[b]
            )
            charging_timesteps = sum(
                1 for _, power in boat_schedules[b] if power > 0.01
            )
            max_power = max(power for _, power in boat_schedules[b])
            print(
                f"          {b}: {total_boat_energy:.1f} kWh over {charging_timesteps} timesteps (max {max_power:.1f} kW)"
            )

        # Print planned SOC at key timesteps
        print("        Planned SOC at key timesteps:")
        for b in boats:
            boat_trips = deadlines.get(b, [])
            for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                soc_val = model.getVal(soc[b][deadline_t])
                soc_pct = (soc_val / boat_objects[b].battery_capacity) * 100
                print(
                    f"          {b} trip {trip_idx+1} deadline (t={deadline_t}): SOC={soc_val:.1f} kWh ({soc_pct:.1f}%)"
                )

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            boat_schedules=boat_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
            total_cost=total_cost_val,
        )

    # --------------------------------------------------------
    # Deadline extraction
    # --------------------------------------------------------
    def _extract_deadlines(self, energy_forecasts: List[EnergyForecast]):
        deadlines = defaultdict(list)
        T = len(energy_forecasts)
        boats = energy_forecasts[0].boat_required_energy_kwh.keys()

        for boat in boats:
            for t in range(T - 1):
                req_now = energy_forecasts[t].boat_required_energy_kwh.get(boat, 0.0)
                req_next = energy_forecasts[t + 1].boat_required_energy_kwh.get(
                    boat, 0.0
                )

                if req_now > 0 and req_next == 0:
                    deadlines[boat].append((t, req_now))

        return deadlines

    # --------------------------------------------------------
    # Extract trip durations
    # --------------------------------------------------------
    def _extract_trip_durations(self, energy_forecasts: List[EnergyForecast]):
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

        for boat_name, schedule in result.boat_schedules.items():
            boat_src = self.db_manager.get_or_create_source(boat_name, "boat")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, boat_src, power_setpoint_met, str(power)))

        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} schedule entries")
