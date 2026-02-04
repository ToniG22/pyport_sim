"""Base optimizer v7: Clean implementation with proper multi-trip handling.

The key insight: We need to model trips as EVENTS that consume energy at specific
times, and ensure boats are charged sufficiently BEFORE each trip.

Approach:
1. For each trip, we have a "departure window" [deadline, cutoff]
2. A binary variable indicates WHETHER the boat departs for each trip
3. Another binary indicates AT WHICH TIMESTEP the boat departs
4. SOC constraints ensure boat has enough energy at departure time
5. Boat cannot charge while on trip (from departure to return)
6. Trip energy is consumed at the return timestep

This version uses a cleaner formulation that should solve faster.
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
    """Minimize cost while maximizing trip completion."""

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
        print("     Running base optimization v7 (multi-trip with late departure)...")

        T = len(energy_forecasts)
        timesteps = list(range(T))
        chargers = self.port.chargers
        boats = list(energy_forecasts[0].boat_required_energy_kwh.keys())

        num_chargers = len(chargers)
        num_boats = len(boats)
        efficiency = 0.95

        print(f"        {num_chargers} chargers, {num_boats} boats, {T} timesteps")
        print(f"        Contracted power: {self.port.contracted_power} kW")

        # Extract trip info
        deadlines = self._extract_deadlines(energy_forecasts)
        trip_durations = self._extract_trip_durations(energy_forecasts)
        boat_objects = {
            b: next(boat for boat in self.port.boats if boat.name == b) for b in boats
        }

        # Add safety margin to required energy to account for:
        # 1. Differences between forecast and actual energy consumption
        # 2. Charging efficiency losses
        # 3. Speed variations during trips
        # Using 15% margin based on observed ~10% discrepancy in testing
        ENERGY_SAFETY_MARGIN = 1.15
        for b in deadlines:
            deadlines[b] = [
                (t, energy * ENERGY_SAFETY_MARGIN) for t, energy in deadlines[b]
            ]

        print(
            f"        Deadlines (with {(ENERGY_SAFETY_MARGIN-1)*100:.0f}% safety margin): {dict(deadlines)}"
        )
        print(f"        Trip durations: {dict(trip_durations)}")

        # Cutoff hour for trips
        cutoff_t = int(18 * 3600 / self.timestep_seconds)  # 18:00

        # Create model
        model = Model("optimizer_v7")
        model.hideOutput()
        model.setRealParam("limits/time", 180.0)

        # =====================================================
        # VARIABLES
        # =====================================================

        # Charging power: p[c][b][t]
        p = {
            c_idx: {
                b: {
                    t: model.addVar(
                        f"p_{c_idx}_{b}_{t}",
                        vtype="C",
                        lb=0,
                        ub=chargers[c_idx].max_power,
                    )
                    for t in timesteps
                }
                for b in boats
            }
            for c_idx in range(num_chargers)
        }

        # Charger assignment: y[c][b][t] = 1 if charger c serves boat b at t
        y = {
            c_idx: {
                b: {t: model.addVar(f"y_{c_idx}_{b}_{t}", vtype="B") for t in timesteps}
                for b in boats
            }
            for c_idx in range(num_chargers)
        }

        # SOC at end of timestep t
        soc = {
            b: {
                t: model.addVar(
                    f"soc_{b}_{t}", vtype="C", lb=0, ub=boat_objects[b].battery_capacity
                )
                for t in timesteps
            }
            for b in boats
        }

        # Grid import
        grid_import = {
            t: model.addVar(f"grid_{t}", vtype="C", lb=0, ub=self.port.contracted_power)
            for t in timesteps
        }

        # PV used
        pv_avail = [
            (
                energy_forecasts[t].power_active_production_kw
                if self.port.pv_systems
                else 0.0
            )
            for t in timesteps
        ]
        pv_used = {
            t: model.addVar(f"pv_{t}", vtype="C", lb=0, ub=pv_avail[t])
            for t in timesteps
        }

        # Trip variables
        # depart_at[b][trip][t] = 1 if boat b departs for trip at timestep t
        # at_port[b][t] = 1 if boat is at port (can charge) at timestep t
        depart_at = {}
        at_port = {}

        for b in boats:
            depart_at[b] = {}
            at_port[b] = {
                t: model.addVar(f"at_port_{b}_{t}", vtype="B") for t in timesteps
            }

            for trip_idx in range(len(deadlines.get(b, []))):
                depart_at[b][trip_idx] = {
                    t: model.addVar(f"depart_{b}_{trip_idx}_{t}", vtype="B")
                    for t in timesteps
                }

        # =====================================================
        # CONSTRAINTS
        # =====================================================

        # 1. Charger capacity
        for c_idx in range(num_chargers):
            for t in timesteps:
                model.addCons(
                    quicksum(p[c_idx][b][t] for b in boats)
                    <= chargers[c_idx].max_power,
                    f"charger_cap_{c_idx}_{t}",
                )

        # 2. One boat per charger
        for c_idx in range(num_chargers):
            for t in timesteps:
                model.addCons(
                    quicksum(y[c_idx][b][t] for b in boats) <= 1,
                    f"one_boat_{c_idx}_{t}",
                )

        # 3. One charger per boat
        for b in boats:
            for t in timesteps:
                model.addCons(
                    quicksum(y[c_idx][b][t] for c_idx in range(num_chargers)) <= 1,
                    f"one_charger_{b}_{t}",
                )

        # 4. Power linked to assignment
        for c_idx in range(num_chargers):
            for b in boats:
                for t in timesteps:
                    model.addCons(
                        p[c_idx][b][t] <= chargers[c_idx].max_power * y[c_idx][b][t],
                        f"link_{c_idx}_{b}_{t}",
                    )

        # 5. Power balance
        for t in timesteps:
            total_charge = quicksum(
                p[c_idx][b][t] for c_idx in range(num_chargers) for b in boats
            )
            model.addCons(grid_import[t] + pv_used[t] == total_charge, f"balance_{t}")

        # 6. Initial SOC
        for b in boats:
            init_soc = boat_objects[b].soc * boat_objects[b].battery_capacity
            model.addCons(soc[b][0] == init_soc, f"init_soc_{b}")
            print(
                f"        {b}: Initial SOC = {init_soc:.1f} kWh ({boat_objects[b].soc:.1%})"
            )

        # 7. Can only charge when at port
        for b in boats:
            for t in timesteps:
                for c_idx in range(num_chargers):
                    model.addCons(
                        p[c_idx][b][t] <= chargers[c_idx].max_power * at_port[b][t],
                        f"charge_at_port_{c_idx}_{b}_{t}",
                    )

        # =====================================================
        # TRIP LOGIC
        # =====================================================

        for b in boats:
            boat_trips = deadlines.get(b, [])
            cap = boat_objects[b].battery_capacity
            M = cap * 2

            for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                _, duration = trip_durations[b][trip_idx]

                # A. Departure only within [deadline, cutoff)
                for t in timesteps:
                    if t < deadline_t or t >= cutoff_t:
                        model.addCons(
                            depart_at[b][trip_idx][t] == 0,
                            f"depart_window_{b}_{trip_idx}_{t}",
                        )

                # B. At most one departure per trip (0 or 1)
                model.addCons(
                    quicksum(depart_at[b][trip_idx][t] for t in timesteps) <= 1,
                    f"at_most_one_depart_{b}_{trip_idx}",
                )

                # C. If departing at t, must have SOC >= required_energy at t
                for t in range(deadline_t, min(cutoff_t, T)):
                    model.addCons(
                        soc[b][t]
                        >= required_energy - M * (1 - depart_at[b][trip_idx][t]),
                        f"soc_for_trip_{b}_{trip_idx}_{t}",
                    )

                # D. If trip_idx > 0, can only depart after returning from previous trip
                if trip_idx > 0:
                    prev_deadline, _ = boat_trips[trip_idx - 1]
                    _, prev_duration = trip_durations[b][trip_idx - 1]

                    # Previous trip's return time depends on when it departed
                    # For simplicity: trip trip_idx can only happen if we departed trip_idx-1
                    # AND enough time has passed

                    for t in range(deadline_t, min(cutoff_t, T)):
                        # Sum of previous departures that would return by time t
                        prev_returns_by_t = quicksum(
                            depart_at[b][trip_idx - 1][t_prev]
                            for t_prev in range(prev_deadline, min(cutoff_t, T))
                            if t_prev + prev_duration
                            < t  # Return at t_prev + duration, ready at t_prev + duration + 1
                        )
                        model.addCons(
                            depart_at[b][trip_idx][t] <= prev_returns_by_t,
                            f"trip_order_{b}_{trip_idx}_{t}",
                        )

        # =====================================================
        # AT_PORT DYNAMICS
        # =====================================================

        for b in boats:
            boat_trips = deadlines.get(b, [])

            for t in timesteps:
                # at_port[t] = 1 if NOT on any trip at time t
                # on_trip at t means: exists trip_idx and t_dep such that
                #   depart_at[trip_idx][t_dep] = 1 AND t_dep < t <= t_dep + duration

                on_trip_indicators = []
                for trip_idx, (deadline_t, _) in enumerate(boat_trips):
                    _, duration = trip_durations[b][trip_idx]

                    for t_dep in range(deadline_t, min(cutoff_t, T)):
                        # If departed at t_dep, on trip during [t_dep+1, t_dep+duration]
                        if t_dep < t <= t_dep + duration:
                            on_trip_indicators.append(depart_at[b][trip_idx][t_dep])

                if on_trip_indicators:
                    # at_port[t] = 1 - any(on_trip_indicators)
                    # at_port[t] <= 1 - each indicator
                    for ind in on_trip_indicators:
                        model.addCons(
                            at_port[b][t] <= 1 - ind, f"not_on_trip_{b}_{t}_{id(ind)}"
                        )
                    # at_port[t] >= 1 - sum(indicators)
                    model.addCons(
                        at_port[b][t] >= 1 - quicksum(on_trip_indicators),
                        f"at_port_lb_{b}_{t}",
                    )
                else:
                    # No trips could affect this timestep
                    model.addCons(at_port[b][t] == 1, f"always_at_port_{b}_{t}")

        # =====================================================
        # SOC DYNAMICS
        # =====================================================

        for b in boats:
            boat_trips = deadlines.get(b, [])
            cap = boat_objects[b].battery_capacity

            for t in range(1, T):
                charging = quicksum(p[c_idx][b][t - 1] for c_idx in range(num_chargers))
                charge_energy = charging * efficiency * self.timestep_hours

                # Energy consumed from trip returns at timestep t
                # A trip returns at t if departure was at t - duration - 1
                trip_energy_consumed = 0
                for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                    _, duration = trip_durations[b][trip_idx]

                    # Return happens at t_dep + duration + 1
                    # So if return is at t, departure was at t - duration - 1
                    t_dep = t - duration - 1
                    if deadline_t <= t_dep < cutoff_t:
                        trip_energy_consumed += (
                            required_energy * depart_at[b][trip_idx][t_dep]
                        )

                model.addCons(
                    soc[b][t] == soc[b][t - 1] + charge_energy - trip_energy_consumed,
                    f"soc_dyn_{b}_{t}",
                )

        # =====================================================
        # OBJECTIVE
        # =====================================================

        # Tariff prices
        tariff = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff[t] = self.port.get_tariff_price(ts)

        # Energy cost
        energy_cost = quicksum(
            grid_import[t] * tariff[t] * self.timestep_hours for t in timesteps
        )

        # Late departure penalty
        late_penalty = 200.0
        late_cost = 0
        for b in boats:
            for trip_idx, (deadline_t, _) in enumerate(deadlines.get(b, [])):
                for t in range(deadline_t, min(cutoff_t, T)):
                    lateness = t - deadline_t
                    late_cost += late_penalty * lateness * depart_at[b][trip_idx][t]

        # Missed trip penalty (didn't depart at all)
        missed_penalty = 50000.0
        missed_cost = 0
        for b in boats:
            for trip_idx in range(len(deadlines.get(b, []))):
                departed = quicksum(depart_at[b][trip_idx][t] for t in timesteps)
                missed_cost += missed_penalty * (1 - departed)

        model.setObjective(energy_cost + late_cost + missed_cost, "minimize")

        # =====================================================
        # SOLVE
        # =====================================================

        print("        Solving optimization problem...")
        model.optimize()
        status = model.getStatus()
        print(f"        Status: {status}, Solutions: {model.getNSols()}")

        if model.getNSols() == 0:
            raise Exception(f"No solution found: {status}")

        # =====================================================
        # EXTRACT RESULTS
        # =====================================================

        # Print trip results
        print("        Trip schedule:")
        for b in boats:
            for trip_idx, (deadline_t, req) in enumerate(deadlines.get(b, [])):
                departed_t = None
                for t in timesteps:
                    if model.getVal(depart_at[b][trip_idx][t]) > 0.5:
                        departed_t = t
                        break

                if departed_t is not None:
                    delay = departed_t - deadline_t
                    delay_min = delay * self.timestep_hours * 60
                    soc_at_dep = model.getVal(soc[b][departed_t])
                    if delay == 0:
                        print(
                            f"          {b} trip {trip_idx+1}: ON TIME at t={departed_t}, SOC={soc_at_dep:.1f}kWh"
                        )
                    else:
                        print(
                            f"          {b} trip {trip_idx+1}: LATE by {delay_min:.0f}min (t={departed_t}), SOC={soc_at_dep:.1f}kWh"
                        )
                else:
                    print(f"          {b} trip {trip_idx+1}: CANCELLED")

        # Build schedules
        charger_schedules = {c.name: [] for c in chargers}
        boat_schedules = {b: [] for b in boats}
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)

            power_t = 0.0
            for c_idx, charger in enumerate(chargers):
                pc = sum(max(0, model.getVal(p[c_idx][b][t])) for b in boats)
                charger_schedules[charger.name].append((ts, pc))
                power_t += pc

            for b in boats:
                bp = sum(
                    max(0, model.getVal(p[c_idx][b][t]))
                    for c_idx in range(num_chargers)
                )
                boat_schedules[b].append((ts, bp))

            peak_power = max(peak_power, power_t)
            total_energy += power_t * self.timestep_hours
            total_cost_val += (
                max(0, model.getVal(grid_import[t])) * tariff[t] * self.timestep_hours
            )

        print(
            f"     Complete: Peak={peak_power:.1f}kW, Energy={total_energy:.1f}kWh, Cost=€{total_cost_val:.2f}"
        )

        # Print charging summary
        print("        Charging summary:")
        for b in boats:
            total_kwh = sum(pw * self.timestep_hours for _, pw in boat_schedules[b])
            num_steps = sum(1 for _, pw in boat_schedules[b] if pw > 0.1)
            print(f"          {b}: {total_kwh:.1f} kWh over {num_steps} timesteps")

        return BaseOptimizationResult(
            status=status,
            charger_schedules=charger_schedules,
            boat_schedules=boat_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
            total_cost=total_cost_val,
        )

    def _extract_deadlines(self, energy_forecasts):
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

    def _extract_trip_durations(self, energy_forecasts):
        trip_durations = defaultdict(list)
        T = len(energy_forecasts)
        boats = energy_forecasts[0].boat_available.keys()
        for boat in boats:
            t = 0
            while t < T:
                prev = (
                    energy_forecasts[t - 1].boat_available.get(boat, 1) if t > 0 else 1
                )
                curr = energy_forecasts[t].boat_available.get(boat, 1)
                if prev == 1 and curr == 0:
                    start_t = t
                    dur = 0
                    while (
                        t < T and energy_forecasts[t].boat_available.get(boat, 1) == 0
                    ):
                        dur += 1
                        t += 1
                    trip_durations[boat].append((start_t - 1, dur))
                else:
                    t += 1
        return trip_durations

    def save_schedules_to_db(self, result: BaseOptimizationResult) -> None:
        schedules = []
        metric_id = self.db_manager.get_metric_id("power_setpoint")
        for name, schedule in result.charger_schedules.items():
            src = self.db_manager.get_or_create_source(name, "charger")
            for ts, pw in schedule:
                schedules.append(
                    (ts.strftime("%Y-%m-%d %H:%M:%S"), src, metric_id, str(pw))
                )
        for name, schedule in result.boat_schedules.items():
            src = self.db_manager.get_or_create_source(name, "boat")
            for ts, pw in schedule:
                schedules.append(
                    (ts.strftime("%Y-%m-%d %H:%M:%S"), src, metric_id, str(pw))
                )
        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} schedule entries")
