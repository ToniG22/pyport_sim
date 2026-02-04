"""Base optimizer v10: Maximize power usage during critical windows.

Key insight from v8/v9 failure:
- Optimizer was only using 44-66 kW when 80 kW was available
- This wasted precious charging time in the critical 12:30-14:00 window
- Result: All boats slightly undercharged instead of some fully charged

Solution: Add objective term that penalizes UNUSED contracted power when boats need charging.
This forces the optimizer to use all available power during critical windows.
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
    """Maximize power usage during critical charging windows."""

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
        print("     Running base optimization v10 (max power usage)...")

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

        # Add 15% safety margin to energy requirements
        ENERGY_SAFETY_MARGIN = 1.15
        for b in deadlines:
            deadlines[b] = [
                (t, energy * ENERGY_SAFETY_MARGIN) for t, energy in deadlines[b]
            ]

        print(f"        Deadlines (with 15% margin): {dict(deadlines)}")
        print(f"        Trip durations: {dict(trip_durations)}")

        for b in boats:
            cap = boat_objects[b].battery_capacity
            init_soc_kwh = boat_objects[b].soc * cap
            print(
                f"        {b}: Initial SOC = {init_soc_kwh:.1f} kWh ({boat_objects[b].soc:.1%})"
            )

        # Cutoff for late departures (18:00)
        cutoff_t = int(18 * 3600 / self.timestep_seconds)

        # Create model
        model = Model("optimizer_v10")
        model.hideOutput()
        model.setRealParam("limits/time", 120.0)
        model.setRealParam("limits/gap", 0.03)

        # =====================================================
        # VARIABLES
        # =====================================================

        # Charging power per boat per timestep
        max_boat_power = min(
            22.0, self.port.contracted_power / num_boats * 2
        )  # Allow flexibility
        p = {
            b: {
                t: model.addVar(f"p_{b}_{t}", vtype="C", lb=0, ub=max_boat_power)
                for t in timesteps
            }
            for b in boats
        }

        # SOC at end of timestep
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

        # Unused power (slack variable to penalize)
        unused_power = {
            t: model.addVar(
                f"unused_{t}", vtype="C", lb=0, ub=self.port.contracted_power
            )
            for t in timesteps
        }

        # Trip departure variables
        depart_at = {}
        for b in boats:
            depart_at[b] = {}
            for trip_idx, (deadline_t, _) in enumerate(deadlines.get(b, [])):
                depart_at[b][trip_idx] = {}
                for t in range(deadline_t, cutoff_t):
                    depart_at[b][trip_idx][t] = model.addVar(
                        f"dep_{b}_{trip_idx}_{t}", vtype="B"
                    )

        # at_port[b][t] = 1 if boat can charge
        at_port = {
            b: {t: model.addVar(f"port_{b}_{t}", vtype="B") for t in timesteps}
            for b in boats
        }

        # needs_charge[b][t] = 1 if boat is at port and not fully charged
        needs_charge = {
            b: {t: model.addVar(f"need_{b}_{t}", vtype="B") for t in timesteps}
            for b in boats
        }

        # =====================================================
        # CONSTRAINTS
        # =====================================================

        # 1. Total power limit
        for t in timesteps:
            model.addCons(
                quicksum(p[b][t] for b in boats) <= self.port.contracted_power,
                f"power_limit_{t}",
            )

        # 2. Power balance with unused tracking
        for t in timesteps:
            total_charging = quicksum(p[b][t] for b in boats)
            model.addCons(grid_import[t] == total_charging, f"balance_{t}")
            # Unused power = contracted - actual used (when boats need charging)
            model.addCons(
                unused_power[t]
                >= self.port.contracted_power
                - total_charging
                - self.port.contracted_power
                * (1 - quicksum(needs_charge[b][t] for b in boats) / num_boats),
                f"unused_{t}",
            )

        # 3. Initial SOC
        for b in boats:
            init_soc = boat_objects[b].soc * boat_objects[b].battery_capacity
            model.addCons(soc[b][0] == init_soc, f"init_soc_{b}")

        # 4. Can only charge when at port
        for b in boats:
            for t in timesteps:
                model.addCons(
                    p[b][t] <= max_boat_power * at_port[b][t], f"at_port_charge_{b}_{t}"
                )

        # 5. needs_charge logic: at_port AND soc < capacity
        for b in boats:
            cap = boat_objects[b].battery_capacity
            for t in timesteps:
                # needs_charge <= at_port
                model.addCons(needs_charge[b][t] <= at_port[b][t], f"need_port_{b}_{t}")
                # needs_charge <= 1 if soc < cap (using big-M)
                # If soc >= cap - epsilon, needs_charge should be 0
                # soc >= cap - epsilon + M*(1 - needs_charge) would force needs_charge=0 when soc is high
                # Simplified: just link needs_charge to at_port for now
                model.addCons(
                    needs_charge[b][t] >= at_port[b][t] - soc[b][t] / cap,
                    f"need_soc_{b}_{t}",
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

                # A. Must depart exactly once
                model.addCons(
                    quicksum(
                        depart_at[b][trip_idx][t] for t in range(deadline_t, cutoff_t)
                    )
                    == 1,
                    f"one_depart_{b}_{trip_idx}",
                )

                # B. SOC requirement at departure
                for t in range(deadline_t, cutoff_t):
                    model.addCons(
                        soc[b][t]
                        >= required_energy - M * (1 - depart_at[b][trip_idx][t]),
                        f"soc_req_{b}_{trip_idx}_{t}",
                    )

                # C. Trip ordering
                if trip_idx > 0:
                    prev_deadline, _ = boat_trips[trip_idx - 1]
                    _, prev_duration = trip_durations[b][trip_idx - 1]

                    for t in range(deadline_t, cutoff_t):
                        prev_returned = quicksum(
                            depart_at[b][trip_idx - 1][t_prev]
                            for t_prev in range(prev_deadline, min(cutoff_t, t))
                            if t_prev + prev_duration < t
                        )
                        model.addCons(
                            depart_at[b][trip_idx][t] <= prev_returned,
                            f"order_{b}_{trip_idx}_{t}",
                        )

        # =====================================================
        # AT_PORT DYNAMICS
        # =====================================================

        for b in boats:
            boat_trips = deadlines.get(b, [])

            for t in timesteps:
                on_trip_indicators = []

                for trip_idx, (deadline_t, _) in enumerate(boat_trips):
                    _, duration = trip_durations[b][trip_idx]

                    for t_dep in range(deadline_t, min(cutoff_t, T)):
                        if t_dep < t <= t_dep + duration:
                            on_trip_indicators.append(depart_at[b][trip_idx][t_dep])

                if on_trip_indicators:
                    for ind in on_trip_indicators:
                        model.addCons(
                            at_port[b][t] <= 1 - ind, f"not_trip_{b}_{t}_{id(ind)}"
                        )
                    model.addCons(
                        at_port[b][t] >= 1 - quicksum(on_trip_indicators),
                        f"port_lb_{b}_{t}",
                    )
                else:
                    model.addCons(at_port[b][t] == 1, f"always_port_{b}_{t}")

        # =====================================================
        # SOC DYNAMICS
        # =====================================================

        for b in boats:
            boat_trips = deadlines.get(b, [])

            for t in range(1, T):
                charge_energy = p[b][t - 1] * efficiency * self.timestep_hours

                # Energy consumed from trip returns
                trip_energy = 0
                for trip_idx, (deadline_t, required_energy) in enumerate(boat_trips):
                    _, duration = trip_durations[b][trip_idx]
                    t_dep = t - duration - 1

                    if deadline_t <= t_dep < cutoff_t:
                        trip_energy += required_energy * depart_at[b][trip_idx][t_dep]

                model.addCons(
                    soc[b][t] == soc[b][t - 1] + charge_energy - trip_energy,
                    f"soc_dyn_{b}_{t}",
                )

        # =====================================================
        # OBJECTIVE
        # =====================================================

        # Get tariffs
        tariff = {}
        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff[t] = self.port.get_tariff_price(ts)

        # 1. Energy cost (small weight)
        energy_cost = quicksum(
            grid_import[t] * tariff[t] * self.timestep_hours for t in timesteps
        )

        # 2. Delay penalty - STRONGLY penalize NUMBER of delayed trips
        # Key insight: We want 3 on-time + 2 very late >> 5 slightly late
        # So the FIRST delay step is EXTREMELY expensive (50000)
        # But subsequent steps are cheap (50 each)
        # This makes: 2 boats delayed = 100,000 penalty
        #             5 boats delayed = 250,000 penalty
        # The optimizer will strongly prefer fewer delayed boats

        FIRST_DELAY_PENALTY = 50000  # Huge penalty for being late AT ALL
        INCREMENTAL_DELAY_PENALTY = 50  # Small penalty for each additional timestep

        delay_penalty = 0
        for b in boats:
            for trip_idx, (deadline_t, _) in enumerate(deadlines.get(b, [])):
                for t in range(deadline_t + 1, cutoff_t):  # Only t > deadline
                    delay_steps = t - deadline_t
                    # First step costs 50000, each additional step costs 50
                    penalty = (
                        FIRST_DELAY_PENALTY
                        + (delay_steps - 1) * INCREMENTAL_DELAY_PENALTY
                    )
                    delay_penalty += penalty * depart_at[b][trip_idx][t]

        # 3. Penalize unused power during critical windows
        # This forces the optimizer to use all available power when boats need charging
        # Higher penalty = stronger incentive to use full contracted power
        unused_penalty = quicksum(unused_power[t] * 100 for t in timesteps)

        model.setObjective(energy_cost + delay_penalty + unused_penalty, "minimize")

        # =====================================================
        # SOLVE
        # =====================================================

        print("        Solving...")
        model.optimize()
        status = model.getStatus()
        print(f"        Status: {status}, Solutions: {model.getNSols()}")

        if model.getNSols() == 0:
            raise Exception(f"No solution found: {status}")

        # =====================================================
        # EXTRACT RESULTS
        # =====================================================

        print("        Trip schedule:")
        on_time = 0
        late = 0

        for b in boats:
            for trip_idx, (deadline_t, req) in enumerate(deadlines.get(b, [])):
                for t in range(deadline_t, cutoff_t):
                    if model.getVal(depart_at[b][trip_idx][t]) > 0.5:
                        soc_val = model.getVal(soc[b][t])
                        delay = t - deadline_t
                        if delay == 0:
                            print(
                                f"          {b} trip {trip_idx+1}: ✓ ON TIME at t={t}, SOC={soc_val:.1f}kWh"
                            )
                            on_time += 1
                        else:
                            delay_min = delay * 15
                            print(
                                f"          {b} trip {trip_idx+1}: LATE by {delay_min}min (t={t}), SOC={soc_val:.1f}kWh"
                            )
                            late += 1
                        break

        print(f"        Summary: {on_time} on-time, {late} late")

        # Debug: Show power usage in critical window
        print("        Power usage in critical window (12:30-14:00):")
        for t in range(50, 56):  # t=50 is 12:30, t=55 is 13:45
            total_p = sum(model.getVal(p[b][t]) for b in boats)
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            print(
                f"          {ts.strftime('%H:%M')}: {total_p:.1f} kW / {self.port.contracted_power} kW"
            )

        # Build schedules
        charger_schedules = {c.name: [] for c in chargers}
        boat_schedules = {b: [] for b in boats}
        peak_power = 0.0
        total_energy = 0.0
        total_cost_val = 0.0

        for t in timesteps:
            ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)

            boat_powers = {b: max(0, model.getVal(p[b][t])) for b in boats}

            # Simple charger assignment
            charger_power = {c.name: 0.0 for c in chargers}
            charger_idx = 0
            for b in boats:
                if boat_powers[b] > 0.1:
                    charger_power[
                        chargers[charger_idx % num_chargers].name
                    ] += boat_powers[b]
                    charger_idx += 1

            for c in chargers:
                charger_schedules[c.name].append((ts, charger_power[c.name]))

            for b in boats:
                boat_schedules[b].append((ts, boat_powers[b]))

            total_power = sum(boat_powers.values())
            peak_power = max(peak_power, total_power)
            total_energy += total_power * self.timestep_hours
            total_cost_val += (
                model.getVal(grid_import[t]) * tariff[t] * self.timestep_hours
            )

        print(
            f"     Complete: Peak={peak_power:.1f}kW, Energy={total_energy:.1f}kWh, Cost=€{total_cost_val:.2f}"
        )

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
