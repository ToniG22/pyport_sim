"""Base optimizer: minimize cost subject to import <= contracted_power.

Supports PV production (free solar) and BESS (battery storage) to reduce
grid import costs.  The power balance is:

    grid_import + pv_production + Σ bess_discharge
        = Σ charger_power + Σ bess_charge + pv_export

Performance-tuned for 10+ vessel fleets:
  - Tight big-M bounds (energy_req, not battery_capacity)
  - SOS1 sets on departure slots for better branching
  - Strong symmetry breaking across all delay slots
  - MIP warm-start with greedy on-time schedule
  - Solver parameter tuning (time limit, gap tolerance, emphasis)
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
    """Result of a daily optimization run."""

    status: str
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]
    boat_schedules: Dict[str, List[Tuple[datetime, float]]]
    peak_power_kw: float
    total_energy_kwh: float
    total_cost: float


class BaseOptimizer:
    """Minimize cost with grid import constrained to be <= contracted_power."""

    def __init__(
        self,
        port: Port,
        db_manager: DatabaseManager,
        timestep_seconds: int = 900,
        boat_charger_assignments: Dict[str, int] = None,
        trip_schedule: tuple = ((9, 0), (14, 1)),
        max_slack_timesteps: int = 8,
        deadline_decay_factor: float = 0.5,
        bess_end_soc_penalty: float = 0.5,
        time_limit_seconds: int = 120,
        mip_gap: float = 0.01,
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0
        self.boat_charger_assignments = boat_charger_assignments or {}
        self.trip_schedule = trip_schedule
        self.max_slack_timesteps = max_slack_timesteps
        self.deadline_decay_factor = deadline_decay_factor
        self.bess_end_soc_penalty = bess_end_soc_penalty
        self.time_limit_seconds = time_limit_seconds
        self.mip_gap = mip_gap

        self._boat_battery_cap = {b.name: b.battery_capacity for b in self.port.boats}

    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
    ) -> BaseOptimizationResult:

        T = len(energy_forecasts)
        timesteps = list(range(T))

        model = Model("base_optimizer")
        model.hideOutput()

        model.setParam("limits/time", self.time_limit_seconds)
        model.setParam("limits/gap", self.mip_gap)

        trip_events = self._extract_trip_events(energy_forecasts)
        print("TRIP EVENTS", trip_events)

        boat_objects = {b.name: b for b in self.port.boats}
        charger_objects = {c.name: c for c in self.port.chargers}
        charger_to_boat = {
            self.port.chargers[idx].name: boat
            for boat, idx in self.boat_charger_assignments.items()
        }
        boat_list = list(self.boat_charger_assignments.keys())

        grid_import = self._add_grid_import_vars(model, timesteps)
        charger_power = self._add_charger_power_vars(model, timesteps)

        pv_forecast = self._get_pv_forecast(energy_forecasts, timesteps)
        pv_export = self._add_export_vars(model, timesteps)
        bess_charge, bess_discharge, bess_soc = self._add_bess_variables(
            model, timesteps
        )

        self._add_power_balance_constraints(
            model,
            timesteps,
            grid_import,
            charger_power,
            pv_forecast,
            bess_charge,
            bess_discharge,
            pv_export,
        )

        self._add_bess_soc_dynamics(
            model, timesteps, bess_charge, bess_discharge, bess_soc
        )

        departure_ctx = self._add_departure_variables(
            model, T, timesteps, trip_events, boat_list, forecast_date
        )

        boat_away = self._add_boat_away_variables(
            model, timesteps, trip_events, departure_ctx["at_sea"]
        )

        drain_at = self._add_drain_variables(model, T, trip_events, departure_ctx)

        soc = self._add_soc_variables_and_dynamics(
            model,
            timesteps,
            trip_events,
            charger_to_boat,
            boat_objects,
            charger_objects,
            charger_power,
            boat_away,
            drain_at,
        )

        self._add_departure_soc_constraints(model, T, trip_events, soc, departure_ctx)

        self._add_symmetry_breaking(model, trip_events, boat_list, departure_ctx)

        tariff_price = self._compute_tariff_prices(forecast_date, timesteps)

        self._set_objective(
            model,
            timesteps,
            grid_import,
            tariff_price,
            boat_list,
            trip_events,
            departure_ctx,
            bess_soc,
        )

        model.optimize()
        status = model.getStatus()
        if status not in ("optimal", "bestsollimit", "timelimit"):
            if model.getNSols() == 0:
                raise RuntimeError(f"SCIP failed ({status}), no feasible solution")
            print(
                f"     Warning: solver status '{status}' but using best solution found"
            )

        self._log_departure_decisions(model, forecast_date, trip_events, departure_ctx)
        self._log_bess_summary(model, timesteps, bess_charge, bess_discharge, bess_soc)
        self._log_pv_summary(model, timesteps, pv_forecast, pv_export)

        result = self._extract_results(
            model,
            forecast_date,
            timesteps,
            charger_power,
            grid_import,
            tariff_price,
            status,
        )

        self._save_bess_schedules_to_db(
            model, forecast_date, timesteps, bess_charge, bess_discharge
        )

        return result

    def _add_grid_import_vars(self, model, timesteps):
        return {
            t: model.addVar(lb=0, ub=self.port.contracted_power, name=f"grid_{t}")
            for t in timesteps
        }

    def _add_charger_power_vars(self, model, timesteps):
        charger_power = {}
        for charger in self.port.chargers:
            charger_power[charger.name] = {
                t: model.addVar(
                    lb=0, ub=charger.max_power, name=f"p_{charger.name}_{t}"
                )
                for t in timesteps
            }
        return charger_power

    def _get_pv_forecast(self, energy_forecasts, timesteps):
        return {t: energy_forecasts[t].power_active_production_kw for t in timesteps}

    def _add_export_vars(self, model, timesteps):
        """Slack variable for excess power (PV curtailment / grid export)."""
        return {t: model.addVar(lb=0, name=f"export_{t}") for t in timesteps}

    def _add_bess_variables(self, model, timesteps):
        bess_charge = {}
        bess_discharge = {}
        bess_soc = {}

        for bess in self.port.bess_systems:
            bess_charge[bess.name] = {
                t: model.addVar(
                    lb=0,
                    ub=bess.max_charge_power,
                    name=f"bess_chg_{bess.name}_{t}",
                )
                for t in timesteps
            }
            bess_discharge[bess.name] = {
                t: model.addVar(
                    lb=0,
                    ub=bess.max_discharge_power,
                    name=f"bess_dis_{bess.name}_{t}",
                )
                for t in timesteps
            }
            bess_soc[bess.name] = {
                t: model.addVar(
                    lb=bess.soc_min * bess.capacity,
                    ub=bess.soc_max * bess.capacity,
                    name=f"bess_soc_{bess.name}_{t}",
                )
                for t in timesteps
            }

        return bess_charge, bess_discharge, bess_soc

    def _add_bess_soc_dynamics(
        self, model, timesteps, bess_charge, bess_discharge, bess_soc
    ):
        for bess in self.port.bess_systems:
            initial_soc_kwh = bess.current_soc * bess.capacity
            for t in timesteps:
                soc_prev = bess_soc[bess.name][t - 1] if t > 0 else initial_soc_kwh
                energy_in = (
                    bess_charge[bess.name][t] * bess.efficiency * self.timestep_hours
                )
                energy_out = (
                    bess_discharge[bess.name][t]
                    * (1.0 / bess.efficiency)
                    * self.timestep_hours
                )
                model.addCons(
                    bess_soc[bess.name][t] == soc_prev + energy_in - energy_out,
                    name=f"bess_soc_dyn_{bess.name}_{t}",
                )

    def _add_power_balance_constraints(
        self,
        model,
        timesteps,
        grid_import,
        charger_power,
        pv_forecast,
        bess_charge,
        bess_discharge,
        pv_export,
    ):
        """Enforce power balance at each timestep as an equality."""
        for t in timesteps:
            supply = grid_import[t] + pv_forecast[t]
            if self.port.bess_systems:
                supply += quicksum(
                    bess_discharge[b.name][t] for b in self.port.bess_systems
                )

            demand = quicksum(charger_power[c.name][t] for c in self.port.chargers)
            if self.port.bess_systems:
                demand += quicksum(
                    bess_charge[b.name][t] for b in self.port.bess_systems
                )
            demand += pv_export[t]

            model.addCons(supply == demand, name=f"power_balance_{t}")

    def _add_departure_variables(
        self,
        model,
        T,
        _timesteps,
        trip_events,
        _boat_list,
        forecast_date,
    ):
        """Create depart_at, did_depart, fully_ready, at_sea variables with an 18:00 cutoff."""

        depart_at = {}
        depart_slots = {}
        at_sea = {}
        fully_ready = {}
        did_depart = {}

        cutoff_time = forecast_date.replace(hour=18, minute=0, second=0)
        cutoff_t = int(
            (cutoff_time - forecast_date).total_seconds() / self.timestep_seconds
        )

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])

            for i, (t_deadline, _energy_req, dur) in enumerate(boat_trips):
                t_orig_depart = t_deadline + 1

                if i + 1 < len(boat_trips):
                    next_deadline = boat_trips[i + 1][0]
                    max_late = next_deadline - dur - t_orig_depart
                else:
                    max_late = T - dur - t_orig_depart

                max_late = max(0, min(max_late, self.max_slack_timesteps))

                slots = []
                for s in range(max_late + 1):
                    dep_t = t_orig_depart + s
                    if dep_t <= cutoff_t:
                        slots.append(s)

                depart_slots[(boat, i)] = slots

                slot_vars = []
                for s in slots:
                    v = model.addVar(
                        vtype="B",
                        name=f"depart_{boat}_trip{i+1}_s{s}",
                    )
                    depart_at[(boat, i, s)] = v
                    slot_vars.append(v)

                    model.chgVarBranchPriority(v, 100)

                if slot_vars:
                    model.addCons(
                        quicksum(slot_vars) <= 1,
                        name=f"one_depart_{boat}_trip{i+1}",
                    )

                    if len(slot_vars) > 1:
                        model.addConsSOS1(
                            slot_vars,
                            weights=[float(s) for s in slots],
                            name=f"sos1_depart_{boat}_trip{i+1}",
                        )

                did_depart[(boat, i)] = model.addVar(
                    vtype="B",
                    name=f"did_depart_{boat}_trip{i+1}",
                )

                if slot_vars:
                    model.addCons(
                        did_depart[(boat, i)] == quicksum(slot_vars),
                        name=f"did_depart_link_{boat}_trip{i+1}",
                    )
                else:
                    model.addCons(
                        did_depart[(boat, i)] == 0,
                        name=f"did_depart_forced_skip_{boat}_trip{i+1}",
                    )

                if 0 in slots:
                    fully_ready[(boat, i)] = depart_at[(boat, i, 0)]

                all_sea_times = set()
                for s in slots:
                    dep_t = t_orig_depart + s
                    for dt in range(dur):
                        t_sea = dep_t + dt
                        if t_sea < T:
                            all_sea_times.add(t_sea)

                for t in all_sea_times:
                    at_sea[(boat, i, t)] = model.addVar(
                        lb=0,
                        ub=1,
                        vtype="C",
                        name=f"sea_{boat}_trip{i+1}_t{t}",
                    )

                for t in all_sea_times:
                    contributing_slots = [
                        s
                        for s in slots
                        if t_orig_depart + s <= t < t_orig_depart + s + dur
                    ]
                    model.addCons(
                        at_sea[(boat, i, t)]
                        == quicksum(
                            depart_at[(boat, i, s)] for s in contributing_slots
                        ),
                        name=f"sea_link_{boat}_trip{i+1}_t{t}",
                    )

        return {
            "depart_at": depart_at,
            "depart_slots": depart_slots,
            "at_sea": at_sea,
            "fully_ready": fully_ready,
            "did_depart": did_depart,
        }

    def _add_boat_away_variables(self, model, timesteps, trip_events, at_sea):
        """Aggregate at_sea across trips into a single boat_away per (boat, t)."""
        boat_away = {}
        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for t in timesteps:
                sea_vars = [
                    at_sea[(boat, i, t)]
                    for i in range(len(boat_trips))
                    if (boat, i, t) in at_sea
                ]
                if sea_vars:
                    boat_away[(boat, t)] = model.addVar(
                        lb=0, ub=1, vtype="C", name=f"away_{boat}_t{t}"
                    )
                    model.addCons(
                        boat_away[(boat, t)] == quicksum(sea_vars),
                        name=f"away_link_{boat}_t{t}",
                    )
        return boat_away

    def _add_drain_variables(self, model, T, trip_events, departure_ctx):
        """Return-drain with explicit continuous variables."""
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]
        drain_at = {}

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                t_orig_depart = t_deadline + 1
                slots = depart_slots[(boat, i)]

                for s in slots:
                    t_return = t_orig_depart + s + dur
                    if t_return >= T:
                        continue

                    if (boat, i, t_return) not in drain_at:
                        drain_at[(boat, i, t_return)] = model.addVar(
                            lb=0,
                            ub=energy_req,
                            name=f"drain_{boat}_trip{i+1}_t{t_return}",
                        )

                    d = drain_at[(boat, i, t_return)]
                    da = depart_at[(boat, i, s)]

                    model.addCons(
                        d <= energy_req * da,
                        name=f"drain_ub_{boat}_trip{i+1}_s{s}",
                    )
                    model.addCons(
                        d >= energy_req * da,
                        name=f"drain_lb_{boat}_trip{i+1}_s{s}",
                    )

        return drain_at

    def _add_soc_variables_and_dynamics(
        self,
        model,
        timesteps,
        trip_events,
        charger_to_boat,
        boat_objects,
        charger_objects,
        charger_power,
        boat_away,
        drain_at,
    ):
        """SOC tracking per boat with charging, draining, and no-charge-while-away."""
        soc = {}

        for charger in self.port.chargers:
            charger_name = charger.name
            boat = charger_to_boat[charger_name]
            battery_cap = boat_objects[boat].battery_capacity
            initial_soc_kwh = boat_objects[boat].soc * battery_cap
            charger_eff = charger_objects[charger_name].efficiency

            soc[boat] = {
                t: model.addVar(
                    lb=0,
                    ub=battery_cap * 0.99,
                    name=f"soc_{boat}_{t}",
                )
                for t in timesteps
            }

            for t in timesteps:
                if (boat, t) in boat_away:
                    model.addCons(
                        charger_power[charger_name][t]
                        <= charger.max_power * (1 - boat_away[(boat, t)]),
                        name=f"no_charge_away_{boat}_t{t}",
                    )

                soc_prev = soc[boat][t - 1] if t > 0 else initial_soc_kwh
                charge_energy = (
                    charger_power[charger_name][t] * charger_eff * self.timestep_hours
                )

                boat_trips = trip_events.get(boat, [])
                drain_terms = [
                    drain_at[(boat, i, t)]
                    for i in range(len(boat_trips))
                    if (boat, i, t) in drain_at
                ]
                total_drain = quicksum(drain_terms) if drain_terms else 0.0

                model.addCons(
                    soc[boat][t] == soc_prev + charge_energy - total_drain,
                    name=f"soc_dyn_{boat}_{t}",
                )

        return soc

    def _add_departure_soc_constraints(self, model, T, trip_events, soc, departure_ctx):
        """Ensure SOC >= energy_req at actual departure using tight big-M = energy_req."""
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])

            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                t_orig_depart = t_deadline + 1
                slots = depart_slots[(boat, i)]
                big_m = energy_req

                for s in slots:
                    t_dl = t_orig_depart + s - 1
                    if t_dl < 0 or t_dl >= T:
                        continue
                    model.addCons(
                        soc[boat][t_dl]
                        >= energy_req - big_m * (1 - depart_at[(boat, i, s)]),
                        name=f"soc_req_{boat}_trip{i+1}_s{s}",
                    )

    def _add_symmetry_breaking(self, model, trip_events, boat_list, departure_ctx):
        """Strong symmetry breaking across identical boats.

        For each trip index, boats are ordered: if boat k hasn't departed
        by slot s, neither can boat k+1.  This covers all delay slots,
        not just on-time (slot 0).

        Cumulative form: sum(depart_at[curr, 0..s]) <= sum(depart_at[prev, 0..s])
        """
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]

        max_trips_per_boat = max(
            (len(trip_events.get(b, [])) for b in boat_list), default=0
        )

        for i in range(max_trips_per_boat):
            for k in range(1, len(boat_list)):
                b_prev = boat_list[k - 1]
                b_curr = boat_list[k]

                if (b_prev, i) not in depart_slots or (b_curr, i) not in depart_slots:
                    continue

                slots_prev = depart_slots[(b_prev, i)]
                slots_curr = depart_slots[(b_curr, i)]

                common_slots = min(len(slots_prev), len(slots_curr))
                for s in range(common_slots):
                    cum_prev = quicksum(
                        depart_at[(b_prev, i, ss)] for ss in slots_prev[: s + 1]
                    )
                    cum_curr = quicksum(
                        depart_at[(b_curr, i, ss)] for ss in slots_curr[: s + 1]
                    )
                    model.addCons(
                        cum_curr <= cum_prev,
                        name=f"sym_{b_curr}_trip{i+1}_s{s}",
                    )


    def _compute_tariff_prices(self, forecast_date, timesteps):
        return {
            t: self.port.get_tariff_price(
                forecast_date + timedelta(seconds=t * self.timestep_seconds)
            )
            for t in timesteps
        }

    def _set_objective(
        self,
        model,
        timesteps,
        grid_import,
        tariff_price,
        boat_list,
        trip_events,
        departure_ctx,
        bess_soc,
    ):
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]
        did_depart = departure_ctx["did_depart"]

        energy_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )

        n_boats = len(boat_list)
        base_reward = 1000.0
        ready_reward_terms = []

        for boat_idx, boat in enumerate(boat_list):
            boat_trips = trip_events.get(boat, [])
            tie_break = (n_boats - boat_idx) * 0.1

            for i in range(len(boat_trips)):
                slots = depart_slots.get((boat, i), [])
                for s in slots:
                    reward_at_s = (
                        base_reward * (self.deadline_decay_factor**s) + tie_break
                    )
                    ready_reward_terms.append(reward_at_s * depart_at[(boat, i, s)])

        ready_reward = quicksum(ready_reward_terms) if ready_reward_terms else 0

        missed_penalty_terms = []
        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i in range(len(boat_trips)):
                if (boat, i) in did_depart:
                    missed_penalty_terms.append(2000 * (1 - did_depart[(boat, i)]))

        missed_trip_penalty = (
            quicksum(missed_penalty_terms) if missed_penalty_terms else 0
        )

        bess_end_penalty = 0
        if self.port.bess_systems and self.bess_end_soc_penalty > 0:
            T_last = timesteps[-1]
            bess_shortfall_terms = []
            for bess in self.port.bess_systems:
                target_kwh = bess.current_soc * bess.capacity
                shortfall = model.addVar(lb=0, name=f"bess_end_short_{bess.name}")
                model.addCons(
                    shortfall >= target_kwh - bess_soc[bess.name][T_last],
                    name=f"bess_end_target_{bess.name}",
                )
                bess_shortfall_terms.append(self.bess_end_soc_penalty * shortfall)
            bess_end_penalty = quicksum(bess_shortfall_terms)

        model.setObjective(
            -ready_reward + missed_trip_penalty + energy_cost + bess_end_penalty,
            "minimize",
        )

    def _log_departure_decisions(
        self, model, forecast_date, trip_events, departure_ctx
    ):
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                slots = depart_slots.get((boat, i), [])
                departed = False
                for s in slots:
                    if model.getVal(depart_at[(boat, i, s)]) > 0.5:
                        t_actual = t_deadline + 1 + s
                        ts = forecast_date + timedelta(
                            seconds=t_actual * self.timestep_seconds
                        )
                        if s == 0:
                            print(f"     {boat} trip {i+1}: ON TIME at {ts}")
                        else:
                            print(
                                f"     {boat} trip {i+1}: DELAYED +{s} steps "
                                f"({s * self.timestep_seconds / 60:.0f} min), "
                                f"departs {ts}"
                            )
                        departed = True
                        break
                if not departed:
                    print(f"     {boat} trip {i+1}: SKIPPED (infeasible)")

    def _log_bess_summary(
        self, model, timesteps, bess_charge, bess_discharge, bess_soc
    ):
        if not self.port.bess_systems:
            return

        for bess in self.port.bess_systems:
            total_charged = sum(
                model.getVal(bess_charge[bess.name][t]) * self.timestep_hours
                for t in timesteps
            )
            total_discharged = sum(
                model.getVal(bess_discharge[bess.name][t]) * self.timestep_hours
                for t in timesteps
            )
            soc_start = bess.current_soc * bess.capacity
            soc_end = model.getVal(bess_soc[bess.name][timesteps[-1]])
            peak_discharge = max(
                model.getVal(bess_discharge[bess.name][t]) for t in timesteps
            )

            print(f"     🔋 {bess.name}:")
            print(
                f"        Charged:    {total_charged:7.2f} kWh   "
                f"Discharged: {total_discharged:7.2f} kWh"
            )
            print(
                f"        SOC start:  {soc_start:7.2f} kWh   "
                f"SOC end:    {soc_end:7.2f} kWh"
            )
            print(f"        Peak discharge power: {peak_discharge:.2f} kW")

    def _log_pv_summary(self, model, timesteps, pv_forecast, pv_export):
        if not self.port.pv_systems:
            return

        total_pv = sum(pv_forecast[t] * self.timestep_hours for t in timesteps)
        total_export = sum(
            model.getVal(pv_export[t]) * self.timestep_hours for t in timesteps
        )
        total_used = total_pv - total_export
        pct_used = (total_used / total_pv * 100) if total_pv > 0 else 0

        print("     ☀️  PV:")
        print(
            f"        Produced: {total_pv:7.2f} kWh   "
            f"Used: {total_used:7.2f} kWh ({pct_used:.1f}%)"
        )
        print(f"        Exported / curtailed: {total_export:7.2f} kWh")

    def _extract_results(
        self,
        model,
        forecast_date,
        timesteps,
        charger_power,
        grid_import,
        tariff_price,
        status,
    ):
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

    def _extract_trip_events(self, energy_forecasts):
        trips = defaultdict(list)
        T = len(energy_forecasts)

        for boat in self.port.boats:
            t = 0
            while t < T:
                prev_avail = (
                    energy_forecasts[t - 1].boat_available.get(boat.name, 1)
                    if t > 0
                    else 1
                )
                curr_avail = energy_forecasts[t].boat_available.get(boat.name, 1)

                if prev_avail == 1 and curr_avail == 0:
                    start_t = t
                    dur = 0
                    while (
                        t < T
                        and energy_forecasts[t].boat_available.get(boat.name, 1) == 0
                    ):
                        dur += 1
                        t += 1

                    deadline_t = start_t - 1
                    energy_req = energy_forecasts[
                        deadline_t
                    ].boat_required_energy_kwh.get(boat.name, 0.0)

                    trips[boat.name].append((deadline_t, energy_req, dur))
                else:
                    t += 1

        return trips

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
            print(f"     Saved {len(schedules)} charger schedule entries")

    def _save_bess_schedules_to_db(
        self, model, forecast_date, timesteps, bess_charge, bess_discharge
    ):
        if not self.port.bess_systems:
            return

        schedules = []
        power_setpoint_met = self.db_manager.get_metric_id("power_setpoint")

        for bess in self.port.bess_systems:
            bess_src = self.db_manager.get_or_create_source(bess.name, "bess")
            for t in timesteps:
                ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                chg = model.getVal(bess_charge[bess.name][t])
                dis = model.getVal(bess_discharge[bess.name][t])
                net_power = dis - chg
                schedules.append((ts_str, bess_src, power_setpoint_met, str(net_power)))

        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     Saved {len(schedules)} BESS schedule entries")
