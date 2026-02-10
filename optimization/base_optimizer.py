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
        max_slack_timesteps: int = 8,
        deadline_decay_factor: float = 0.5,
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0
        self.boat_charger_assignments = boat_charger_assignments or {}
        self.trip_schedule = trip_schedule
        self.max_slack_timesteps = max_slack_timesteps
        self.deadline_decay_factor = deadline_decay_factor

        # Pre-compute per-boat battery capacities for tight Big-M bounds
        self._boat_battery_cap = {b.name: b.battery_capacity for b in self.port.boats}

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
    ) -> BaseOptimizationResult:

        T = len(energy_forecasts)
        timesteps = list(range(T))

        model = Model("base_optimizer")
        model.hideOutput()
        # model.setParam("limits/time", 120)

        trip_events = self._extract_trip_events(energy_forecasts)
        print("TRIP EVENTS", trip_events)

        # Lookups
        boat_objects = {b.name: b for b in self.port.boats}
        charger_objects = {c.name: c for c in self.port.chargers}
        charger_to_boat = {
            self.port.chargers[idx].name: boat
            for boat, idx in self.boat_charger_assignments.items()
        }
        boat_list = list(self.boat_charger_assignments.keys())

        # --- Decision variables ---
        grid_import = self._add_grid_import_vars(model, timesteps)
        charger_power = self._add_charger_power_vars(model, timesteps)
        self._add_power_balance_constraints(
            model, timesteps, grid_import, charger_power
        )

        # --- Departure / at-sea variables ---
        departure_ctx = self._add_departure_variables(
            model, T, timesteps, trip_events, boat_list
        )

        # --- Aggregate boat_away ---
        boat_away = self._add_boat_away_variables(
            model, timesteps, trip_events, departure_ctx["at_sea"]
        )

        # --- Return-drain ---
        drain_at = self._add_drain_variables(model, T, trip_events, departure_ctx)

        # --- SOC dynamics ---
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

        # --- SOC >= energy_req at departure ---
        self._add_departure_soc_constraints(model, T, trip_events, soc, departure_ctx)

        # --- Symmetry breaking ---
        self._add_symmetry_breaking(
            model, trip_events, boat_list, departure_ctx["fully_ready"]
        )

        # --- Tariffs & objective ---
        tariff_price = self._compute_tariff_prices(forecast_date, timesteps)

        self._set_objective(
            model,
            timesteps,
            grid_import,
            tariff_price,
            boat_list,
            trip_events,
            departure_ctx,
        )

        # --- Solve ---
        model.optimize()
        status = model.getStatus()
        if status != "optimal":
            raise RuntimeError(f"SCIP failed ({status})")

        # --- Post-processing ---
        self._log_departure_decisions(model, forecast_date, trip_events, departure_ctx)

        return self._extract_results(
            model,
            forecast_date,
            timesteps,
            charger_power,
            grid_import,
            tariff_price,
            status,
        )

    # ------------------------------------------------------------------ #
    #  Variable & constraint builders                                      #
    # ------------------------------------------------------------------ #

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

    def _add_power_balance_constraints(
        self, model, timesteps, grid_import, charger_power
    ):
        for t in timesteps:
            model.addCons(
                grid_import[t]
                == quicksum(charger_power[c.name][t] for c in self.port.chargers),
                name=f"power_balance_{t}",
            )

    def _add_departure_variables(self, model, T, timesteps, trip_events, boat_list):
        """Create depart_at, did_depart, fully_ready, at_sea variables and constraints."""
        depart_at = {}
        depart_slots = {}
        at_sea = {}
        fully_ready = {}
        did_depart = {}

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                t_orig_depart = t_deadline + 1

                # Max feasible delay
                if i + 1 < len(boat_trips):
                    next_deadline = boat_trips[i + 1][0]
                    max_late = next_deadline - dur - t_orig_depart
                else:
                    max_late = T - dur - t_orig_depart

                max_late = max(0, min(max_late, self.max_slack_timesteps))
                slots = list(range(max_late + 1))
                depart_slots[(boat, i)] = slots

                # Binary: which slot is chosen
                for s in slots:
                    depart_at[(boat, i, s)] = model.addVar(
                        vtype="B", name=f"depart_{boat}_trip{i+1}_s{s}"
                    )

                # At most one departure slot
                model.addCons(
                    quicksum(depart_at[(boat, i, s)] for s in slots) <= 1,
                    name=f"one_depart_{boat}_trip{i+1}",
                )

                # did_depart binary
                did_depart[(boat, i)] = model.addVar(
                    vtype="B", name=f"did_depart_{boat}_trip{i+1}"
                )
                model.addCons(
                    did_depart[(boat, i)]
                    == quicksum(depart_at[(boat, i, s)] for s in slots),
                    name=f"did_depart_link_{boat}_trip{i+1}",
                )

                # On-time departure
                fully_ready[(boat, i)] = depart_at[(boat, i, 0)]

                # --- at_sea variables and linking ---
                all_sea_times = set()
                for s in slots:
                    dep_t = t_orig_depart + s
                    for dt in range(dur):
                        t_sea = dep_t + dt
                        if t_sea < T:
                            all_sea_times.add(t_sea)

                for t in all_sea_times:
                    at_sea[(boat, i, t)] = model.addVar(
                        vtype="B", name=f"sea_{boat}_trip{i+1}_t{t}"
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
        """Aggregate at_sea across all trips per boat into a single boat_away indicator."""
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
                        vtype="B", name=f"away_{boat}_t{t}"
                    )
                    model.addCons(
                        boat_away[(boat, t)] == quicksum(sea_vars),
                        name=f"away_link_{boat}_t{t}",
                    )
        return boat_away

    def _add_drain_variables(self, model, T, trip_events, departure_ctx):
        """Return-drain: energy consumed during a trip, applied at the return timestep."""
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
                t: model.addVar(lb=0, ub=battery_cap, name=f"soc_{boat}_{t}")
                for t in timesteps
            }

            for t in timesteps:
                # Zero charger power while at sea
                if (boat, t) in boat_away:
                    model.addCons(
                        charger_power[charger_name][t]
                        <= charger.max_power * (1 - boat_away[(boat, t)]),
                        name=f"no_charge_away_{boat}_t{t}",
                    )

                # SOC dynamics
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
        """SOC >= energy_req at actual departure (hard constraint).

        Uses tight Big-M = battery_capacity per boat instead of a loose
        fixed constant.  This strengthens the LP relaxation and helps the
        solver prune branches much faster.
        """
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            # Tight Big-M: the max possible gap between SOC and energy_req
            # is at most battery_capacity (SOC ∈ [0, battery_cap], energy_req >= 0).
            big_m = self._boat_battery_cap[boat]

            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                t_orig_depart = t_deadline + 1
                slots = depart_slots[(boat, i)]

                for s in slots:
                    t_dl = t_orig_depart + s - 1
                    if t_dl < 0 or t_dl >= T:
                        continue
                    model.addCons(
                        soc[boat][t_dl]
                        >= energy_req - big_m * (1 - depart_at[(boat, i, s)]),
                        name=f"soc_req_{boat}_trip{i+1}_s{s}",
                    )

    def _add_symmetry_breaking(self, model, trip_events, boat_list, fully_ready):
        """Break symmetry across boats for same-index trips."""
        max_trips_per_boat = max(
            (len(trip_events.get(b, [])) for b in boat_list), default=0
        )
        for i in range(max_trips_per_boat):
            for k in range(1, len(boat_list)):
                b_prev = boat_list[k - 1]
                b_curr = boat_list[k]
                if (b_prev, i) in fully_ready and (b_curr, i) in fully_ready:
                    model.addCons(
                        fully_ready[(b_curr, i)] <= fully_ready[(b_prev, i)],
                        name=f"sym_{b_curr}_trip{i+1}",
                    )

    # ------------------------------------------------------------------ #
    #  Tariffs & objective                                                 #
    # ------------------------------------------------------------------ #

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
    ):
        depart_at = departure_ctx["depart_at"]
        depart_slots = departure_ctx["depart_slots"]
        did_depart = departure_ctx["did_depart"]

        # Energy cost
        energy_cost = quicksum(
            grid_import[t] * tariff_price[t] * self.timestep_hours for t in timesteps
        )

        # Decaying reward for departure time
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

        # Missed-trip penalty
        missed_penalty_terms = []
        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i in range(len(boat_trips)):
                if (boat, i) in did_depart:
                    missed_penalty_terms.append(2000 * (1 - did_depart[(boat, i)]))

        missed_trip_penalty = (
            quicksum(missed_penalty_terms) if missed_penalty_terms else 0
        )

        model.setObjective(
            -ready_reward + missed_trip_penalty + energy_cost,
            "minimize",
        )

    # ------------------------------------------------------------------ #
    #  Post-solve extraction                                               #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Trip extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_trip_events(self, energy_forecasts):
        """
        Extract trip events per boat.

        Returns:
            Dict[boat, List[(t_deadline, energy_required, duration)]]
        """
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

    # ------------------------------------------------------------------ #
    #  Database persistence                                                #
    # ------------------------------------------------------------------ #

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
