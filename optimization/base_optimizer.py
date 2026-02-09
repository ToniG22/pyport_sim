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
        # Extract trip events per boat from forecasts
        # --------------------------------------------------
        trip_events = self._extract_trip_events(energy_forecasts)
        print("TRIP EVENTS", trip_events)

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
        # Shortfall & fully-ready variables (per boat, per trip)
        # --------------------------------------------------
        # shortfall[(boat, i)]  – continuous >= 0, energy missing at deadline
        # fully_ready[(boat, i)] – binary, 1 iff shortfall == 0
        # --------------------------------------------------
        shortfall = {}
        fully_ready = {}
        BIG_M = 1e4  # kWh, safely large

        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i in range(len(boat_trips)):
                _, energy_req, _ = boat_trips[i]
                shortfall[(boat, i)] = model.addVar(
                    lb=0, ub=energy_req, name=f"shortfall_{boat}_trip{i+1}"
                )
                fully_ready[(boat, i)] = model.addVar(
                    vtype="B", name=f"ready_{boat}_trip{i+1}"
                )

        # --------------------------------------------------
        # Build per-boat trip lookup structures
        # --------------------------------------------------
        # away_timesteps[boat]: set of timesteps when boat is at sea
        # return_drain[boat][t]: (energy_req, trip_index) for the trip
        #   that ends just before timestep t — we deduct
        #   (energy_req - shortfall) when the boat returns.
        # --------------------------------------------------
        away_timesteps = {}
        return_drain = {}

        for boat in self.boat_charger_assignments:
            away_ts = set()
            ret_drain = {}
            for i, (t_deadline, energy_req, dur) in enumerate(
                trip_events.get(boat, [])
            ):
                t_depart = t_deadline + 1
                t_return = t_depart + dur  # first timestep back at port
                for t in range(t_depart, min(t_return, T)):
                    away_ts.add(t)
                # Record drain info at return timestep
                if t_return < T:
                    ret_drain[t_return] = (energy_req, i)
            away_timesteps[boat] = away_ts
            return_drain[boat] = ret_drain

        # --------------------------------------------------
        # SOC variables (kWh) and dynamics per boat
        # --------------------------------------------------
        # soc[boat][t] = energy stored in battery at END of timestep t
        # --------------------------------------------------
        soc = {}
        boat_objects = {b.name: b for b in self.port.boats}
        charger_objects = {c.name: c for c in self.port.chargers}

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

            away = away_timesteps[boat]
            ret_drain_map = return_drain[boat]

            for t in timesteps:
                # --- Zero charger power while at sea ---
                if t in away:
                    model.addCons(
                        charger_power[charger_name][t] == 0,
                        name=f"away_{boat}_t{t}",
                    )

                # --- SOC dynamics ---
                soc_prev = soc[boat][t - 1] if t > 0 else initial_soc_kwh
                charge_energy = (
                    charger_power[charger_name][t] * charger_eff * self.timestep_hours
                )

                # Energy deducted when returning from a trip.
                # Actual drain = energy_req - shortfall  (boat only loses
                # what it actually had available).
                if t in ret_drain_map:
                    energy_req, trip_idx = ret_drain_map[t]
                    trip_drain = energy_req - shortfall[(boat, trip_idx)]
                else:
                    trip_drain = 0.0

                model.addCons(
                    soc[boat][t] == soc_prev + charge_energy - trip_drain,
                    name=f"soc_dyn_{boat}_{t}",
                )

        # --------------------------------------------------
        # Shortfall definition: shortfall >= energy_req - soc at deadline
        # --------------------------------------------------
        for boat in self.boat_charger_assignments:
            boat_trips = trip_events.get(boat, [])
            for i, (t_deadline, energy_req, dur) in enumerate(boat_trips):
                model.addCons(
                    shortfall[(boat, i)] >= energy_req - soc[boat][t_deadline],
                    name=f"shortfall_def_{boat}_trip{i+1}",
                )

        # --------------------------------------------------
        # Link fully_ready to shortfall:
        #   shortfall <= BIG_M * (1 - fully_ready)
        # i.e. fully_ready=1  =>  shortfall=0
        # --------------------------------------------------
        for key in shortfall:
            model.addCons(
                shortfall[key] <= BIG_M * (1 - fully_ready[key]),
                name=f"link_ready_{key[0]}_trip{key[1]+1}",
            )

        # --------------------------------------------------
        # Symmetry breaking: enforce ordering among boats
        # --------------------------------------------------
        boat_list = list(self.boat_charger_assignments.keys())
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
        # Objective
        # --------------------------------------------------
        # 1) Huge bonus for each fully-ready trip (maximise count)
        # 2) Penalise remaining shortfall (drives partial gaps to 0)
        # 3) Minimise energy cost
        # --------------------------------------------------
        n_boats = len(boat_list)

        # Large reward per fully-ready boat-trip, with small
        # tie-breaking bonus favouring earlier boats.
        ready_reward = quicksum(
            (1000 + (n_boats - k) * 0.1) * fully_ready[(boat_list[k], i)]
            for i in range(max_trips_per_boat)
            for k in range(n_boats)
            if (boat_list[k], i) in fully_ready
        )

        # Penalise any remaining shortfall so the optimizer
        # still tries to reduce partial gaps.
        shortfall_penalty = quicksum(10 * shortfall[key] for key in shortfall)

        model.setObjective(
            -ready_reward + shortfall_penalty + energy_cost,
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
        # Extract schedules
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

                # Trip starts here
                if prev_avail == 1 and curr_avail == 0:
                    start_t = t

                    # Trip duration
                    dur = 0
                    while (
                        t < T
                        and energy_forecasts[t].boat_available.get(boat.name, 1) == 0
                    ):
                        dur += 1
                        t += 1

                    # Deadline is timestep BEFORE departure
                    deadline_t = start_t - 1

                    # Energy requirement at deadline
                    energy_req = energy_forecasts[
                        deadline_t
                    ].boat_required_energy_kwh.get(boat.name, 0.0)

                    trips[boat.name].append((deadline_t, energy_req, dur))
                else:
                    t += 1

        return trips

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
