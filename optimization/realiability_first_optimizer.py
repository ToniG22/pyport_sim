"""
Reliability-first port charging optimizer (SCIP MILP) — SCALABLE VERSION.

Optimized for large-scale problems (20+ boats):
1) Per-boat energy constraints without explicit boat-charger assignments
2) Uses auxiliary continuous variables to track energy per boat
3) Boat availability constraints limit charging when boats are sailing
4) Cost-aware charging when no trips scheduled
5) Efficient DER utilization

This version scales much better than the binary assignment model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from pyscipopt import Model, quicksum

from models import Port, Trip, BoatState
from database import DatabaseManager
from forecasting import EnergyForecast


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

TRIP_SLOTS: List[Tuple[int, int]] = [
    (9, 0),  # 09:00, slot 0
    (14, 1),  # 14:00, slot 1
]

# Small margin to reduce SOC edge cases / tapering effects
ENERGY_MARGIN_FRAC = 0.05  # 5% margin for robustness

# If True, energy delivered during the departure timestep counts toward the deadline.
INCLUDE_DEPARTURE_TIMESTEP_IN_DEADLINE = False

# Conservative charging efficiency if boat doesn't expose one
DEFAULT_CHARGE_EFF = 0.95

# Objective weights
W_EARLY = 0.2  # Increased early charging bias to satisfy deadlines comfortably
W_TOTAL = 1.0
W_COST = 0.1  # Cost minimization when no trips scheduled
W_DEADLINE_PENALTY = (
    500.0  # Large penalty for missing boat energy deadlines (soft constraint)
)
W_URGENT = 2.0  # Weight for urgent charging (boats with trips coming soon)


# -----------------------------------------------------------------------------
# RESULT
# -----------------------------------------------------------------------------


@dataclass
class ReliabilityFirstOptimizationResult:
    status: str
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]
    bess_schedules: Dict[str, List[Tuple[datetime, float]]]
    peak_power_kw: float
    total_energy_kwh: float


# -----------------------------------------------------------------------------
# OPTIMIZER
# -----------------------------------------------------------------------------


class ReliabilityFirstOptimizer:
    def __init__(
        self, port: Port, db_manager: DatabaseManager, timestep_seconds: int = 900
    ):
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.dt_h = timestep_seconds / 3600.0

    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
        trip_assignments: Dict[str, List[Trip]],
    ) -> ReliabilityFirstOptimizationResult:
        T = len(energy_forecasts)
        timesteps = range(T)

        model = Model("reliability_first_scalable")
        model.hideOutput()
        model.setRealParam("limits/time", 30.0)

        num_chargers = len(self.port.chargers)
        num_boats = len(self.port.boats)

        if num_chargers == 0:
            return ReliabilityFirstOptimizationResult(
                status="no_chargers",
                charger_schedules={},
                bess_schedules={b.name: [] for b in self.port.bess_systems},
                peak_power_kw=0.0,
                total_energy_kwh=0.0,
            )

        # -----------------------------
        # PV power (kW) from forecast kWh/timestep
        # -----------------------------
        pv_kw: Dict[int, float] = {}
        for t in timesteps:
            pv_kw[t] = (
                float(energy_forecasts[t].power_active_production_kwh) / self.dt_h
                if self.dt_h > 0
                else 0.0
            )

        # -----------------------------
        # Boat availability per timestep
        # -----------------------------
        boat_available: Dict[int, Dict[int, bool]] = {}
        boats_available_count: Dict[int, int] = (
            {}
        )  # timestep -> count of available boats
        for b_idx, boat in enumerate(self.port.boats):
            boat_available[b_idx] = {}
            for t in timesteps:
                state = energy_forecasts[t].boat_states.get(boat.name, BoatState.IDLE)
                boat_available[b_idx][t] = state != BoatState.SAILING

        for t in timesteps:
            boats_available_count[t] = sum(
                1 for b_idx in range(num_boats) if boat_available[b_idx][t]
            )

        # -----------------------------
        # Tariff prices per timestep
        # -----------------------------
        tariff_price: Dict[int, float] = {}
        for t in timesteps:
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            tariff_price[t] = self.port.get_tariff_price(timestamp)

        # -----------------------------
        # Decision vars
        # -----------------------------
        # Charger power per charger and timestep (kW)
        p: Dict[int, Dict[int, object]] = {}
        for c in range(num_chargers):
            p[c] = {}
            ub = float(self.port.chargers[c].max_power)
            for t in timesteps:
                p[c][t] = model.addVar(name=f"p_{c}_{t}", vtype="C", lb=0.0, ub=ub)

        # Energy delivered per boat per timestep (kWh) - continuous auxiliary variable
        # This tracks how much energy each boat receives without explicit charger assignment
        boat_energy: Dict[int, Dict[int, object]] = (
            {}
        )  # boat_idx -> timestep -> energy_kWh
        max_boat_energy_per_timestep = max(
            float(c.max_power) * self.dt_h for c in self.port.chargers
        )

        for b_idx in range(num_boats):
            boat_energy[b_idx] = {}
            for t in timesteps:
                if boat_available[b_idx][t]:
                    # Boat can receive energy from available chargers
                    boat_energy[b_idx][t] = model.addVar(
                        name=f"boat_energy_{b_idx}_{t}",
                        vtype="C",
                        lb=0.0,
                        ub=max_boat_energy_per_timestep
                        * num_chargers,  # Max if all chargers charge this boat
                    )
                else:
                    # Boat is sailing - cannot receive energy
                    boat_energy[b_idx][t] = None

        # Grid import per timestep (kW)
        g: Dict[int, object] = {}
        for t in timesteps:
            g[t] = model.addVar(
                name=f"grid_{t}",
                vtype="C",
                lb=0.0,
                ub=float(self.port.contracted_power),
            )

        # BESS net power per timestep (kW): + discharge, - charge
        has_bess = bool(self.port.bess_systems)
        bnet: Dict[int, object] = {}
        bess_max_dis = 0.0
        bess_max_chg = 0.0
        bess_energy_available_kwh = 0.0
        bess_energy_capacity_kwh = 0.0

        if has_bess:
            bess_energy_available_kwh = float(energy_forecasts[0].bess_available_kwh)
            bess_energy_capacity_kwh = float(energy_forecasts[0].bess_capacity_kwh)
            bess_max_dis = sum(
                float(b.max_discharge_power) for b in self.port.bess_systems
            )
            bess_max_chg = sum(
                float(b.max_charge_power) for b in self.port.bess_systems
            )

            for t in timesteps:
                bnet[t] = model.addVar(
                    name=f"bnet_{t}",
                    vtype="C",
                    lb=-bess_max_chg,
                    ub=bess_max_dis,
                )

        def total_charger_power(t: int):
            return quicksum(p[c][t] for c in range(num_chargers))

        # -----------------------------
        # CONSTRAINTS
        # -----------------------------

        # Constraint 1: Power balance - grid + PV + BESS >= chargers
        for t in timesteps:
            total_p = total_charger_power(t)
            if has_bess:
                model.addCons(g[t] + pv_kw[t] + bnet[t] >= total_p, name=f"balance_{t}")
            else:
                model.addCons(g[t] + pv_kw[t] >= total_p, name=f"balance_{t}")

            # Grid import cannot exceed contracted power
            model.addCons(
                g[t] <= float(self.port.contracted_power), name=f"grid_limit_{t}"
            )

        # Constraint 2: Total charger power equals sum of boat energy (energy conservation)
        for t in timesteps:
            boat_energy_sum = quicksum(
                boat_energy[b_idx][t]
                for b_idx in range(num_boats)
                if boat_energy[b_idx][t] is not None
            )
            model.addCons(
                total_charger_power(t) * self.dt_h == boat_energy_sum,
                name=f"energy_conservation_{t}",
            )

        # Constraint 3: Boat energy is bounded by available chargers and boat availability
        for t in timesteps:
            if boats_available_count[t] == 0:
                # No boats available - all chargers should be idle
                for c in range(num_chargers):
                    model.addCons(p[c][t] == 0, name=f"no_boats_{c}_{t}")
            else:
                # Each boat can receive at most total charger power (kWh form)
                for b_idx in range(num_boats):
                    if boat_energy[b_idx][t] is not None:
                        model.addCons(
                            boat_energy[b_idx][t] <= total_charger_power(t) * self.dt_h,
                            name=f"boat_energy_bound_{b_idx}_{t}",
                        )

        # Constraint 4: BESS energy budgets (linearized)
        if has_bess:
            bdis: Dict[int, object] = {}
            bchg: Dict[int, object] = {}

            for t in timesteps:
                bdis[t] = model.addVar(
                    name=f"bdis_{t}", vtype="C", lb=0.0, ub=bess_max_dis
                )
                bchg[t] = model.addVar(
                    name=f"bchg_{t}", vtype="C", lb=0.0, ub=bess_max_chg
                )
                model.addCons(bnet[t] == bdis[t] - bchg[t], name=f"bnet_split_{t}")

            model.addCons(
                quicksum(bdis[t] * self.dt_h for t in timesteps)
                <= bess_energy_available_kwh * (1.0 - ENERGY_MARGIN_FRAC),
                name="bess_discharge_energy_limit",
            )
            model.addCons(
                quicksum(bchg[t] * self.dt_h for t in timesteps)
                <= bess_energy_capacity_kwh * (1.0 - ENERGY_MARGIN_FRAC),
                name="bess_charge_energy_limit",
            )

        # Constraint 5: PER-BOAT energy deadline constraints (SOFT constraints with penalties)
        slot_deadline_t: Dict[int, int] = {}
        for hour, slot in TRIP_SLOTS:
            idx = self._timestep_index_for_time(energy_forecasts, hour, 0)
            if idx is None:
                print(
                    f"     ⚠️ Deadline timestep not found for {hour:02d}:00 (slot {slot}). "
                    f"Constraints for this slot will be skipped."
                )
            else:
                slot_deadline_t[slot] = idx

        deadline_slack: Dict[int, Dict[int, object]] = (
            {}
        )  # boat_idx -> slot -> slack (kWh shortfall)

        if slot_deadline_t:
            boat_requirements = self._required_energy_per_boat_kwh(trip_assignments)

            for b_idx, boat in enumerate(self.port.boats):
                deadline_slack[b_idx] = {}

                for slot, deadline_idx in sorted(
                    slot_deadline_t.items(), key=lambda x: x[1]
                ):
                    req_kwh = boat_requirements.get(boat.name, {}).get(slot, 0.0)
                    if req_kwh <= 1e-6:
                        continue

                    upper = (
                        deadline_idx + 1
                        if INCLUDE_DEPARTURE_TIMESTEP_IN_DEADLINE
                        else deadline_idx
                    )
                    upper = max(0, min(upper, T))

                    delivered_to_boat = quicksum(
                        boat_energy[b_idx][t]
                        for t in range(upper)
                        if boat_energy[b_idx][t] is not None
                    )

                    slack = model.addVar(
                        name=f"deadline_slack_{b_idx}_slot_{slot}",
                        vtype="C",
                        lb=0.0,
                        ub=req_kwh * 2.0,
                    )
                    deadline_slack[b_idx][slot] = slack

                    model.addCons(
                        delivered_to_boat + slack
                        >= req_kwh * (1.0 + ENERGY_MARGIN_FRAC),
                        name=f"boat_{b_idx}_deadline_energy_slot_{slot}_soft",
                    )

        # -----------------------------
        # OBJECTIVE
        # -----------------------------

        energy_obj = quicksum(
            p[c][t] * self.dt_h * W_TOTAL
            for c in range(num_chargers)
            for t in timesteps
        )

        early_obj = quicksum(
            p[c][t] * self.dt_h * W_EARLY * (T - t) / max(1, T)
            for c in range(num_chargers)
            for t in timesteps
        )

        urgent_obj = 0.0
        for t in timesteps:
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            for b_idx in range(num_boats):
                if boat_energy[b_idx][t] is None:
                    continue

                boat = self.port.boats[b_idx]
                trips = trip_assignments.get(boat.name, [])
                hours_until_trip = float("inf")

                for hour, slot in TRIP_SLOTS:
                    if slot < len(trips) and trips[slot] is not None:
                        trip_time = timestamp.replace(hour=hour, minute=0, second=0)
                        if trip_time > timestamp:
                            hrs = (trip_time - timestamp).total_seconds() / 3600.0
                            if hrs < hours_until_trip:
                                hours_until_trip = hrs

                if hours_until_trip < 5.0:
                    urgency_weight = W_URGENT * (5.0 - hours_until_trip) / 5.0
                    urgent_obj += boat_energy[b_idx][t] * urgency_weight

        max_price = max(tariff_price.values()) if tariff_price.values() else 1.0
        cost_obj = 0.0
        for t in timesteps:
            if max_price > 0:
                normalized_price = tariff_price[t] / max_price
                cost_weight = W_COST * (1.0 - normalized_price)
            else:
                cost_weight = W_COST

            boats_available_for_cost = sum(
                1
                for b_idx in range(num_boats)
                if boat_available[b_idx][t]
                and self._has_no_imminent_trip(
                    b_idx, t, energy_forecasts, trip_assignments
                )
            )

            if boats_available_for_cost > 0:
                cost_obj += g[t] * self.dt_h * cost_weight

        deadline_penalty = 0.0
        if slot_deadline_t:
            for b_idx in range(num_boats):
                for slot in deadline_slack.get(b_idx, {}):
                    slack = deadline_slack[b_idx][slot]
                    deadline_penalty += slack * W_DEADLINE_PENALTY

        obj = energy_obj + early_obj + urgent_obj - cost_obj - deadline_penalty
        model.setObjective(obj, "maximize")

        # -----------------------------
        # Solve
        # -----------------------------
        model.optimize()
        status = model.getStatus()

        # -----------------------------
        # Extract schedules
        # -----------------------------
        charger_schedules = {c.name: [] for c in self.port.chargers}
        bess_schedules = {b.name: [] for b in self.port.bess_systems}

        peak_power = 0.0
        total_energy = 0.0

        if status in ["optimal", "bestsollimit", "timelimit"]:
            try:
                for t in timesteps:
                    ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                    total_p_t = 0.0

                    for c_idx, charger in enumerate(self.port.chargers):
                        try:
                            val = float(model.getVal(p[c_idx][t]))
                            if val < 0:
                                val = 0.0
                            charger_schedules[charger.name].append((ts, val))
                            total_p_t += val
                        except Exception as e:
                            print(
                                f"     ⚠️ Could not get charger {charger.name} power at {ts}: {e}"
                            )
                            charger_schedules[charger.name].append((ts, 0.0))

                    peak_power = max(peak_power, total_p_t)
                    total_energy += total_p_t * self.dt_h

                if has_bess:
                    for t in timesteps:
                        ts = forecast_date + timedelta(
                            seconds=t * self.timestep_seconds
                        )
                        try:
                            net = float(model.getVal(bnet[t]))
                            per = (
                                net / len(self.port.bess_systems)
                                if self.port.bess_systems
                                else 0.0
                            )
                            for bess in self.port.bess_systems:
                                bess_schedules[bess.name].append((ts, per))
                        except Exception as e:
                            print(f"     ⚠️ Could not get BESS power at {ts}: {e}")
                            for bess in self.port.bess_systems:
                                bess_schedules[bess.name].append((ts, 0.0))
                else:
                    for bess in self.port.bess_systems:
                        for t in timesteps:
                            ts = forecast_date + timedelta(
                                seconds=t * self.timestep_seconds
                            )
                            bess_schedules[bess.name].append((ts, 0.0))
            except Exception as e:
                print(f"     ⚠️ Error extracting solution values: {e}")
                print(f"     Model status: {status}")
        else:
            print(f"     ⚠️ SCIP status {status} — returning empty schedules.")

        return ReliabilityFirstOptimizationResult(
            status=str(status),
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            peak_power_kw=peak_power,
            total_energy_kwh=total_energy,
        )

    def save_schedules_to_db(self, result: ReliabilityFirstOptimizationResult) -> None:
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
            print(f"     ✓ Saved {len(schedules)} reliability-first schedule entries")

    # -----------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------

    @staticmethod
    def _timestep_index_for_time(
        forecasts: List[EnergyForecast],
        hour: int,
        minute: int,
    ) -> Optional[int]:
        for i, f in enumerate(forecasts):
            if f.timestamp.hour == hour and f.timestamp.minute == minute:
                return i
        return None

    def _has_no_imminent_trip(
        self,
        boat_idx: int,
        timestep: int,
        energy_forecasts: List[EnergyForecast],
        trip_assignments: Dict[str, List[Trip]],
    ) -> bool:
        """Check if boat has no imminent trip (within next 2 hours)."""
        boat = self.port.boats[boat_idx]
        trips = trip_assignments.get(boat.name, [])
        if not trips:
            return True

        current_time = energy_forecasts[timestep].timestamp
        for hour, slot in TRIP_SLOTS:
            if slot < len(trips):
                trip_time = current_time.replace(hour=hour, minute=0, second=0)
                hours_until_trip = (trip_time - current_time).total_seconds() / 3600.0
                if 0 < hours_until_trip < 2:
                    return False
        return True

    def _required_energy_per_boat_kwh(
        self, trip_assignments: Dict[str, List[Trip]]
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate required CHARGED energy per boat per slot deadline.

        Returns:
            Dict[boat_name, Dict[slot, required_kwh]]
        """
        boat_requirements: Dict[str, Dict[int, float]] = {}

        for boat in self.port.boats:
            boat_requirements[boat.name] = {}

            boat_now_kwh = float(boat.soc) * float(boat.battery_capacity)
            eff = float(getattr(boat, "charge_efficiency", DEFAULT_CHARGE_EFF))
            eff = max(0.7, min(1.0, eff))

            trips = trip_assignments.get(boat.name, [])
            cum_trip_energy = 0.0

            for hour, slot in TRIP_SLOTS:
                if slot < len(trips) and trips[slot] is not None:
                    trip_kwh = float(trips[slot].estimate_energy_required(boat.k))
                    trip_kwh_delivered = trip_kwh / eff
                    cum_trip_energy += trip_kwh_delivered

                need = max(0.0, cum_trip_energy - boat_now_kwh)
                boat_requirements[boat.name][slot] = need

        return boat_requirements

    def _required_energy_by_slot_kwh(
        self, trip_assignments: Dict[str, List[Trip]]
    ) -> Dict[int, float]:
        boat_reqs = self._required_energy_per_boat_kwh(trip_assignments)

        required_by_slot: Dict[int, float] = {slot: 0.0 for _, slot in TRIP_SLOTS}
        for _, slot_reqs in boat_reqs.items():
            for slot, req_kwh in slot_reqs.items():
                required_by_slot[slot] += req_kwh

        return required_by_slot
