"""
Reliability-first port charging optimizer (SCIP MILP) — FIXED VERSION.

Fixes included vs your current version:
1) Deadline timestep semantics: configurable inclusion of the departure timestep.
   - If trips depart at the START of the timestep (most common): EXCLUDE departure timestep.
   - If trips depart at the END of the timestep window: INCLUDE departure timestep.
   Default here matches your engine: trips trigger when hour==start_hour and minute < timestep/60,
   i.e., at the START → exclude the departure timestep.

2) Charging efficiency / taper safety:
   - Applies an efficiency factor to required energy (if boats expose charge_efficiency),
     otherwise uses a conservative default.
   - Keeps ENERGY_MARGIN_FRAC to cover taper / rounding.

3) Robust trip slot indexing:
   - Finds exact (hour, minute) indices; fails loudly (prints warning) if not found.

4) BESS: keep your aggregated net-power model, but:
   - Use energy_forecasts[t].bess_available_kwh/capacity_kwh (t=0) as before.
   - Preserve split vars bdis/bchg for linear energy budgets.

Important note (modeling trade-off):
- This remains an aggregated model (fleet-level energy-by-deadline), not per-boat SOC trajectories.
  It is usually enough to fix your "50% reliability" failure mode and is fast/scalable.

Expected outcome:
- 5 vessels, contracted 80 kW, no DER should reach 100% reliability,
  assuming physically feasible (it is, based on your earlier baseline behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from pyscipopt import Model, quicksum

from models import Port, Trip
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
ENERGY_MARGIN_FRAC = 0.05  # bumped to 5% for robustness

# If True, energy delivered during the departure timestep counts toward the deadline.
# Your engine starts trips at the start of the timestep window → keep False.
INCLUDE_DEPARTURE_TIMESTEP_IN_DEADLINE = False

# Conservative charging efficiency if boat doesn't expose one
DEFAULT_CHARGE_EFF = 0.95

# Objective weights
W_EARLY = 0.05  # slightly stronger early bias to satisfy deadlines comfortably
W_TOTAL = 1.0


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

    # -----------------------------
    # Public API expected by engine
    # -----------------------------
    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
        trip_assignments: Dict[str, List[Trip]],
    ) -> ReliabilityFirstOptimizationResult:
        T = len(energy_forecasts)
        timesteps = range(T)

        model = Model("reliability_first_fixed")
        model.hideOutput()
        model.setRealParam("limits/time", 30.0)

        num_chargers = len(self.port.chargers)
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
        # Decision vars
        # -----------------------------
        # Charger power per charger and timestep (kW)
        p: Dict[int, Dict[int, object]] = {}
        for c in range(num_chargers):
            p[c] = {}
            ub = float(self.port.chargers[c].max_power)
            for t in timesteps:
                p[c][t] = model.addVar(name=f"p_{c}_{t}", vtype="C", lb=0.0, ub=ub)

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
        # Constraints: balance + grid limit
        # -----------------------------
        for t in timesteps:
            total_p = total_charger_power(t)
            if has_bess:
                # g + pv + bnet >= chargers
                model.addCons(g[t] + pv_kw[t] + bnet[t] >= total_p, name=f"balance_{t}")
            else:
                model.addCons(g[t] + pv_kw[t] >= total_p, name=f"balance_{t}")

            model.addCons(
                g[t] <= float(self.port.contracted_power), name=f"grid_limit_{t}"
            )

        # -----------------------------
        # BESS energy budgets (linearized)
        # -----------------------------
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

        # -----------------------------
        # DEADLINE ENERGY CONSTRAINTS (critical for reliability)
        # -----------------------------
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

        if slot_deadline_t:
            required_by_slot_kwh = self._required_energy_by_slot_kwh(trip_assignments)

            for slot, deadline_idx in sorted(
                slot_deadline_t.items(), key=lambda x: x[1]
            ):
                req_kwh = float(required_by_slot_kwh.get(slot, 0.0))
                if req_kwh <= 1e-6:
                    continue

                # Decide inclusive/exclusive horizon
                upper = (
                    deadline_idx + 1
                    if INCLUDE_DEPARTURE_TIMESTEP_IN_DEADLINE
                    else deadline_idx
                )
                upper = max(0, min(upper, T))

                delivered_up_to_deadline = quicksum(
                    p[c][t] * self.dt_h
                    for c in range(num_chargers)
                    for t in range(upper)
                )

                model.addCons(
                    delivered_up_to_deadline >= req_kwh * (1.0 + ENERGY_MARGIN_FRAC),
                    name=f"deadline_energy_slot_{slot}",
                )

        # -----------------------------
        # Objective: maximize total delivered energy + early bias
        # -----------------------------
        obj = quicksum(
            p[c][t] * self.dt_h * (W_TOTAL + W_EARLY * (T - t) / max(1, T))
            for c in range(num_chargers)
            for t in timesteps
        )
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
            for t in timesteps:
                ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                total_p_t = 0.0

                for c_idx, charger in enumerate(self.port.chargers):
                    val = float(model.getVal(p[c_idx][t]))
                    if val < 0:
                        val = 0.0
                    charger_schedules[charger.name].append((ts, val))
                    total_p_t += val

                peak_power = max(peak_power, total_p_t)
                total_energy += total_p_t * self.dt_h

            if has_bess:
                for t in timesteps:
                    ts = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                    net = float(model.getVal(bnet[t]))
                    per = (
                        net / len(self.port.bess_systems)
                        if self.port.bess_systems
                        else 0.0
                    )
                    for bess in self.port.bess_systems:
                        bess_schedules[bess.name].append((ts, per))
            else:
                for bess in self.port.bess_systems:
                    for t in timesteps:
                        ts = forecast_date + timedelta(
                            seconds=t * self.timestep_seconds
                        )
                        bess_schedules[bess.name].append((ts, 0.0))
        else:
            # If infeasible, still return empty schedules (engine can handle no schedule)
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

    def _required_energy_by_slot_kwh(
        self, trip_assignments: Dict[str, List[Trip]]
    ) -> Dict[int, float]:
        """
        Aggregate required CHARGED energy by each slot deadline.

        For each boat:
        - compute current stored energy (kWh) from SOC * capacity
        - compute cumulative trip energy required up to each slot
        - required charged energy by that slot is max(0, cum_trip_kwh - current_energy)

        Fix: account for charging efficiency so the optimizer doesn't under-provision.
        """
        # Current energy per boat (kWh)
        boat_now_kwh: Dict[str, float] = {
            b.name: float(b.soc) * float(b.battery_capacity) for b in self.port.boats
        }

        # Per boat, per slot trip requirement (kWh)
        boat_trip_req: Dict[str, Dict[int, float]] = {
            b.name: {} for b in self.port.boats
        }

        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            # get efficiency if present, else conservative default
            eff = float(getattr(boat, "charge_efficiency", DEFAULT_CHARGE_EFF))
            eff = max(0.7, min(1.0, eff))  # clamp sanity

            for hour, slot in TRIP_SLOTS:
                if slot < len(trips) and trips[slot] is not None:
                    trip_kwh = float(trips[slot].estimate_energy_required(boat.k))
                    # convert required "battery energy" to required "delivered energy"
                    trip_kwh_delivered = trip_kwh / eff
                    boat_trip_req[boat.name][slot] = trip_kwh_delivered
                else:
                    boat_trip_req[boat.name][slot] = 0.0

        # Aggregate cumulative requirement by slot
        required_by_slot: Dict[int, float] = {slot: 0.0 for _, slot in TRIP_SLOTS}

        for boat in self.port.boats:
            cum = 0.0
            for _, slot in TRIP_SLOTS:
                cum += boat_trip_req[boat.name].get(slot, 0.0)
                need = max(0.0, cum - boat_now_kwh[boat.name])
                required_by_slot[slot] += need

        return required_by_slot
