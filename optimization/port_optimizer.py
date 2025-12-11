"""Port energy optimization using SCIP."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pyscipopt import Model, quicksum
from models import Port, Boat, BoatState
from database import DatabaseManager
from forecasting import EnergyForecast


# ===========================================================================
# CONFIGURATION CONSTANTS FOR OPTIMIZATION CONSTRAINTS
# ===========================================================================

# BESS constraints
BESS_SOC_MIN = 0.10          # Minimum SOC (10%) - don't discharge below this
BESS_SOC_MAX = 0.90          # Maximum SOC (90%) - don't charge above this
BESS_END_OF_DAY_TARGET = 0.50  # Target SOC at end of day for next day flexibility
BESS_END_OF_DAY_TOLERANCE = 0.20  # Allow ±20% deviation from target (30%-70% range)
BESS_RAMP_RATE_FACTOR = 0.50  # Max power change per timestep as fraction of max power
BESS_CYCLING_PENALTY = 0.01  # Small penalty per kW of power change to reduce cycling

# Boat constraints
BOAT_SOC_SAFETY_RESERVE = 0.10  # Keep 10% reserve for safety/emergencies
BOAT_MIN_SOC_FOR_TRIP = 0.15  # Additional buffer above trip requirement

# Charger constraints
CHARGER_MIN_POWER_FACTOR = 0.10  # Minimum operating power as fraction of max (10%)

# Objective weights
WEIGHT_BOAT_SHORTFALL = 100000  # HUGE penalty for not meeting boat energy needs
WEIGHT_GRID_COST = 1.0         # Weight for grid energy cost
WEIGHT_BESS_CYCLING = 0.001    # Small weight to discourage unnecessary cycling
WEIGHT_END_OF_DAY_SOC = 10.0   # Moderate penalty for ending day at wrong SOC


@dataclass
class OptimizationResult:
    """Result from optimization."""
    
    status: str  # 'optimal', 'feasible', 'infeasible'
    objective_value: float
    charger_schedules: Dict[str, List[Tuple[datetime, float]]]  # {charger_name: [(time, power)]}
    bess_schedules: Dict[str, List[Tuple[datetime, float]]]  # {bess_name: [(time, power)]}
    boat_charging_plan: Dict[str, List[Tuple[datetime, float]]]  # {boat_name: [(time, soc)]}
    energy_shortfalls: Dict[str, float]  # {boat_name: shortfall_kwh} - energy that couldn't be provided


class PortOptimizer:
    """Optimize port energy management using SCIP."""
    
    def __init__(
        self,
        port: Port,
        db_manager: DatabaseManager,
        timestep_seconds: int = 900
    ):
        """
        Initialize the port optimizer.
        
        Args:
            port: Port instance
            db_manager: Database manager
            timestep_seconds: Simulation timestep in seconds
        """
        self.port = port
        self.db_manager = db_manager
        self.timestep_seconds = timestep_seconds
        self.timestep_hours = timestep_seconds / 3600.0
    
    def optimize_daily_schedule(
        self,
        forecast_date: datetime,
        energy_forecasts: List[EnergyForecast],
        trip_assignments: Dict[str, List]
    ) -> OptimizationResult:
        """
        Optimize charger and BESS schedules for a day.
        
        This optimization model focuses on:
        - Minimizing grid usage by using PV and BESS optimally
        - Ensuring boats have enough charge for trips (with safety buffer)
        - Respecting BESS operating constraints (SOC limits, ramp rates)
        - Not exceeding contracted grid power
        - Minimizing energy costs based on time-of-use tariffs
        
        Args:
            forecast_date: Date to optimize for
            energy_forecasts: Energy forecasts for the day
            trip_assignments: Trip assignments per boat
            
        Returns:
            OptimizationResult with optimal schedules
        """
        print("     🔧 Optimizing energy schedule (with constraints)...")
        
        # Create optimization model
        model = Model("port_energy_optimization")
        model.hideOutput()  # Suppress solver output
        model.setRealParam('limits/time', 30.0)  # 30 second time limit
        
        # Number of timesteps (24 hours)
        T = len(energy_forecasts)
        timesteps = list(range(T))
        
        # ===================================================================
        # DECISION VARIABLES
        # ===================================================================
        
        # BESS variables
        bess_power = {}       # Net power: positive=discharge, negative=charge
        bess_soc = {}         # State of charge (0-1)
        bess_charge = {}      # Charging power (>=0)
        bess_discharge = {}   # Discharging power (>=0)
        bess_is_charging = {} # Binary: 1 if charging, 0 if discharging
        
        for bess in self.port.bess_systems:
            bess_power[bess.name] = {}
            bess_soc[bess.name] = {}
            bess_charge[bess.name] = {}
            bess_discharge[bess.name] = {}
            bess_is_charging[bess.name] = {}
            
            # Use BESS-specific limits if available, otherwise use defaults
            soc_min = getattr(bess, 'soc_min', BESS_SOC_MIN)
            soc_max = getattr(bess, 'soc_max', BESS_SOC_MAX)
            
            for t in timesteps:
                # Net power variable
                bess_power[bess.name][t] = model.addVar(
                    name=f"bess_power_{bess.name}_t{t}",
                    vtype="C",
                    lb=-bess.max_charge_power,  # Negative = charging
                    ub=bess.max_discharge_power  # Positive = discharging
                )
                
                # SOC variable - constrained to operating range
                bess_soc[bess.name][t] = model.addVar(
                    name=f"bess_soc_{bess.name}_t{t}",
                    vtype="C",
                    lb=soc_min,
                    ub=soc_max
                )
                
                # Separate charge/discharge variables for efficiency handling
                bess_charge[bess.name][t] = model.addVar(
                    name=f"bess_charge_{bess.name}_t{t}",
                    vtype="C",
                    lb=0,
                    ub=bess.max_charge_power
                )
                
                bess_discharge[bess.name][t] = model.addVar(
                    name=f"bess_discharge_{bess.name}_t{t}",
                    vtype="C",
                    lb=0,
                    ub=bess.max_discharge_power
                )
                
                # Binary variable to prevent simultaneous charge/discharge
                bess_is_charging[bess.name][t] = model.addVar(
                    name=f"bess_is_charging_{bess.name}_t{t}",
                    vtype="B"  # Binary
                )
        
        # Charger power variables
        charger_power = {}
        charger_is_on = {}  # Binary: 1 if charger is operating
        
        for charger in self.port.chargers:
            charger_power[charger.name] = {}
            charger_is_on[charger.name] = {}
            
            min_power = charger.max_power * CHARGER_MIN_POWER_FACTOR
            
            for t in timesteps:
                charger_power[charger.name][t] = model.addVar(
                    name=f"charger_{charger.name}_t{t}",
                    vtype="C",
                    lb=0,
                    ub=charger.max_power
                )
                
                # Binary variable for charger on/off (for minimum power constraint)
                charger_is_on[charger.name][t] = model.addVar(
                    name=f"charger_on_{charger.name}_t{t}",
                    vtype="B"
                )
        
        # Grid import variable (for contracted power constraint)
        grid_import = {}
        for t in timesteps:
            grid_import[t] = model.addVar(
                name=f"grid_import_t{t}",
                vtype="C",
                lb=0,
                ub=self.port.contracted_power  # Hard limit on grid import
            )
        
        # ===================================================================
        # CONSTRAINTS
        # ===================================================================
        
        # Pre-compute boat availability at each timestep
        boat_available = {}
        for boat in self.port.boats:
            boat_available[boat.name] = {}
            for t in timesteps:
                forecast = energy_forecasts[t]
                boat_state = forecast.boat_states.get(boat.name, BoatState.IDLE)
                boat_available[boat.name][t] = (boat_state != BoatState.SAILING)
        
        # Pre-compute if ANY boat is available at each timestep
        any_boat_available = {}
        for t in timesteps:
            forecast = energy_forecasts[t]
            any_boat_available[t] = any(
                forecast.boat_states.get(boat.name, BoatState.IDLE) != BoatState.SAILING
                for boat in self.port.boats
            )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 1: BESS SOC dynamics with efficiency
        # ---------------------------------------------------------------------
        for bess in self.port.bess_systems:
            soc_min = getattr(bess, 'soc_min', BESS_SOC_MIN)
            soc_max = getattr(bess, 'soc_max', BESS_SOC_MAX)
            
            for t in timesteps:
                if t == 0:
                    # Initial BESS SOC (clamped to operating range)
                    initial_soc = max(soc_min, min(soc_max, bess.current_soc))
                    model.addCons(
                        bess_soc[bess.name][t] == initial_soc,
                        name=f"initial_bess_soc_{bess.name}"
                    )
                else:
                    # SOC dynamics: charging adds energy, discharging removes energy
                    # Charging: energy_in = charge_power * time * efficiency
                    # Discharging: energy_out = discharge_power * time / efficiency
                    energy_stored = bess_charge[bess.name][t-1] * self.timestep_hours * bess.efficiency
                    energy_removed = bess_discharge[bess.name][t-1] * self.timestep_hours / bess.efficiency
                    
                    soc_change = (energy_stored - energy_removed) / bess.capacity
                    
                    model.addCons(
                        bess_soc[bess.name][t] == bess_soc[bess.name][t-1] + soc_change,
                        name=f"bess_soc_dynamics_{bess.name}_t{t}"
                    )
                
                # Power balance: net power = discharge - charge
                model.addCons(
                    bess_power[bess.name][t] == bess_discharge[bess.name][t] - bess_charge[bess.name][t],
                    name=f"bess_power_balance_{bess.name}_t{t}"
                )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 2: Prevent simultaneous BESS charge and discharge (big-M)
        # ---------------------------------------------------------------------
        for bess in self.port.bess_systems:
            M = max(bess.max_charge_power, bess.max_discharge_power) + 1
            
            for t in timesteps:
                # If charging (is_charging=1), discharge must be 0
                model.addCons(
                    bess_discharge[bess.name][t] <= M * (1 - bess_is_charging[bess.name][t]),
                    name=f"no_discharge_when_charging_{bess.name}_t{t}"
                )
                
                # If discharging (is_charging=0), charge must be 0
                model.addCons(
                    bess_charge[bess.name][t] <= M * bess_is_charging[bess.name][t],
                    name=f"no_charge_when_discharging_{bess.name}_t{t}"
                )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 3: BESS power ramp rate limits
        # ---------------------------------------------------------------------
        for bess in self.port.bess_systems:
            max_ramp = max(bess.max_charge_power, bess.max_discharge_power) * BESS_RAMP_RATE_FACTOR
            
            for t in timesteps[1:]:  # Start from t=1
                # Limit power increase
                model.addCons(
                    bess_power[bess.name][t] - bess_power[bess.name][t-1] <= max_ramp,
                    name=f"bess_ramp_up_{bess.name}_t{t}"
                )
                
                # Limit power decrease
                model.addCons(
                    bess_power[bess.name][t-1] - bess_power[bess.name][t] <= max_ramp,
                    name=f"bess_ramp_down_{bess.name}_t{t}"
                )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 4: BESS end-of-day SOC target (soft constraint via objective)
        # We add a slack variable and penalize deviation from target
        # ---------------------------------------------------------------------
        bess_end_soc_deviation = {}
        for bess in self.port.bess_systems:
            target_soc = getattr(bess, 'initial_soc', BESS_END_OF_DAY_TARGET)
            
            # Slack variable for deviation from target
            bess_end_soc_deviation[bess.name] = model.addVar(
                name=f"bess_end_soc_dev_{bess.name}",
                vtype="C",
                lb=0
            )
            
            # Absolute deviation constraint (linearized)
            last_t = timesteps[-1]
            model.addCons(
                bess_end_soc_deviation[bess.name] >= bess_soc[bess.name][last_t] - target_soc,
                name=f"bess_end_soc_dev_pos_{bess.name}"
            )
            model.addCons(
                bess_end_soc_deviation[bess.name] >= target_soc - bess_soc[bess.name][last_t],
                name=f"bess_end_soc_dev_neg_{bess.name}"
            )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 5: Charger minimum operating power (when on)
        # ---------------------------------------------------------------------
        for charger in self.port.chargers:
            min_power = charger.max_power * CHARGER_MIN_POWER_FACTOR
            
            for t in timesteps:
                # If charger is on, power must be at least min_power
                model.addCons(
                    charger_power[charger.name][t] >= min_power * charger_is_on[charger.name][t],
                    name=f"charger_min_power_{charger.name}_t{t}"
                )
                
                # If charger is off, power must be 0
                model.addCons(
                    charger_power[charger.name][t] <= charger.max_power * charger_is_on[charger.name][t],
                    name=f"charger_max_when_on_{charger.name}_t{t}"
                )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 6: Chargers can only operate when boats are available
        # ---------------------------------------------------------------------
        for charger in self.port.chargers:
            for t in timesteps:
                if not any_boat_available[t]:
                    # All boats are sailing - charger must be off
                    model.addCons(
                        charger_power[charger.name][t] == 0,
                        name=f"charger_{charger.name}_off_when_all_sailing_t{t}"
                    )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 7: Boat energy requirements with safety buffer
        # ---------------------------------------------------------------------
        boat_energy_shortfall = {}
        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            if trips:
                # Calculate TOTAL energy needed for all trips + safety buffer
                total_trip_energy = sum(trip.estimate_energy_required(boat.k) for trip in trips)
                
                # Energy boat currently has
                current_energy = boat.soc * boat.battery_capacity
                
                # Safety reserve energy (keep BOAT_SOC_SAFETY_RESERVE in battery)
                safety_reserve = BOAT_SOC_SAFETY_RESERVE * boat.battery_capacity
                
                # Additional buffer above trip requirement
                trip_buffer = BOAT_MIN_SOC_FOR_TRIP * boat.battery_capacity
                
                # Total energy needed = fill battery + trip energy + buffers
                energy_needed = (
                    boat.battery_capacity - current_energy  # Fill to 100%
                    + total_trip_energy                      # Trip consumption
                    + safety_reserve                         # Keep reserve
                    + trip_buffer                            # Additional buffer
                )
                
                # Clamp to reasonable maximum (don't ask for more than 2x battery capacity)
                energy_needed = min(energy_needed, 2 * boat.battery_capacity)
                
                # Total energy provided by chargers to this boat (when available)
                available_timesteps = [t for t in timesteps if boat_available[boat.name][t]]
                total_charging_energy = quicksum(
                    charger_power[charger.name][t] * charger.efficiency * self.timestep_hours
                    for charger in self.port.chargers
                    for t in available_timesteps
                )
                
                # Slack variable for energy shortfall
                boat_energy_shortfall[boat.name] = model.addVar(
                    name=f"shortfall_{boat.name}",
                    vtype="C",
                    lb=0
                )
                
                # Constraint: charging + shortfall >= needed
                model.addCons(
                    total_charging_energy + boat_energy_shortfall[boat.name] >= energy_needed,
                    name=f"boat_{boat.name}_energy_requirement"
                )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 8: Grid import limit (contracted power)
        # ---------------------------------------------------------------------
        for t in timesteps:
            forecast = energy_forecasts[t]
            
            # Total charger power
            total_charger_power = quicksum(
                charger_power[charger.name][t]
                for charger in self.port.chargers
            )
            
            # BESS charging power (counts as load)
            total_bess_charging = quicksum(
                bess_charge[bess.name][t]
                for bess in self.port.bess_systems
            )
            
            # BESS discharging power (provides power, reduces grid need)
            total_bess_discharging = quicksum(
                bess_discharge[bess.name][t]
                for bess in self.port.bess_systems
            )
            
            # PV production (kW)
            pv_power_kw = forecast.power_active_production_kwh / self.timestep_hours
            
            # Grid import = total load - local generation - BESS discharge
            # Grid import + PV + BESS_discharge >= chargers + BESS_charge
            model.addCons(
                grid_import[t] >= total_charger_power + total_bess_charging - pv_power_kw - total_bess_discharging,
                name=f"grid_import_balance_t{t}"
            )
            
            # Grid import cannot exceed contracted power (already in variable bounds)
            # But add explicit constraint for clarity
            model.addCons(
                grid_import[t] <= self.port.contracted_power,
                name=f"grid_import_limit_t{t}"
            )
        
        # ---------------------------------------------------------------------
        # CONSTRAINT 9: Power balance (ensure supply meets demand)
        # ---------------------------------------------------------------------
        for t in timesteps:
            forecast = energy_forecasts[t]
            
            total_charger_power = quicksum(
                charger_power[charger.name][t]
                for charger in self.port.chargers
            )
            
            total_bess_power = quicksum(
                bess_power[bess.name][t]
                for bess in self.port.bess_systems
            )
            
            pv_power_kw = forecast.power_active_production_kwh / self.timestep_hours
            
            # Power balance: Grid + PV + BESS_discharge >= Chargers + BESS_charge
            # With BESS_power = discharge - charge, this becomes:
            # Grid + PV + BESS_power >= Chargers
            model.addCons(
                grid_import[t] + pv_power_kw + total_bess_power >= total_charger_power,
                name=f"power_balance_t{t}"
            )
        
        # ===================================================================
        # OBJECTIVE FUNCTION
        # ===================================================================
        
        # Term 1: HUGE penalty for boat energy shortfalls (MUST charge boats!)
        boat_shortfall_penalty = quicksum(boat_energy_shortfall.values()) * WEIGHT_BOAT_SHORTFALL
        
        # Term 2: Grid energy cost (time-of-use pricing)
        total_grid_cost = 0
        for t in timesteps:
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            price_per_kwh = self.port.get_tariff_price(timestamp)
            
            # Cost = grid import * price * timestep_hours
            grid_energy_kwh = grid_import[t] * self.timestep_hours
            total_grid_cost += grid_energy_kwh * price_per_kwh * WEIGHT_GRID_COST
        
        # Term 3: BESS cycling penalty (reduce wear and tear)
        bess_cycling_cost = 0
        for bess in self.port.bess_systems:
            for t in timesteps:
                # Penalize both charging and discharging to reduce unnecessary cycling
                bess_cycling_cost += (
                    bess_charge[bess.name][t] + bess_discharge[bess.name][t]
                ) * WEIGHT_BESS_CYCLING
        
        # Term 4: End-of-day SOC deviation penalty
        end_soc_penalty = quicksum(bess_end_soc_deviation.values()) * WEIGHT_END_OF_DAY_SOC
        
        # Combined objective
        model.setObjective(
            boat_shortfall_penalty + total_grid_cost + bess_cycling_cost + end_soc_penalty,
            "minimize"
        )
        
        # ===================================================================
        # SOLVE
        # ===================================================================
        
        model.optimize()
        
        # ===================================================================
        # EXTRACT RESULTS
        # ===================================================================
        
        status = model.getStatus()
        
        # Initialize empty structures
        charger_schedules = {}
        bess_schedules = {}
        boat_plan = {}
        energy_shortfalls = {}
        
        # Only extract solution if optimization succeeded
        if status in ['optimal', 'bestsollimit', 'timelimit']:
            try:
                obj_value = model.getObjVal()
                
                # Extract charger schedules
                for charger in self.port.chargers:
                    schedule = []
                    for t in timesteps:
                        power_val = model.getVal(charger_power[charger.name][t])
                        timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                        schedule.append((timestamp, power_val))
                    charger_schedules[charger.name] = schedule
                
                # Extract BESS schedules (net power: positive=discharge, negative=charge)
                for bess in self.port.bess_systems:
                    schedule = []
                    for t in timesteps:
                        net_power = model.getVal(bess_power[bess.name][t])
                        timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
                        schedule.append((timestamp, net_power))
                    bess_schedules[bess.name] = schedule
                
                # Extract energy shortfalls for each boat
                for boat in self.port.boats:
                    if boat.name in boat_energy_shortfall:
                        shortfall_val = model.getVal(boat_energy_shortfall[boat.name])
                        if shortfall_val > 0.01:  # Only report significant shortfalls (>0.01 kWh)
                            energy_shortfalls[boat.name] = shortfall_val
                
                # Boat SOC plan (empty in simplified model)
                for boat in self.port.boats:
                    boat_plan[boat.name] = []
                
                # Print optimization summary
                self._print_optimization_summary(
                    status, obj_value, energy_shortfalls, 
                    model, bess_soc, bess_end_soc_deviation, timesteps
                )
                
            except Exception as e:
                print(f"     ⚠️  Error extracting solution: {e}")
                obj_value = float('inf')
        else:
            print(f"     ⚠️  Optimization failed: {status}")
            obj_value = float('inf')
        
        return OptimizationResult(
            status=status,
            objective_value=obj_value,
            charger_schedules=charger_schedules,
            bess_schedules=bess_schedules,
            boat_charging_plan=boat_plan,
            energy_shortfalls=energy_shortfalls
        )
    
    def _print_optimization_summary(
        self, 
        status: str, 
        obj_value: float, 
        energy_shortfalls: Dict[str, float],
        model: Model,
        bess_soc: Dict,
        bess_end_soc_deviation: Dict,
        timesteps: List[int]
    ):
        """Print a summary of the optimization results."""
        print(f"     ✓ Optimization complete: {status}")
        print(f"       Objective value: {obj_value:.2f}")
        
        # Print BESS end-of-day SOC
        for bess in self.port.bess_systems:
            end_soc = model.getVal(bess_soc[bess.name][timesteps[-1]])
            deviation = model.getVal(bess_end_soc_deviation[bess.name])
            target = getattr(bess, 'initial_soc', BESS_END_OF_DAY_TARGET)
            print(f"       {bess.name} end SOC: {end_soc:.1%} (target: {target:.0%}, deviation: {deviation:.1%})")
        
        # Warn about energy shortfalls
        if energy_shortfalls:
            print("     ⚠️  Energy shortfalls detected:")
            for boat_name, shortfall_kwh in energy_shortfalls.items():
                boat = next(b for b in self.port.boats if b.name == boat_name)
                shortfall_pct = (shortfall_kwh / boat.battery_capacity) * 100
                print(f"       {boat_name}: {shortfall_kwh:.2f} kWh ({shortfall_pct:.1f}% of battery)")
    
    def _estimate_trip_consumption_soc(
        self,
        boat: Boat,
        timestamp: datetime,
        trips: List
    ) -> float:
        """
        Estimate SOC consumption for a boat at a given timestamp.
        
        Args:
            boat: Boat instance
            timestamp: Timestamp to check
            trips: List of trips assigned to boat
            
        Returns:
            Estimated SOC decrease (0 if not on trip)
        """
        hour = timestamp.hour
        
        # Check if boat is on a trip
        # Morning trip: ~9:00-12:30
        if hour >= 9 and hour < 13 and len(trips) > 0:
            trip = trips[0]
            avg_power = trip.estimate_energy_required(boat.k) / (trip.duration / 3600)
            energy_kwh = avg_power * self.timestep_hours
            return energy_kwh / boat.battery_capacity
        
        # Afternoon trip: ~14:00-17:30
        elif hour >= 14 and hour < 18 and len(trips) > 1:
            trip = trips[1]
            avg_power = trip.estimate_energy_required(boat.k) / (trip.duration / 3600)
            energy_kwh = avg_power * self.timestep_hours
            return energy_kwh / boat.battery_capacity
        
        return 0.0
    
    def save_schedules_to_db(self, result: OptimizationResult) -> None:
        """
        Save optimization results to the scheduling database table.
        
        Args:
            result: Optimization result with schedules
        """
        schedules = []
        power_setpoint_met = self.db_manager.get_metric_id("power_setpoint")
        
        # Save charger schedules
        charger_count = 0
        for charger_name, schedule in result.charger_schedules.items():
            charger_src = self.db_manager.get_or_create_source(charger_name, "charger")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, charger_src, power_setpoint_met, str(power)))
                charger_count += 1
        
        # Save BESS schedules
        bess_count = 0
        for bess_name, schedule in result.bess_schedules.items():
            bess_src = self.db_manager.get_or_create_source(bess_name, "bess")
            for timestamp, power in schedule:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                schedules.append((ts_str, bess_src, power_setpoint_met, str(power)))
                bess_count += 1
        
        # Save to database
        # IMPORTANT: Save ALL schedules, including zero values when boats are sailing
        if schedules:
            self.db_manager.save_records_batch("scheduling", schedules)
            print(f"     ✓ Saved {len(schedules)} schedule entries to database")
            print(f"       (Chargers: {charger_count}, BESS: {bess_count})")
