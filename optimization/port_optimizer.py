"""Port energy optimization using SCIP."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pyscipopt import Model, quicksum
from models import Port, Boat, Charger, BESS, BoatState
from database import DatabaseManager
from forecasting import PortForecaster, EnergyForecast


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
        
        This is a SIMPLIFIED optimization model that focuses on:
        - Minimizing grid usage by using PV and BESS optimally
        - Ensuring boats have enough charge for trips
        
        Args:
            forecast_date: Date to optimize for
            energy_forecasts: Energy forecasts for the day
            trip_assignments: Trip assignments per boat
            
        Returns:
            OptimizationResult with optimal schedules
        """
        print(f"     🔧 Optimizing energy schedule (with tariff pricing)...")
        
        # Create optimization model
        model = Model("port_energy_optimization")
        model.hideOutput()  # Suppress solver output
        model.setRealParam('limits/time', 30.0)  # 30 second time limit
        
        # Number of timesteps (24 hours)
        T = len(energy_forecasts)
        timesteps = list(range(T))
        
        # ===================================================================
        # SIMPLIFIED DECISION VARIABLES
        # ===================================================================
        
        # BESS power at each timestep (simplified: just net power)
        # Positive = discharge, Negative = charge
        bess_power = {}
        bess_soc = {}
        
        for bess in self.port.bess_systems:
            bess_power[bess.name] = {}
            bess_soc[bess.name] = {}
            
            for t in timesteps:
                bess_power[bess.name][t] = model.addVar(
                    name=f"bess_power_{bess.name}_t{t}",
                    vtype="C",
                    lb=-bess.max_charge_power,  # Negative = charging
                    ub=bess.max_discharge_power  # Positive = discharging
                )
                bess_soc[bess.name][t] = model.addVar(
                    name=f"bess_soc_{bess.name}_t{t}",
                    vtype="C",
                    lb=bess.soc_min,
                    ub=bess.soc_max
                )
        
        # Charger power (simplified: just power, not assignment)
        charger_power = {}
        for charger in self.port.chargers:
            charger_power[charger.name] = {}
            for t in timesteps:
                charger_power[charger.name][t] = model.addVar(
                    name=f"charger_{charger.name}_t{t}",
                    vtype="C",  # Continuous
                    lb=0,
                    ub=charger.max_power
                )
        
        # 1. BESS SOC dynamics
        for bess in self.port.bess_systems:
            for t in timesteps:
                if t == 0:
                    # Initial BESS SOC
                    model.addCons(
                        bess_soc[bess.name][t] == bess.current_soc,
                        name=f"initial_bess_soc_{bess.name}"
                    )
                else:
                    # BESS SOC change based on power
                    # Negative power = charging (increases SOC)
                    # Positive power = discharging (decreases SOC)
                    power = bess_power[bess.name][t-1]
                    
                    # We need to handle charging and discharging differently due to efficiency
                    # For linear programming, we use separate charge and discharge variables
                    # Charge power (positive when charging)
                    charge_power = model.addVar(
                        name=f"bess_charge_{bess.name}_t{t-1}",
                        vtype="C",
                        lb=0,
                        ub=bess.max_charge_power
                    )
                    
                    # Discharge power (positive when discharging)
                    discharge_power = model.addVar(
                        name=f"bess_discharge_{bess.name}_t{t-1}",
                        vtype="C",
                        lb=0,
                        ub=bess.max_discharge_power
                    )
                    
                    # Constraint: net power = discharge - charge
                    model.addCons(
                        power == discharge_power - charge_power,
                        name=f"bess_power_split_{bess.name}_t{t-1}"
                    )
                    
                    # SOC change accounting for efficiency:
                    # When charging: energy stored = charge_power * time * efficiency
                    # When discharging: energy removed = discharge_power * time / efficiency
                    energy_stored = charge_power * self.timestep_hours * bess.efficiency
                    energy_removed = discharge_power * self.timestep_hours / bess.efficiency
                    
                    # SOC change (normalized by capacity)
                    soc_change = (energy_stored - energy_removed) / bess.capacity
                    
                    model.addCons(
                        bess_soc[bess.name][t] == bess_soc[bess.name][t-1] + soc_change,
                        name=f"bess_soc_dynamics_{bess.name}_t{t}"
                    )
        
        # Pre-compute boat availability at each timestep
        # boat_available[boat_name][t] = True if boat is available (not sailing) at timestep t
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
        
        # 2. CRITICAL: Ensure boats get charged before trips
        # This is THE PRIORITY - boats must be able to complete their trips
        # IMPORTANT: Only charge when boats are NOT sailing (available at port)
        boat_energy_shortfall = {}
        for boat in self.port.boats:
            trips = trip_assignments.get(boat.name, [])
            if trips:
                # Calculate TOTAL energy needed for all trips
                total_trip_energy = sum(trip.estimate_energy_required(boat.k) for trip in trips)
                
                # Energy boat currently has
                current_energy = boat.soc * boat.battery_capacity
                
                # Energy needed to charge to full + complete trips
                energy_needed = boat.battery_capacity - current_energy + total_trip_energy
                
                # Total energy provided by ALL chargers to this boat
                # ONLY when boat is NOT sailing (available at port)
                # Use pre-computed availability to filter timesteps
                available_timesteps = [t for t in timesteps if boat_available[boat.name][t]]
                total_charging_energy = quicksum(
                    charger_power[charger.name][t] * charger.efficiency * self.timestep_hours
                    for charger in self.port.chargers
                    for t in available_timesteps
                )
                
                # Create slack variable for energy shortfall
                boat_energy_shortfall[boat.name] = model.addVar(
                    name=f"shortfall_{boat.name}",
                    vtype="C",
                    lb=0
                )
                
                # Constraint: charging + shortfall >= needed
                # We want shortfall to be 0 (penalized heavily in objective)
                model.addCons(
                    total_charging_energy + boat_energy_shortfall[boat.name] >= energy_needed,
                    name=f"boat_{boat.name}_energy_requirement"
                )
        
        # 2b. Constraint: Chargers can only operate when boats are available (not sailing)
        # This prevents chargers from running when boats are out on trips
        for charger in self.port.chargers:
            for t in timesteps:
                if not any_boat_available[t]:
                    # All boats are sailing - charger must be off
                    model.addCons(
                        charger_power[charger.name][t] == 0,
                        name=f"charger_{charger.name}_off_when_all_sailing_t{t}"
                    )
        
        # 3. Power balance constraint
        # Note: When boats are sailing, they consume energy from their own batteries,
        # not from chargers, so charger power should be 0 (enforced above)
        for t in timesteps:
            forecast = energy_forecasts[t]
            
            # Total charger power needed (should be 0 when all boats are sailing)
            total_charger_power = quicksum(
                charger_power[charger.name][t]
                for charger in self.port.chargers
            )
            
            # Total BESS power (positive = discharge, negative = charge)
            total_bess_power = quicksum(
                bess_power[bess.name][t]
                for bess in self.port.bess_systems
            )
            
            # PV production (kW)
            pv_power_kw = forecast.pv_production_kwh / self.timestep_hours
            
            # Power balance: Grid + PV + BESS >= Chargers
            # Where BESS is positive when discharging (providing power)
            # When boats are sailing, charger_power should be 0, so this constraint
            # allows PV and BESS to be used for other purposes or stored
            model.addCons(
                self.port.contracted_power + pv_power_kw + total_bess_power >= total_charger_power,
                name=f"power_balance_t{t}"
            )
        
        # ===================================================================
        # OBJECTIVE FUNCTION: Prioritize boat charging, then minimize cost-weighted grid usage
        # ===================================================================
        
        # PRIMARY: Penalize energy shortfall for boats (MUST charge boats for trips!)
        boat_shortfall_penalty = quicksum(boat_energy_shortfall.values()) * 100000  # HUGE penalty
        
        # SECONDARY: Minimize cost-weighted grid energy usage
        # This encourages:
        # - Charging BESS during low-price periods (to store energy for later)
        # - Discharging BESS during high-price periods (to avoid buying expensive grid power)
        # - Using grid power during low-price periods
        total_grid_cost = 0
        for t in timesteps:
            forecast = energy_forecasts[t]
            timestamp = forecast_date + timedelta(seconds=t * self.timestep_seconds)
            
            # Get tariff price for this timestep
            price_per_kwh = self.port.get_tariff_price(timestamp)
            
            # Charger load (from grid)
            charger_load = quicksum(charger_power[charger.name][t] for charger in self.port.chargers)
            
            # BESS net power (positive = discharge, negative = charge)
            bess_net = quicksum(bess_power[bess.name][t] for bess in self.port.bess_systems)
            
            # PV production (kW)
            pv_power = forecast.pv_production_kwh / self.timestep_hours
            
            # Grid usage = load - PV - BESS_discharge
            # Positive = drawing from grid, Negative = excess (shouldn't happen due to constraints)
            grid_power = charger_load - pv_power - bess_net
            
            # Grid energy (kWh) for this timestep
            grid_energy = grid_power * self.timestep_hours
            
            # Cost-weighted grid energy
            # The optimizer will naturally:
            # - Charge BESS when prices are low (bess_net < 0 increases grid_power, but price is low)
            # - Discharge BESS when prices are high (bess_net > 0 decreases grid_power, avoiding high costs)
            # Since grid_energy can theoretically be negative (excess PV), we need to handle that
            # For linear programming, we create a variable for grid energy purchased (always >= 0)
            grid_energy_purchased = model.addVar(
                name=f"grid_energy_purchased_t{t}",
                vtype="C",
                lb=0
            )
            
            # Constraint: grid_energy_purchased >= grid_energy
            # Since we're minimizing cost, it will equal max(0, grid_energy)
            model.addCons(
                grid_energy_purchased >= grid_energy,
                name=f"grid_energy_purchased_constraint_t{t}"
            )
            
            # Cost for this timestep: grid energy purchased * price
            timestep_cost = grid_energy_purchased * price_per_kwh
            
            total_grid_cost += timestep_cost
        
        # Combined objective: HEAVILY prioritize boat charging, then minimize costs
        model.setObjective(
            boat_shortfall_penalty + total_grid_cost,
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
                
                print(f"     ✓ Optimization complete: {status}")
                print(f"       Objective value: {obj_value:.2f}")
                
                # Warn about energy shortfalls
                if energy_shortfalls:
                    print(f"     ⚠️  Energy shortfalls detected:")
                    for boat_name, shortfall_kwh in energy_shortfalls.items():
                        boat = next(b for b in self.port.boats if b.name == boat_name)
                        shortfall_pct = (shortfall_kwh / boat.battery_capacity) * 100
                        print(f"       {boat_name}: {shortfall_kwh:.2f} kWh ({shortfall_pct:.1f}% of battery)")
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

