# Electric Port Simulator

A comprehensive simulation system for managing electric recreational ports with boats, charging infrastructure, solar PV systems, and battery energy storage systems (BESS). The simulator optimizes energy usage, manages boat charging schedules, and handles trip assignments while respecting grid power constraints.

## Features

- **Port Management**: Simulate electric ports with configurable power contracts and location
- **Boat Fleet**: Model multiple electric boats with different specifications (motor power, battery capacity, weight, etc.)
- **Charging Infrastructure**: Multiple charging stations with configurable power and efficiency
- **Solar PV System**: Realistic PV production modeling using weather data from Open-Meteo API
- **Battery Storage (BESS)**: Energy storage system with configurable capacity and control strategies
- **Trip Management**: Load and assign boat trips from CSV route files
- **Energy Forecasting**: Predict energy consumption and production for optimal scheduling
- **Optimization**: SCIP-based optimization engine to minimize grid usage and trip delays
- **Weather Integration**: Real-time weather data fetching for accurate PV production forecasts
- **Database Storage**: SQLite database for storing simulation results and forecasts
- **Visualization**: Streamlit web app for viewing simulation results and analyzing data

## Installation

### Prerequisites

- Python 3.11 or higher
- SCIP Optimization Suite (required for optimization features)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pyport_sim
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install SCIP (for optimization features):
   - **Linux**: Download from [SCIP website](https://scipopt.org/index.php/download) or use package manager
   - **macOS**: `brew install scip`
   - **Windows**: Download installer from SCIP website

## Quick Start

1. Run a simulation:
```bash
python main.py
```

2. View results in the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
pyport_sim/
├── main.py                 # Main entry point for simulations
├── streamlit_app.py        # Web dashboard for viewing results
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.py        # Simulation settings and configuration
├── models/
│   ├── port.py            # Port model
│   ├── boat.py            # Boat model
│   ├── charger.py         # Charger model
│   ├── pv.py              # PV system model
│   ├── bess.py            # Battery storage model
│   └── trip.py            # Trip/route model
├── simulation/
│   ├── engine.py          # Core simulation engine
│   └── trip_manager.py    # Trip assignment and management
├── optimization/
│   └── port_optimizer.py  # SCIP-based energy optimization
├── forecasting/
│   └── port_forecaster.py # Energy consumption/production forecasting
├── database/
│   └── db_manager.py      # SQLite database operations
├── weather/
│   └── openmeteo.py       # Weather API client
└── assets/
    ├── trips/             # CSV route files for boat trips
    └── tariff/            # Electricity tariff configuration
```

## Configuration

### Port Setup

Configure your port in `main.py`:

```python
port = Port(
    name="Marina del Sol",
    contracted_power=40,  # kW
    lat=32.64542,
    lon=-16.90841,
)
```

### Boats

Add boats with their specifications:

```python
boat = Boat(
    name="SeaBreeze",
    motor_power=120,        # kW
    weight=2500,            # kg
    length=8.5,             # m
    battery_capacity=150,   # kWh
    range_speed=15.0,       # knots
    soc=0.30,              # Initial state of charge (0-1)
)
```

### Chargers

Configure charging stations:

```python
charger = Charger(
    name="FastCharger_A",
    max_power=22,          # kW
    efficiency=0.95,       # 95% efficiency
)
```

### PV System

Set up solar panels:

```python
pv_system = PV(
    name="Solar_Array_1",
    capacity=30.0,          # kW DC
    tilt=30.0,             # degrees
    azimuth=180.0,         # degrees (180 = South-facing)
    efficiency=0.85,       # System efficiency
    latitude=port.lat,
    longitude=port.lon,
)
```

### BESS (Battery Storage)

Configure battery storage:

```python
bess = BESS(
    name="Battery_Storage_1",
    capacity=100.0,         # kWh
    max_charge_power=25.0,  # kW
    max_discharge_power=25.0,  # kW
    efficiency=0.90,       # Round-trip efficiency
    soc_min=0.10,          # Minimum SOC (10%)
    soc_max=0.90,          # Maximum SOC (90%)
    initial_soc=0.50,      # Initial SOC (50%)
    control_strategy=BESSControlStrategy.DEFAULT,
)
```

### Simulation Settings

Configure simulation parameters:

```python
settings = Settings(
    timestep=900,                    # 15 minutes in seconds
    mode=SimulationMode.BATCH,      # BATCH or REAL_TIME
    db_path="port_simulation.db",    # Database file path
    use_optimizer=True,              # Enable SCIP optimization
)
```

## Usage

### Running a Simulation

1. Edit `main.py` to configure your port, boats, chargers, PV, and BESS
2. Run the simulation:
```bash
python main.py
```

The simulation will:
- Load trip routes from `assets/trips/`
- Fetch weather forecasts from Open-Meteo API
- Generate energy forecasts
- Optimize charging schedules (if optimizer enabled)
- Simulate boat trips and charging
- Save all data to SQLite database

### Viewing Results

Launch the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard allows you to:
- Select database files
- Filter by sources (boats, chargers, PV, BESS, port)
- Filter by metrics (power, SOC, status, etc.)
- View time-series plots
- Download filtered data as CSV

## Key Components

### Simulation Engine

The `SimulationEngine` manages the entire simulation:
- Time-stepped simulation (default: 15-minute intervals)
- Boat trip assignment and execution
- Charging management
- PV production calculation
- BESS charge/discharge control
- Grid power monitoring

### Optimization

The `PortOptimizer` uses SCIP to optimize:
- Charger power schedules
- BESS charge/discharge schedules
- Minimize grid usage
- Minimize trip delays
- Consider electricity tariffs

### Forecasting

The `PortForecaster` predicts:
- Energy consumption (boats on trips + charging)
- PV production (based on weather forecasts)
- BESS availability
- Net energy balance

### Trip Management

Trips are loaded from CSV files in `assets/trips/` with format:
- `timestamp`: DateTime of waypoint
- `type`: Point type (Static, Dock, Terrestrial, Interpolated)
- `speed`: Speed in knots
- `heading`: Heading in degrees
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate

## Database Schema

The simulator uses SQLite with three main tables:

- **measurements**: Actual simulated data (power, SOC, status, etc.)
- **forecast**: Predicted energy consumption and production
- **scheduling**: Optimized schedules for chargers and BESS

## Dependencies

Key dependencies:
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `pyscipopt`: SCIP optimization solver
- `streamlit`: Web dashboard
- `plotly`: Interactive visualizations
- `requests`: HTTP requests for weather API

See `requirements.txt` for complete list.

## Notes

- Weather data is fetched from Open-Meteo API (free tier, 7-day forecast limit)
- Optimization requires SCIP solver installation
- Simulation supports up to 7 days per run
- Trip routes must be in CSV format with specific columns
- Database file is created automatically on first run
