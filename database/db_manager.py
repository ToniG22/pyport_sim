"""Database manager for storing simulation data."""

import sqlite3
from typing import Optional, List, Tuple
from contextlib import contextmanager

# Valid table names for data records
VALID_TABLES = ("measurements", "forecast", "scheduling")

# Default metrics to initialize with the database
# Format: (metric_name, unit, data_type)
DEFAULT_METRICS = (
    # Power metrics
    ("power_active_consumption", "kW", "float"),
    ("power_active_production", "kW", "float"),
    ("power_active_import", "kW", "float"),
    ("power_active_export", "kW", "float"),
    ("available_power", "kW", "float"),
    ("contracted_power", "kW", "float"),
    ("power_active", "kW", "float"),
    ("power_setpoint", "kW", "float"),
    # Battery/BESS metrics
    ("soc", "%", "float"),
    ("state", "", "str"),
    ("bess_discharge", "kW", "float"),
    ("bess_charge", "kW", "float"),
    ("energy_stored", "kWh", "float"),
    ("bess_available", "kWh", "float"),
    ("bess_capacity", "kWh", "float"),
    # Weather metrics
    ("temperature", "°C", "float"),
    ("humidity", "%", "float"),
    ("dew_point", "°C", "float"),
    ("precipitation", "mm", "float"),
    ("weather_code", "", "int"),
    ("cloud_cover", "%", "float"),
    ("wind_speed", "m/s", "float"),
    ("wind_direction", "°", "float"),
    # Solar radiation metrics
    ("ghi", "W/m²", "float"),
    ("direct_radiation", "W/m²", "float"),
    ("dhi", "W/m²", "float"),
    ("dni", "W/m²", "float"),
    # Energy/production metrics
    ("net_balance", "kW", "float"),
)


class DatabaseManager:
    """Manages SQLite database operations for the simulator."""

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        # Cache for source and metric IDs to avoid repeated lookups
        self._source_cache: dict[str, int] = {}
        self._metric_cache: dict[str, int] = {}

    def connect(self):
        """Establish database connection."""
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def initialize_schema(self):
        """Create database tables for simulation data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Source table - registry of data sources
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS source (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL
                )
            """
            )

            # Metric table - registry of metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metric (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL UNIQUE,
                    unit TEXT NOT NULL,
                    data_type TEXT NOT NULL
                )
            """
            )

            # Measurements table - for actual measured/simulated data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    metric_id INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES source(source_id),
                    FOREIGN KEY (metric_id) REFERENCES metric(metric_id)
                )
            """
            )

            # Create indexes for measurements
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
                ON measurements(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_source_id 
                ON measurements(source_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_metric_id 
                ON measurements(metric_id)
            """
            )

            # Forecast table - for predicted/forecasted data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast (
                    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    metric_id INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES source(source_id),
                    FOREIGN KEY (metric_id) REFERENCES metric(metric_id)
                )
            """
            )

            # Create indexes for forecast
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_timestamp 
                ON forecast(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_source_id 
                ON forecast(source_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_metric_id 
                ON forecast(metric_id)
            """
            )

            # Scheduling table - for scheduled events/commands
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduling (
                    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    metric_id INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES source(source_id),
                    FOREIGN KEY (metric_id) REFERENCES metric(metric_id)
                )
            """
            )

            # Create indexes for scheduling
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduling_timestamp 
                ON scheduling(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduling_source_id 
                ON scheduling(source_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduling_metric_id 
                ON scheduling(metric_id)
            """
            )

            conn.commit()

    def initialize_default_metrics(self):
        """
        Initialize the database with default metrics.
        
        Inserts all metrics from DEFAULT_METRICS if they don't already exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for metric_name, unit, data_type in DEFAULT_METRICS:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO metric (metric_name, unit, data_type)
                    VALUES (?, ?, ?)
                """,
                    (metric_name, unit, data_type),
                )
            conn.commit()

    def _validate_table(self, table: str):
        """Validate that the table name is allowed."""
        if table not in VALID_TABLES:
            raise ValueError(f"Invalid table '{table}'. Must be one of: {VALID_TABLES}")

    # Unified data record methods (measurements, forecast, scheduling)
    def save_record(
        self, table: str, timestamp: str, source_id: int, metric_id: int, value: str
    ) -> int:
        """
        Save a single record to the specified table.

        Args:
            table: Table name ('measurements', 'forecast', or 'scheduling')
            timestamp: Timestamp (ISO format UTC string)
            source_id: Foreign key to source table
            metric_id: Foreign key to metric table
            value: Value as string

        Returns:
            The measurement_id of the inserted record
        """
        self._validate_table(table)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {table} (timestamp, source_id, metric_id, value)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp, source_id, metric_id, value),
            )
            return cursor.lastrowid

    def save_records_batch(
        self, table: str, records: List[Tuple[str, int, int, str]]
    ):
        """
        Save multiple records in a single transaction.

        Args:
            table: Table name ('measurements', 'forecast', or 'scheduling')
            records: List of tuples (timestamp, source_id, metric_id, value)
        """
        self._validate_table(table)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT INTO {table} (timestamp, source_id, metric_id, value)
                VALUES (?, ?, ?, ?)
            """,
                records,
            )

    def get_records(
        self,
        table: str,
        source_id: int = None,
        metric_id: int = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Retrieve records from the specified table with optional filtering.

        Args:
            table: Table name ('measurements', 'forecast', or 'scheduling')
            source_id: Filter by source_id (optional)
            metric_id: Filter by metric_id (optional)
            start_time: Filter by start timestamp ISO string (optional)
            end_time: Filter by end timestamp ISO string (optional)

        Returns:
            List of records
        """
        self._validate_table(table)
        query = f"SELECT * FROM {table} WHERE 1=1"
        params = []

        if source_id is not None:
            query += " AND source_id = ?"
            params.append(source_id)

        if metric_id is not None:
            query += " AND metric_id = ?"
            params.append(metric_id)

        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def clear_records(
        self, table: str, source_id: int = None, from_time: str = None
    ):
        """
        Clear records from the specified table.

        Args:
            table: Table name ('measurements', 'forecast', or 'scheduling')
            source_id: Optional source_id filter (clears only this source if provided)
            from_time: Optional timestamp filter (clears only records from this time onwards)
        """
        self._validate_table(table)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_id is not None and from_time is not None:
                cursor.execute(
                    f"DELETE FROM {table} WHERE source_id = ? AND timestamp >= ?",
                    (source_id, from_time),
                )
            elif source_id is not None:
                cursor.execute(
                    f"DELETE FROM {table} WHERE source_id = ?", (source_id,)
                )
            elif from_time is not None:
                cursor.execute(
                    f"DELETE FROM {table} WHERE timestamp >= ?", (from_time,)
                )
            else:
                cursor.execute(f"DELETE FROM {table}")

    # Source table methods
    def save_source(self, source_name: str, source_type: str) -> int:
        """
        Save a source to the database.

        Args:
            source_name: Name of the source (e.g., 'boat_001', 'charger_01')
            source_type: Type of source (e.g., 'boat', 'charger', 'battery')

        Returns:
            The source_id of the inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO source (source_name, source_type)
                VALUES (?, ?)
            """,
                (source_name, source_type),
            )
            return cursor.lastrowid

    def get_source(self, source_id: int = None, source_name: str = None):
        """
        Retrieve a source by ID or name.

        Args:
            source_id: Source ID (optional)
            source_name: Source name (optional)

        Returns:
            Source record or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_id is not None:
                cursor.execute("SELECT * FROM source WHERE source_id = ?", (source_id,))
            elif source_name is not None:
                cursor.execute("SELECT * FROM source WHERE source_name = ?", (source_name,))
            else:
                return None
            return cursor.fetchone()

    def get_all_sources(self, source_type: str = None):
        """
        Retrieve all sources, optionally filtered by type.

        Args:
            source_type: Filter by source type (optional)

        Returns:
            List of source records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_type:
                cursor.execute("SELECT * FROM source WHERE source_type = ?", (source_type,))
            else:
                cursor.execute("SELECT * FROM source")
            return cursor.fetchall()

    def delete_source(self, source_id: int):
        """
        Delete a source by ID.

        Args:
            source_id: Source ID to delete
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM source WHERE source_id = ?", (source_id,))

    # Metric table methods
    def save_metric(self, metric_name: str, unit: str, data_type: str) -> int:
        """
        Save a metric to the database.

        Args:
            metric_name: Name of the metric (e.g., 'power', 'soc', 'temperature')
            unit: Unit of measurement (e.g., 'kW', '%', 'C')
            data_type: Data type (e.g., 'float', 'int', 'bool')

        Returns:
            The metric_id of the inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metric (metric_name, unit, data_type)
                VALUES (?, ?, ?)
            """,
                (metric_name, unit, data_type),
            )
            return cursor.lastrowid

    def get_metric(self, metric_id: int = None, metric_name: str = None):
        """
        Retrieve a metric by ID or name.

        Args:
            metric_id: Metric ID (optional)
            metric_name: Metric name (optional)

        Returns:
            Metric record or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if metric_id is not None:
                cursor.execute("SELECT * FROM metric WHERE metric_id = ?", (metric_id,))
            elif metric_name is not None:
                cursor.execute("SELECT * FROM metric WHERE metric_name = ?", (metric_name,))
            else:
                return None
            return cursor.fetchone()

    def get_all_metrics(self, data_type: str = None):
        """
        Retrieve all metrics, optionally filtered by data type.

        Args:
            data_type: Filter by data type (optional)

        Returns:
            List of metric records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if data_type:
                cursor.execute("SELECT * FROM metric WHERE data_type = ?", (data_type,))
            else:
                cursor.execute("SELECT * FROM metric")
            return cursor.fetchall()

    def delete_metric(self, metric_id: int):
        """
        Delete a metric by ID.

        Args:
            metric_id: Metric ID to delete
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metric WHERE metric_id = ?", (metric_id,))

    # Helper methods for ID resolution
    def get_or_create_source(self, source_name: str, source_type: str) -> int:
        """
        Get source_id for a source, creating it if it doesn't exist.
        
        Uses caching to avoid repeated database lookups.

        Args:
            source_name: Name of the source (e.g., 'port', 'SeaBreeze', 'FastCharger_A')
            source_type: Type of source (e.g., 'port', 'boat', 'charger', 'pv', 'bess', 'weather')

        Returns:
            The source_id
        """
        # Check cache first
        if source_name in self._source_cache:
            return self._source_cache[source_name]
        
        # Try to get existing source
        source = self.get_source(source_name=source_name)
        if source:
            source_id = source["source_id"]
            self._source_cache[source_name] = source_id
            return source_id
        
        # Create new source
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO source (source_name, source_type)
                VALUES (?, ?)
            """,
                (source_name, source_type),
            )
            source_id = cursor.lastrowid
        
        self._source_cache[source_name] = source_id
        return source_id

    def get_metric_id(self, metric_name: str) -> int:
        """
        Get metric_id for a metric by name.
        
        Uses caching to avoid repeated database lookups.
        Assumes metric already exists (from DEFAULT_METRICS or manual creation).

        Args:
            metric_name: Name of the metric (e.g., 'power_active', 'soc')

        Returns:
            The metric_id

        Raises:
            ValueError: If metric doesn't exist
        """
        # Check cache first
        if metric_name in self._metric_cache:
            return self._metric_cache[metric_name]
        
        # Get existing metric
        metric = self.get_metric(metric_name=metric_name)
        if metric:
            metric_id = metric["metric_id"]
            self._metric_cache[metric_name] = metric_id
            return metric_id
        
        raise ValueError(f"Metric '{metric_name}' not found. Ensure it's in DEFAULT_METRICS or created manually.")

    def clear_caches(self):
        """Clear the source and metric ID caches."""
        self._source_cache.clear()
        self._metric_cache.clear()
