"""Database manager for storing and querying simulation data (SQLite)."""

import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Tuple

VALID_TABLES = ("measurements", "forecast", "scheduling")

DEFAULT_METRICS = (
    ("power_active_consumption", "kW", "float"),
    ("power_active_production", "kW", "float"),
    ("power_active_import", "kW", "float"),
    ("power_active_export", "kW", "float"),
    ("available_power", "kW", "float"),
    ("contracted_power", "kW", "float"),
    ("power_active", "kW", "float"),
    ("power_setpoint", "kW", "float"),
    ("soc", "%", "float"),
    ("state", "", "str"),
    ("bess_discharge", "kW", "float"),
    ("bess_charge", "kW", "float"),
    ("energy_stored", "kWh", "float"),
    ("bess_available", "kWh", "float"),
    ("bess_capacity", "kWh", "float"),
    ("boat_required_energy", "kWh", "float"),
    ("boat_available", "", "int"),
    ("temperature", "°C", "float"),
    ("humidity", "%", "float"),
    ("dew_point", "°C", "float"),
    ("precipitation", "mm", "float"),
    ("weather_code", "", "int"),
    ("cloud_cover", "%", "float"),
    ("wind_speed", "m/s", "float"),
    ("wind_direction", "°", "float"),
    ("ghi", "W/m²", "float"),
    ("direct_radiation", "W/m²", "float"),
    ("dhi", "W/m²", "float"),
    ("dni", "W/m²", "float"),
    ("net_balance", "kW", "float"),
)


class DatabaseManager:
    """
    Manages SQLite database operations for the simulator.

    Uses in-memory caches for source_id and metric_id to avoid repeated lookups.
    """

    def __init__(self, db_path: str):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._source_cache: dict[str, int] = {}
        self._metric_cache: dict[str, int] = {}

    def connect(self):
        """Open a persistent database connection and set row_factory to sqlite3.Row."""
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row

    def close(self):
        """Close the persistent database connection if open."""
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextmanager
    def get_connection(self):
        """Context manager: yields a connection with foreign keys on; commits on success, rolls back on exception."""
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
        """Create all tables (source, metric, measurements, forecast, scheduling) and their indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS source (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL
                )
            """
            )

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
        """Insert all DEFAULT_METRICS into the metric table (INSERT OR IGNORE)."""
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
        """Raise ValueError if table is not one of VALID_TABLES."""
        if table not in VALID_TABLES:
            raise ValueError(f"Invalid table '{table}'. Must be one of: {VALID_TABLES}")

    def save_record(
        self, table: str, timestamp: str, source_id: int, metric_id: int, value: str
    ) -> int:
        """
        Insert a single record into the given table.

        Args:
            table: One of 'measurements', 'forecast', 'scheduling'.
            timestamp: ISO-format UTC timestamp string.
            source_id: Foreign key to source.
            metric_id: Foreign key to metric.
            value: Value stored as string.

        Returns:
            The measurement_id of the inserted row.
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

    def save_records_batch(self, table: str, records: List[Tuple[str, int, int, str]]):
        """
        Insert multiple records in one transaction.

        Args:
            table: One of 'measurements', 'forecast', 'scheduling'.
            records: List of (timestamp, source_id, metric_id, value) tuples.
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
        Return records from the table with optional filters.

        Args:
            table: One of 'measurements', 'forecast', 'scheduling'.
            source_id: Optional filter by source_id.
            metric_id: Optional filter by metric_id.
            start_time: Optional inclusive start timestamp (ISO string).
            end_time: Optional inclusive end timestamp (ISO string).

        Returns:
            List of rows (sqlite3.Row), ordered by timestamp.
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

    def clear_records(self, table: str, source_id: int = None, from_time: str = None):
        """
        Delete records from the table, optionally by source_id and/or from_time.

        Args:
            table: One of 'measurements', 'forecast', 'scheduling'.
            source_id: If set, delete only records for this source.
            from_time: If set, delete only records with timestamp >= from_time (ISO string).
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
                cursor.execute(f"DELETE FROM {table} WHERE source_id = ?", (source_id,))
            elif from_time is not None:
                cursor.execute(
                    f"DELETE FROM {table} WHERE timestamp >= ?", (from_time,)
                )
            else:
                cursor.execute(f"DELETE FROM {table}")

    def save_source(self, source_name: str, source_type: str) -> int:
        """
        Insert a source row.

        Args:
            source_name: Unique name (e.g. 'boat_001', 'charger_01').
            source_type: Type (e.g. 'boat', 'charger', 'battery').

        Returns:
            The inserted source_id.
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
        Return one source by source_id or source_name; at least one must be given.

        Returns:
            sqlite3.Row or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_id is not None:
                cursor.execute("SELECT * FROM source WHERE source_id = ?", (source_id,))
            elif source_name is not None:
                cursor.execute(
                    "SELECT * FROM source WHERE source_name = ?", (source_name,)
                )
            else:
                return None
            return cursor.fetchone()

    def get_all_sources(self, source_type: str = None):
        """
        Return all source rows, optionally filtered by source_type.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_type:
                cursor.execute(
                    "SELECT * FROM source WHERE source_type = ?", (source_type,)
                )
            else:
                cursor.execute("SELECT * FROM source")
            return cursor.fetchall()

    def delete_source(self, source_id: int):
        """Delete the source row with the given source_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM source WHERE source_id = ?", (source_id,))

    def save_metric(self, metric_name: str, unit: str, data_type: str) -> int:
        """
        Insert a metric row.

        Args:
            metric_name: Unique name (e.g. 'power', 'soc', 'temperature').
            unit: Unit string (e.g. 'kW', '%', '°C').
            data_type: One of 'float', 'int', 'str', etc.

        Returns:
            The inserted metric_id.
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
        Return one metric by metric_id or metric_name; at least one must be given.

        Returns:
            sqlite3.Row or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if metric_id is not None:
                cursor.execute("SELECT * FROM metric WHERE metric_id = ?", (metric_id,))
            elif metric_name is not None:
                cursor.execute(
                    "SELECT * FROM metric WHERE metric_name = ?", (metric_name,)
                )
            else:
                return None
            return cursor.fetchone()

    def get_all_metrics(self, data_type: str = None):
        """Return all metric rows, optionally filtered by data_type."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if data_type:
                cursor.execute("SELECT * FROM metric WHERE data_type = ?", (data_type,))
            else:
                cursor.execute("SELECT * FROM metric")
            return cursor.fetchall()

    def delete_metric(self, metric_id: int):
        """Delete the metric row with the given metric_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metric WHERE metric_id = ?", (metric_id,))

    def get_or_create_source(self, source_name: str, source_type: str) -> int:
        """
        Return source_id for the given source; insert the source if it does not exist.
        Results are cached to avoid repeated lookups.

        Args:
            source_name: Unique source name (e.g. 'port', 'SeaBreeze', 'FastCharger_A').
            source_type: Type (e.g. 'port', 'boat', 'charger', 'pv', 'bess', 'weather').

        Returns:
            The source_id.
        """
        if source_name in self._source_cache:
            return self._source_cache[source_name]

        source = self.get_source(source_name=source_name)
        if source:
            source_id = source["source_id"]
            self._source_cache[source_name] = source_id
            return source_id

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
        Return metric_id for the given metric name. Uses caching.
        The metric must already exist (e.g. from DEFAULT_METRICS or manual creation).

        Args:
            metric_name: Metric name (e.g. 'power_active', 'soc').

        Returns:
            The metric_id.

        Raises:
            ValueError: If no metric with that name exists.
        """
        if metric_name in self._metric_cache:
            return self._metric_cache[metric_name]

        metric = self.get_metric(metric_name=metric_name)
        if metric:
            metric_id = metric["metric_id"]
            self._metric_cache[metric_name] = metric_id
            return metric_id

        raise ValueError(
            f"Metric '{metric_name}' not found. Ensure it's in DEFAULT_METRICS or created manually."
        )

    def clear_caches(self):
        """Clear in-memory source_id and metric_id caches."""
        self._source_cache.clear()
        self._metric_cache.clear()
