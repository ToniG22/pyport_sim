"""Database manager for storing simulation data."""

import sqlite3
from typing import Optional
from contextlib import contextmanager


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

            # Measurements table - for actual measured/simulated data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL
                )
            """
            )

            # Create index for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
                ON measurements(timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_source 
                ON measurements(source)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurements_metric 
                ON measurements(metric)
            """
            )

            # Forecast table - for predicted/forecasted data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL
                )
            """
            )

            # Create index for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_timestamp 
                ON forecast(timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_source 
                ON forecast(source)
            """
            )

            # Scheduling table - for scheduled events/commands
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduling (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL
                )
            """
            )

            # Create index for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduling_timestamp 
                ON scheduling(timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduling_source 
                ON scheduling(source)
            """
            )

            conn.commit()

    def save_measurement(self, timestamp: str, source: str, metric: str, value: float):
        """
        Save a single measurement to the database.

        Args:
            timestamp: Simulation timestamp (ISO format UTC string)
            source: Source identifier (e.g., 'boat_001', 'charger_01', 'port')
            metric: Metric name (e.g., 'power', 'soc', 'status')
            value: Measured value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO measurements (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp, source, metric, value),
            )

    def save_measurements_batch(self, measurements: list):
        """
        Save multiple measurements in a single transaction.

        Args:
            measurements: List of tuples (timestamp, source, metric, value)
                         where timestamp is ISO format UTC string
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO measurements (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                measurements,
            )

    def get_measurements(
        self,
        source: str = None,
        metric: str = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Retrieve measurements with optional filtering.

        Args:
            source: Filter by source (optional)
            metric: Filter by metric (optional)
            start_time: Filter by start timestamp ISO string (optional)
            end_time: Filter by end timestamp ISO string (optional)

        Returns:
            List of measurement records
        """
        query = "SELECT * FROM measurements WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if metric:
            query += " AND metric = ?"
            params.append(metric)

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

    def save_forecast(self, timestamp: str, source: str, metric: str, value: float):
        """
        Save a single forecast to the database.

        Args:
            timestamp: Forecast timestamp (ISO format UTC string)
            source: Source identifier (e.g., 'weather', 'pv_forecast')
            metric: Metric name (e.g., 'temperature', 'ghi', 'dni')
            value: Forecast value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO forecast (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp, source, metric, value),
            )

    def save_forecasts_batch(self, forecasts: list):
        """
        Save multiple forecasts in a single transaction.

        Args:
            forecasts: List of tuples (timestamp, source, metric, value)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO forecast (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                forecasts,
            )

    def get_forecasts(
        self,
        source: str = None,
        metric: str = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Retrieve forecasts with optional filtering.

        Args:
            source: Filter by source (optional)
            metric: Filter by metric (optional)
            start_time: Filter by start timestamp ISO string (optional)
            end_time: Filter by end timestamp ISO string (optional)

        Returns:
            List of forecast records
        """
        query = "SELECT * FROM forecast WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if metric:
            query += " AND metric = ?"
            params.append(metric)

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
    
    def save_schedule(self, timestamp: str, source: str, metric: str, value: float):
        """
        Save a single schedule entry to the database.
        
        Args:
            timestamp: Schedule timestamp (ISO format UTC string)
            source: Source identifier (e.g., 'FastCharger_A', 'Battery_Storage_1')
            metric: Metric/command name (e.g., 'power_setpoint', 'charge_command')
            value: Scheduled value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scheduling (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp, source, metric, value),
            )
    
    def save_schedules_batch(self, schedules: list):
        """
        Save multiple schedules in a single transaction.
        
        Args:
            schedules: List of tuples (timestamp, source, metric, value)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO scheduling (timestamp, source, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                schedules,
            )
    
    def get_schedules(
        self,
        source: str = None,
        metric: str = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Retrieve schedules from the database with optional filtering.
        
        Args:
            source: Filter by source (e.g., 'FastCharger_A')
            metric: Filter by metric (e.g., 'power_setpoint')
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            
        Returns:
            List of schedule records
        """
        query = "SELECT * FROM scheduling WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if metric:
            query += " AND metric = ?"
            params.append(metric)
        
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
    
    def clear_schedules(self, source: str = None):
        """
        Clear scheduling entries from the database.
        
        Args:
            source: Optional source filter (clears only this source if provided)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source:
                cursor.execute("DELETE FROM scheduling WHERE source = ?", (source,))
            else:
                cursor.execute("DELETE FROM scheduling")
