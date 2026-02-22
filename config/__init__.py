"""
Configuration for the port simulator.

Exposes Settings (global simulator options) and SimulationMode (real-time vs batch).
"""

from .settings import Settings, SimulationMode

__all__ = ["Settings", "SimulationMode"]
