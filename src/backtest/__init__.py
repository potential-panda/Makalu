"""
Backtesting system for cross-exchange trading strategies.

This module provides an event-driven backtesting framework that:
- Processes market data from multiple exchanges in chronological order
- Calculates features in real-time
- Makes trading decisions using trained ML models
- Tracks PnL and generates performance reports
"""

from .data_loader import BacktestDataConfig, BacktestDataLoader
from .event_merger import Event, EventMerger, EventType
from .feature_calculator import FeatureCalculator, FeatureConfig
from .position_manager import Position, PositionConfig, PositionManager
from .result import SimulationResult, TradeRecord
from .simulator import Simulator, SimulatorConfig, SimulatorV2
from .volatility_tracker import VolatilityConfig, VolatilityTracker

__all__ = [
    "BacktestDataConfig",
    "BacktestDataLoader",
    "Event",
    "EventMerger",
    "EventType",
    "FeatureCalculator",
    "FeatureConfig",
    "Position",
    "PositionConfig",
    "PositionManager",
    "SimulationResult",
    "Simulator",
    "SimulatorConfig",
    "SimulatorV2",
    "TradeRecord",
    "VolatilityConfig",
    "VolatilityTracker",
]
