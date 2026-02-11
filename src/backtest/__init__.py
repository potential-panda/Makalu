"""
Backtesting system for cross-exchange trading strategies.

This module provides an event-driven backtesting framework that:
- Processes market data from multiple exchanges in chronological order
- Calculates features in real-time
- Makes trading decisions using trained ML models
- Tracks PnL and generates performance reports
"""

from .data_loader import BacktestDataConfig, BacktestDataLoader
from .event_merger import Event, EventType, EventMerger
from .feature_calculator import FeatureCalculator, FeatureConfig
from .volatility_tracker import VolatilityTracker, VolatilityConfig
from .position_manager import Position, PositionManager, PositionConfig
from .simulator import Simulator, SimulatorConfig, SimulatorV2
from .result import SimulationResult, TradeRecord

__all__ = [
    "BacktestDataConfig",
    "BacktestDataLoader",
    "Event",
    "EventType",
    "EventMerger",
    "FeatureCalculator",
    "FeatureConfig",
    "VolatilityTracker",
    "VolatilityConfig",
    "Position",
    "PositionManager",
    "PositionConfig",
    "Simulator",
    "SimulatorConfig",
    "SimulatorV2",
    "SimulationResult",
    "TradeRecord",
]
