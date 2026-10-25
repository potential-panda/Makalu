"""
Main simulator for event-driven backtesting.

Processes events from multiple exchanges in chronological order,
calculates features, makes trading decisions, and tracks PnL.
"""

import logging
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from qate.core.model import OrderBook, Trade

from .event_merger import Event, EventType
from .feature_calculator import FeatureCalculator, FeatureConfig
from .position_manager import Position, PositionConfig, PositionManager
from .result import SimulationResult, TradeRecord
from .volatility_tracker import VolatilityConfig, VolatilityTracker

LOG = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for the simulator."""

    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    volatility_config: VolatilityConfig = field(default_factory=VolatilityConfig)
    position_config: PositionConfig = field(default_factory=PositionConfig)
    model_path: str = ""
    feature_columns: list[str] = field(default_factory=list)

    # Prediction mapping: model output -> direction
    # Default: 0=Short, 1=Long, 2=Timeout (no trade)
    long_prediction: int = 1
    short_prediction: int = 0


class Simulator:
    """
    Main simulation class for event-driven backtesting.

    Processes events from multiple exchanges:
    - BINANCE events: Updates feature calculator
    - GMO events: Updates volatility tracker, checks entry/exit

    Trading logic:
    - Only enter trades when volatility > spread * min_vol_spread_ratio
    - Use ML model to predict direction (Long/Short/Timeout)
    - Exit based on triple barrier (profit-take, stop-loss, timeout)
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config

        # Components
        self._feature_calc = FeatureCalculator(config.feature_config)
        self._vol_tracker = VolatilityTracker(config.volatility_config)
        self._position_mgr = PositionManager(config.position_config)

        # Load model
        self._model = None
        if config.model_path:
            LOG.info(f"Loading model from {config.model_path}")
            with open(config.model_path, "rb") as f:
                self._model = pickle.load(f)

        # State
        self._result = SimulationResult()
        self._current_position: Position | None = None
        self._cumulative_pnl = 0.0
        self._event_count = 0
        self._last_features: dict | None = None

    def run(self, events: Iterable[Event]) -> SimulationResult:
        """
        Run the simulation on a stream of events.

        Args:
            events: Iterable of Event objects in chronological order

        Returns:
            SimulationResult with trade records and PnL history
        """
        LOG.info("Starting simulation")

        for event in events:
            self._process_event(event)
            self._event_count += 1

            if self._event_count % 100000 == 0:
                LOG.info(
                    f"Processed {self._event_count} events, {len(self._result.trades)} trades"
                )

        # Force exit any remaining position
        if self._position_mgr.has_position:
            last_ob = self._vol_tracker.get_last_orderbook()
            if last_ob is not None:
                pnl, reason = self._position_mgr.force_exit(
                    last_ob.get_ts(), last_ob.best_bid, last_ob.best_ask
                )
                self._record_trade(last_ob.get_ts(), pnl, reason)

        LOG.info(
            f"Simulation complete: {self._event_count} events, {len(self._result.trades)} trades"
        )
        return self._result

    def _process_event(self, event: Event) -> None:
        """Process a single event."""
        if event.event_type == EventType.BINANCE_ORDERBOOK:
            self._on_binance_orderbook(event.data)
        elif event.event_type == EventType.BINANCE_TRADE:
            self._on_binance_trade(event.data)
        elif event.event_type == EventType.GMO_ORDERBOOK:
            self._on_gmo_orderbook(event.data)

    def _on_binance_orderbook(self, orderbook: OrderBook) -> None:
        """Process BINANCE order book update."""
        self._feature_calc.on_orderbook(orderbook)
        # Cache features for when GMO event arrives
        self._last_features = self._feature_calc.get_features()

    def _on_binance_trade(self, trade: Trade) -> None:
        """Process BINANCE trade."""
        self._feature_calc.on_trade(trade)

    def _on_gmo_orderbook(self, orderbook: OrderBook) -> None:
        """Process GMO order book update."""
        ts = orderbook.get_ts()
        bid = orderbook.best_bid
        ask = orderbook.best_ask

        self._vol_tracker.on_orderbook(orderbook)

        # Check exit for existing position
        if self._position_mgr.has_position:
            exited, pnl, reason = self._position_mgr.check_exit(ts, bid, ask)
            if exited:
                self._record_trade(ts, pnl, reason)

        # Check entry for new position
        if not self._position_mgr.has_position:
            self._maybe_enter_trade(ts, orderbook)

    def _maybe_enter_trade(self, ts: float, gmo_orderbook: OrderBook) -> None:
        """Check if we should enter a new trade."""
        # Check volatility/spread condition
        if not self._vol_tracker.is_tradeable():
            return

        # Check if we have features
        if self._last_features is None:
            return

        # Check if we have a model
        if self._model is None:
            return

        # Get volatility for barrier calculation
        volatility_ema = self._vol_tracker.get_volatility_ema()
        if volatility_ema is None:
            return

        # Prepare features for prediction
        try:
            feature_vector = self._prepare_features()
            if feature_vector is None:
                return
        except Exception as e:
            LOG.warning(f"Error preparing features: {e}")
            return

        # Make prediction
        try:
            prediction = self._model.predict([feature_vector])[0]
        except Exception as e:
            LOG.warning(f"Error making prediction: {e}")
            return

        # Determine direction
        direction = None
        if prediction == self.config.long_prediction:
            direction = 1  # Long
            entry_price = gmo_orderbook.best_ask  # Buy at ask
        elif prediction == self.config.short_prediction:
            direction = -1  # Short
            entry_price = gmo_orderbook.best_bid  # Sell at bid
        else:
            return  # Timeout / no trade

        # Enter position
        self._position_mgr.enter_position(
            ts=ts,
            direction=direction,
            entry_price=entry_price,
            volatility_ema=volatility_ema,
        )

    def _prepare_features(self) -> Optional[list]:
        """Prepare feature vector for model prediction."""
        if not self.config.feature_columns:
            # Use all available features if no specific columns specified
            return list(self._last_features.values())

        # Extract features in the specified order
        feature_vector = []
        for col in self.config.feature_columns:
            if col not in self._last_features:
                LOG.warning(f"Missing feature: {col}")
                return None
            feature_vector.append(self._last_features[col])

        return feature_vector

    def _record_trade(self, exit_ts: float, pnl: float, reason: str) -> None:
        """Record a completed trade."""
        pos = self._position_mgr.position
        if pos is None:
            # Position already cleared, reconstruct from last known state
            LOG.warning("Position already cleared when recording trade")
            return

        # Actually, position is cleared in check_exit/force_exit before this is called
        # We need to capture position info before it's cleared
        # This is a design issue - let me fix the flow

        # For now, just update PnL tracking
        self._cumulative_pnl += pnl
        self._result.pnl_history.append((exit_ts, self._cumulative_pnl))

    def _on_position_exit(
        self,
        entry_ts: float,
        exit_ts: float,
        direction: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> None:
        """Called when a position is exited. Records the trade."""
        trade = TradeRecord(
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason,
        )
        self._result.trades.append(trade)

        self._cumulative_pnl += pnl
        self._result.pnl_history.append((exit_ts, self._cumulative_pnl))

    def reset(self) -> None:
        """Reset simulator state for a new run."""
        self._feature_calc.reset()
        self._vol_tracker.reset()
        self._position_mgr.reset()
        self._result = SimulationResult()
        self._current_position = None
        self._cumulative_pnl = 0.0
        self._event_count = 0
        self._last_features = None


class SimulatorV2(Simulator):
    """
    Improved simulator with proper trade recording.

    Fixes the issue where position info is lost before trade recording.
    """

    def _on_gmo_orderbook(self, orderbook: OrderBook) -> None:
        """Process GMO order book update with proper trade recording."""
        ts = orderbook.get_ts()
        bid = orderbook.best_bid
        ask = orderbook.best_ask

        self._vol_tracker.on_orderbook(orderbook)

        # Check exit for existing position
        if self._position_mgr.has_position:
            pos = self._position_mgr.position
            exited, pnl, reason = self._position_mgr.check_exit(ts, bid, ask)
            if exited:
                # Calculate exit price
                if pos.direction == 1:
                    exit_price = bid
                else:
                    exit_price = ask

                # Record trade with position info captured before exit
                self._on_position_exit(
                    entry_ts=pos.entry_ts,
                    exit_ts=ts,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason=reason,
                )

        # Check entry for new position
        if not self._position_mgr.has_position:
            self._maybe_enter_trade(ts, orderbook)

    def run(self, events: Iterable[Event]) -> SimulationResult:
        """Run the simulation."""
        LOG.info("Starting simulation (V2)")

        for event in events:
            self._process_event(event)
            self._event_count += 1

            if self._event_count % 100000 == 0:
                LOG.info(
                    f"Processed {self._event_count} events, {len(self._result.trades)} trades"
                )

        # Force exit any remaining position
        if self._position_mgr.has_position:
            pos = self._position_mgr.position
            last_ob = self._vol_tracker.get_last_orderbook()
            if last_ob is not None and pos is not None:
                pnl, reason = self._position_mgr.force_exit(
                    last_ob.get_ts(), last_ob.best_bid, last_ob.best_ask
                )
                if pos.direction == 1:
                    exit_price = last_ob.best_bid
                else:
                    exit_price = last_ob.best_ask

                self._on_position_exit(
                    entry_ts=pos.entry_ts,
                    exit_ts=last_ob.get_ts(),
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason=reason,
                )

        LOG.info(
            f"Simulation complete: {self._event_count} events, {len(self._result.trades)} trades"
        )
        return self._result
