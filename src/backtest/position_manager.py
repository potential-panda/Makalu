"""
Position manager for backtesting.

Manages positions and handles trade entry/exit based on triple barrier logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)


@dataclass
class PositionConfig:
    """Configuration for position management."""

    barrier_multiplier: float = 0.7
    timeout_sec: int = 30


@dataclass
class Position:
    """Represents an open position."""

    entry_ts: float
    entry_price: float
    direction: int  # 1 for long, -1 for short
    pt_level: float  # profit-take price level
    sl_level: float  # stop-loss price level
    timeout_ts: float  # timeout timestamp


class PositionManager:
    """
    Manages positions and handles trade entry/exit based on triple barrier logic.

    Triple barrier exit conditions:
    - Profit-take: Price reaches profit target
    - Stop-loss: Price reaches stop level
    - Timeout: Position held for too long
    """

    def __init__(self, config: PositionConfig):
        self.config = config
        self._position: Optional[Position] = None

    @property
    def has_position(self) -> bool:
        """Check if there is an open position."""
        return self._position is not None

    @property
    def position(self) -> Optional[Position]:
        """Get the current position."""
        return self._position

    def enter_position(
        self,
        ts: float,
        direction: int,
        entry_price: float,
        volatility_ema: float,
    ) -> Position:
        """
        Enter a new position.

        Args:
            ts: Entry timestamp
            direction: 1 for long, -1 for short
            entry_price: Entry price (ask for long, bid for short)
            volatility_ema: Current volatility EMA for barrier calculation

        Returns:
            The created Position object
        """
        if self._position is not None:
            raise ValueError("Cannot enter position while already holding one")

        barrier_pct = volatility_ema * self.config.barrier_multiplier

        if direction == 1:  # Long
            pt_level = entry_price * (1 + barrier_pct)
            sl_level = entry_price * (1 - barrier_pct)
        else:  # Short
            pt_level = entry_price * (1 - barrier_pct)
            sl_level = entry_price * (1 + barrier_pct)

        self._position = Position(
            entry_ts=ts,
            entry_price=entry_price,
            direction=direction,
            pt_level=pt_level,
            sl_level=sl_level,
            timeout_ts=ts + self.config.timeout_sec,
        )

        LOG.debug(
            f"Entered {'LONG' if direction == 1 else 'SHORT'} at {entry_price:.6f}, "
            f"PT={pt_level:.6f}, SL={sl_level:.6f}, timeout={self.config.timeout_sec}s"
        )

        return self._position

    def check_exit(
        self,
        ts: float,
        bid: float,
        ask: float,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Check if the current position should be exited.

        Args:
            ts: Current timestamp
            bid: Current bid price (exit price for long)
            ask: Current ask price (exit price for short)

        Returns:
            Tuple of (exited, pnl, exit_reason)
            - exited: True if position was closed
            - pnl: PnL of the trade (None if not exited)
            - exit_reason: 'pt', 'sl', or 'timeout' (None if not exited)
        """
        if self._position is None:
            return False, None, None

        pos = self._position
        exit_price = None
        exit_reason = None

        if pos.direction == 1:  # Long position
            # Check profit-take (using mid approximation for barrier check)
            mid = (bid + ask) / 2
            if mid >= pos.pt_level:
                exit_price = bid  # Exit long at bid
                exit_reason = "pt"
            # Check stop-loss
            elif mid <= pos.sl_level:
                exit_price = bid  # Exit long at bid
                exit_reason = "sl"
            # Check timeout
            elif ts >= pos.timeout_ts:
                exit_price = bid  # Exit long at bid
                exit_reason = "timeout"

        else:  # Short position
            mid = (bid + ask) / 2
            # Check profit-take (price goes down)
            if mid <= pos.pt_level:
                exit_price = ask  # Exit short at ask
                exit_reason = "pt"
            # Check stop-loss (price goes up)
            elif mid >= pos.sl_level:
                exit_price = ask  # Exit short at ask
                exit_reason = "sl"
            # Check timeout
            elif ts >= pos.timeout_ts:
                exit_price = ask  # Exit short at ask
                exit_reason = "timeout"

        if exit_price is not None:
            # Calculate PnL
            if pos.direction == 1:  # Long
                pnl = exit_price - pos.entry_price
            else:  # Short
                pnl = pos.entry_price - exit_price

            LOG.debug(
                f"Exited {'LONG' if pos.direction == 1 else 'SHORT'} at {exit_price:.6f}, "
                f"reason={exit_reason}, PnL={pnl:.6f}"
            )

            self._position = None
            return True, pnl, exit_reason

        return False, None, None

    def force_exit(self, ts: float, bid: float, ask: float) -> Tuple[float, str]:
        """
        Force exit the current position (e.g., at end of simulation).

        Args:
            ts: Current timestamp
            bid: Current bid price
            ask: Current ask price

        Returns:
            Tuple of (pnl, exit_reason)
        """
        if self._position is None:
            raise ValueError("No position to exit")

        pos = self._position

        if pos.direction == 1:  # Long
            exit_price = bid
            pnl = exit_price - pos.entry_price
        else:  # Short
            exit_price = ask
            pnl = pos.entry_price - exit_price

        LOG.debug(
            f"Force exited {'LONG' if pos.direction == 1 else 'SHORT'} at {exit_price:.6f}, "
            f"PnL={pnl:.6f}"
        )

        self._position = None
        return pnl, "force"

    def reset(self) -> None:
        """Reset position state."""
        self._position = None
