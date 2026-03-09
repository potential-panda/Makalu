# Backtesting Simulation Specification

## Overview

This document specifies a comprehensive event-driven backtesting system for cross-exchange trading strategies. The system processes market data from multiple exchanges in chronological order, calculates features in real-time, makes trading decisions using a trained ML model, and tracks PnL.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Backtesting System                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │  BINANCE    │   │  BINANCE    │   │    GMO      │                   │
│  │ OrderBooks  │   │   Trades    │   │ OrderBooks  │                   │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                   │
│         │                 │                 │                           │
│         └────────────┬────┴─────────────────┘                           │
│                      ▼                                                  │
│            ┌─────────────────┐                                          │
│            │  Event Merger   │  (chronological ordering)                │
│            └────────┬────────┘                                          │
│                     ▼                                                   │
│            ┌─────────────────┐                                          │
│            │   Simulator     │                                          │
│            │                 │                                          │
│            │  ┌───────────┐  │     ┌─────────────────┐                  │
│            │  │ BINANCE   │  │     │  Feature Store  │                  │
│            │  │ Processor │──┼────►│  (rolling)      │                  │
│            │  └───────────┘  │     └─────────────────┘                  │
│            │                 │                                          │
│            │  ┌───────────┐  │     ┌─────────────────┐                  │
│            │  │   GMO     │  │     │ Volatility/     │                  │
│            │  │ Processor │──┼────►│ Spread Tracker  │                  │
│            │  └───────────┘  │     └─────────────────┘                  │
│            │                 │              │                           │
│            │  ┌───────────┐  │              ▼                           │
│            │  │ Position  │◄─┼─────┌─────────────────┐                  │
│            │  │ Manager   │  │     │   ML Model      │                  │
│            │  └───────────┘  │     │  (LightGBM)     │                  │
│            │        │        │     └─────────────────┘                  │
│            └────────┼────────┘                                          │
│                     ▼                                                   │
│            ┌─────────────────┐                                          │
│            │   PnL Tracker   │                                          │
│            └─────────────────┘                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Loading (`src/backtest/data_loader.py`)

Responsible for loading market data from parquet stores.

```python
@dataclass
class BacktestDataConfig:
    dt_range: DtRange
    market_binance: Market  # BINANCE:XRP_USDT
    market_gmo: Market      # GMO:XRP_JPY
    parquet_root: str

class BacktestDataLoader:
    def __init__(self, config: BacktestDataConfig)

    def load_binance_orderbooks(self) -> list[OrderBook]
    def load_binance_trades(self) -> list[Trade]
    def load_gmo_orderbooks(self) -> list[OrderBook]
```

### 2. Event Merger (`src/backtest/event_merger.py`)

Merges events from multiple sources into a single chronologically ordered stream.

```python
@dataclass
class Event:
    ts: float
    event_type: EventType  # BINANCE_ORDERBOOK, BINANCE_TRADE, GMO_ORDERBOOK
    data: Union[OrderBook, Trade]

class EventMerger:
    @staticmethod
    def merge(
        binance_orderbooks: list[OrderBook],
        binance_trades: list[Trade],
        gmo_orderbooks: list[OrderBook]
    ) -> Generator[Event, None, None]
```

### 3. Feature Calculator (`src/backtest/feature_calculator.py`)

Calculates features from BINANCE market data in real-time.

```python
@dataclass
class FeatureConfig:
    # Order book features
    book_levels: int = 5

    # Trade aggression features
    aggression_windows: list[int] = field(default_factory=lambda: [100, 500, 1000])

    # Book change features
    book_change_window: int = 100

class FeatureCalculator:
    def __init__(self, config: FeatureConfig)

    def on_orderbook(self, orderbook: OrderBook) -> None
    def on_trade(self, trade: Trade) -> None
    def get_features(self) -> dict[str, float] | None
```

**Features calculated:**
- **Order Book Features** (from `intent.py`):
  - `microprice`: Volume-weighted mid price
  - `volume_imbalance`: Bid/ask volume ratio
  - `spread_bps`: Spread in basis points
  - `depth_imbalance_X`: Depth imbalance at level X

- **Trade Aggression Features** (from `aggression.py`):
  - `buy_volume_Xms`: Buy volume in last X ms
  - `sell_volume_Xms`: Sell volume in last X ms
  - `aggression_ratio_Xms`: Buy/Sell ratio in last X ms

- **Book Change Features** (from `book_change.py`):
  - `bid_add_volume`: Volume added to bid side
  - `bid_cancel_volume`: Volume cancelled from bid side
  - `ask_add_volume`: Volume added to ask side
  - `ask_cancel_volume`: Volume cancelled from ask side

### 4. Volatility Tracker (`src/backtest/volatility_tracker.py`)

Tracks volatility and spread from GMO market data for trade entry filtering.

```python
@dataclass
class VolatilityConfig:
    return_shift_sec: int = 30
    volatility_lookback_sec: int = 1800
    ema_span_sec: int = 600
    min_vol_spread_ratio: float = 1.2

class VolatilityTracker:
    def __init__(self, config: VolatilityConfig)

    def on_orderbook(self, orderbook: OrderBook) -> None
    def get_volatility_ema(self) -> float | None
    def get_spread_pct_ema(self) -> float | None
    def is_tradeable(self) -> bool  # vol > spread * min_vol_spread_ratio
```

### 5. Position Manager (`src/backtest/position_manager.py`)

Manages positions and handles trade entry/exit based on triple barrier logic.

```python
@dataclass
class PositionConfig:
    barrier_multiplier: float = 0.7
    timeout_sec: int = 30

class Position:
    entry_ts: float
    entry_price: float  # GMO ask for long, bid for short
    direction: int      # 1 for long, -1 for short
    pt_level: float     # profit-take price level
    sl_level: float     # stop-loss price level
    timeout_ts: float   # timeout timestamp

class PositionManager:
    def __init__(self, config: PositionConfig)

    @property
    def has_position(self) -> bool

    def enter_position(
        self,
        ts: float,
        direction: int,  # 1=long, -1=short
        entry_price: float,
        volatility_ema: float
    ) -> Position

    def check_exit(
        self,
        ts: float,
        bid: float,
        ask: float
    ) -> tuple[bool, float | None]  # (exited, pnl)
```

### 6. Simulator (`src/backtest/simulator.py`)

Main simulation loop that processes events and coordinates all components.

```python
@dataclass
class SimulatorConfig:
    feature_config: FeatureConfig
    volatility_config: VolatilityConfig
    position_config: PositionConfig
    model_path: str
    feature_columns: list[str]

@dataclass
class TradeRecord:
    entry_ts: float
    exit_ts: float
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str  # 'pt', 'sl', 'timeout'

class Simulator:
    def __init__(self, config: SimulatorConfig)

    def run(self, events: Iterable[Event]) -> SimulationResult

    def _on_binance_orderbook(self, orderbook: OrderBook) -> None
    def _on_binance_trade(self, trade: Trade) -> None
    def _on_gmo_orderbook(self, orderbook: OrderBook) -> None
    def _maybe_enter_trade(self, ts: float) -> None
    def _check_exit(self, ts: float, gmo_orderbook: OrderBook) -> None
```

### 7. Simulation Result (`src/backtest/result.py`)

Container for simulation results and analysis methods.

```python
@dataclass
class SimulationResult:
    trades: list[TradeRecord]
    pnl_history: list[tuple[float, float]]  # (ts, cumulative_pnl)

    @property
    def total_pnl(self) -> float

    @property
    def trade_count(self) -> int

    @property
    def win_rate(self) -> float

    @property
    def avg_pnl_per_trade(self) -> float

    def to_dataframe(self) -> pd.DataFrame

    def plot_pnl(self) -> None

    def plot_trades_by_hour(self) -> None

    def summary(self) -> dict
```

## Simulation Flow

```
for each event in merged_events:

    if event.type == BINANCE_ORDERBOOK:
        feature_calculator.on_orderbook(event.data)

    elif event.type == BINANCE_TRADE:
        feature_calculator.on_trade(event.data)

    elif event.type == GMO_ORDERBOOK:
        volatility_tracker.on_orderbook(event.data)

        # Check exit for existing position
        if position_manager.has_position:
            exited, pnl = position_manager.check_exit(
                event.ts,
                event.data.best_bid,
                event.data.best_ask
            )
            if exited:
                record_trade(pnl)

        # Check entry for new position
        if not position_manager.has_position:
            if volatility_tracker.is_tradeable():
                features = feature_calculator.get_features()
                if features is not None:
                    prediction = model.predict(features)
                    if prediction == 2:  # Long
                        position_manager.enter_position(
                            ts, direction=1,
                            entry_price=event.data.best_ask,
                            volatility_ema=volatility_tracker.get_volatility_ema()
                        )
                    elif prediction == 0:  # Short
                        position_manager.enter_position(
                            ts, direction=-1,
                            entry_price=event.data.best_bid,
                            volatility_ema=volatility_tracker.get_volatility_ema()
                        )
```

## File Structure

```
/home/lulurun/workspace/Makalu/
├── src/
│   └── backtest/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── event_merger.py
│       ├── feature_calculator.py
│       ├── volatility_tracker.py
│       ├── position_manager.py
│       ├── simulator.py
│       └── result.py
└── notebooks/
    └── backtest_simulation.ipynb
```

## Notebook Usage

```python
# Load configuration
from backtest.data_loader import BacktestDataConfig, BacktestDataLoader
from backtest.simulator import Simulator, SimulatorConfig
from backtest.event_merger import EventMerger

# Configure
data_config = BacktestDataConfig(
    dt_range=DtRange("2025-01-15", "2025-01-16"),
    market_binance=Market(ExchangeName.BINANCE, Symbol.XRP_USDT),
    market_gmo=Market(ExchangeName.GMO, Symbol.XRP_JPY),
    parquet_root="/path/to/data"
)

sim_config = SimulatorConfig(
    model_path="/path/to/model.pkl",
    feature_columns=[...],
    ...
)

# Load data
loader = BacktestDataLoader(data_config)
binance_obs = loader.load_binance_orderbooks()
binance_trades = loader.load_binance_trades()
gmo_obs = loader.load_gmo_orderbooks()

# Merge events
events = EventMerger.merge(binance_obs, binance_trades, gmo_obs)

# Run simulation
simulator = Simulator(sim_config)
result = simulator.run(events)

# Analyze results
print(result.summary())
result.plot_pnl()
result.plot_trades_by_hour()
```

## Key Design Decisions

1. **Event-Driven Architecture**: Process events one-by-one in chronological order to accurately simulate real-time trading conditions.

2. **Generator Pattern**: Use generators for event merging to minimize memory usage when processing large datasets.

3. **Separation of Concerns**: Each component (feature calculation, volatility tracking, position management) is isolated for testability and maintainability.

4. **GMO as Execution Venue**: All trades are executed on GMO (the lagging market), using BINANCE data for signal generation.

5. **Triple Barrier Exit**: Positions are closed when hitting profit-take, stop-loss, or timeout barriers.

6. **Volatility Filter**: Only enter trades when market conditions are favorable (volatility exceeds spread by configured ratio).

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `barrier_multiplier` | 0.7 | Multiplier for volatility to set PT/SL barriers |
| `timeout_sec` | 30 | Maximum time to hold a position |
| `min_vol_spread_ratio` | 1.2 | Minimum volatility/spread ratio for entry |
| `return_shift_sec` | 30 | Return calculation period for volatility |
| `volatility_lookback_sec` | 1800 | Lookback window for volatility calculation |
| `ema_span_sec` | 600 | EMA span for smoothing |

## Notes

- The simulation uses GMO order book updates as the timing for both entry and exit decisions, since that's where execution happens.
- Features are calculated from BINANCE data but used when GMO events arrive.
- The model predicts 0=Short, 1=NoEntry, 2=Long based on the training labels.
