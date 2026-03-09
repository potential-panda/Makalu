# Smart Data Warehouse (SDW) Specification

## 1. Motivation
The Makalu pipeline generates massive amounts of microstructure data (trades, orderbooks, features) and backtest results stored in Parquet files. While powerful, this data is difficult to query across multiple days or pairs without specialized code. The SDW provides an LLM-integrated, SQL-capable "Virtual Warehouse" to enable:
- **Natural Language Querying**: Ask questions like "What was the average lead-lag correlation during yesterday's BTC dump?"
- **Narrative PnL Attribution**: Automatically explain PnL drops by correlating them with market regimes (volatility, spread, signal decay).
- **Metric Prototyping**: Generate new metrics (e.g., "Spread-to-Volatility ratio") on the fly using LLM-generated Python/SQL.
- **Smart Monitoring**: Push AI-generated annotations to Grafana to explain anomalies.

---

## 2. Architecture

### 2.1 Data Foundation (DuckDB Virtualization)
- **Source**: Date-partitioned Parquet files in `/panda/makalu-data/`.
- **Mechanism**: Use DuckDB's zero-copy Parquet reader to create a unified view across symbols and dates.
- **Schema Mapping**:
    - `microstructure`: Raw features from `feature_engineering`.
    - `backtest_results`: Artifacts from `SimulationResult`.
    - `active_scores`: Seasonality data.

### 2.2 Semantic Layer (Feature Dictionary)
A machine-readable manifest (`features.json`) that maps technical column prefixes to human-readable concepts:
- `ta_`: Trade Aggression (Buy/Sell pressure)
- `bs_`: Book State (Imbalance, Microprice)
- `bc_`: Book Change (Pulls, Refills)
- `ep_`: Execution Price (Slippage, Spread expansion)
- `cf_`: Cross-Feature (Interactions between trade and book)
- `ll_`: Lead-Lag (Correlation and Lags)
- `vo_`: Volatility dynamics

### 2.3 WarehouseAgent (LLM Orchestrator)
- **Interface**: A Python-based agent (LangChain/AutoGPT style) with access to DuckDB and a Python REPL.
- **Context Injection**: The Feature Dictionary and Market Pair mappings are injected into the system prompt to ensure SQL accuracy.
- **Reasoning**: Uses Chain-of-Thought (CoT) to decompose complex queries (e.g., "Find the win-rate during high-volatility regimes").

---

## 3. Implementation Plan

### Phase 1: Metadata & Engine (Foundation)
- `sdw_metadata.py`: Scans `src/feature_engineering/` to generate the Feature Dictionary.
- `sdw_engine.py`: DuckDB wrapper to manage connections and Parquet view registration.

### Phase 2: The Intelligence (Agent)
- `sdw_agent.py`: Implementation of the `WarehouseAgent` class.
- Integration with an LLM provider (OpenAI/Gemini) to handle NL-to-SQL translation.

### Phase 3: Reporting & Attribution
- `sdw_reporter.py`: Logic for correlating backtest PnL with microstructure features.
- Markdown report generator for daily summaries.

### Phase 4: Integration
- Grafana Annotation bridge.
- Command-line interface for ad-hoc analysis.

---

## 4. File Locations

| Component | Path |
|---|---|
| Spec | `doc/smart_warehouse_spec.md` |
| Metadata Tool | `src/warehouse/sdw_metadata.py` |
| DuckDB Engine | `src/warehouse/sdw_engine.py` |
| Warehouse Agent | `src/warehouse/sdw_agent.py` |
| Reporting | `src/warehouse/sdw_reporter.py` |
