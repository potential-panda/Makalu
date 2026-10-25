# Implementing a New Derived Metric

New metrics are implemented as pipeline **Steps** in `src/pipeline/`, registered in
`src/makalu/cli/populate.py`, and documented in `doc/catalog.yaml` and
`doc/dataset_definitions.md`. After populating, run `scan` to make the data queryable.

---

## 1 Pipeline Architecture

```
Pipeline
  └── Stage           (loads raw source data for one date-chunk)
        └── Step(s)   (compute + write one derived dataset)
```

`Pipeline.process(dt_range)` splits the range into 3-day chunks and runs each
`Stage → Step` in order. A Stage loads the source parquet/msgpack data once and
passes it to all its Steps via a `StageContext`.

### 1.1 Source → Stage mapping

| Source data | Stage class | stage_context attribute |
|-------------|-------------|------------------------|
| Bronze OrderBook (parquet) | `OrderBookDerivedStage` | `stage_context.order_books` — `list[OrderBook]` |
| Bronze Trade (parquet) | `TradeDerivedStage` | `stage_context.trades` — `list[Trade]` |
| Silver MarketPrice (parquet) | `ReturnDerivedStage` | `stage_context.price_df` — pandas DataFrame indexed by ts |

All Stage classes are in `src/pipeline/stages.py`.

---

## 2 Step Interface

Implement `Step` from `pipeline.framework`:

```python
from pipeline.framework import PipelineContext, StageContext, Step, StepProgressLogger


class PopulateMyMetricStep(Step):
    def __init__(self, my_param: float):
        super().__init__()
        self.my_param = my_param

    def populate(self, pipeline_context: PipelineContext, stage_context) -> None:
        # 1. Create a parquet writer (see section 3 for factory methods)
        store = pipeline_context.parquet_store.create_single_market(
            "MyMetric", stage_context.market
        )
        count = 0
        # 2. Iterate source data, compute fields, write rows
        for ts, fields in self._compute(stage_context):
            store.add(ts, fields)
            count += 1
        # 3. Always close to flush
        store.close()

    def _compute(self, stage_context):
        """Generator yielding (ts_float, fields_dict) per output row."""
        progress = StepProgressLogger(
            self.__class__.__name__, len(stage_context.order_books)
        )
        for ob in stage_context.order_books:
            # Skip rows outside the populate date range
            if not stage_context.populate_dt_range.is_within(ob.get_ts()):
                continue
            fields = {
                "my_value": ob.mid * self.my_param,
            }
            yield (ob.get_ts(), fields)
            progress.update()
```

`store.add(ts, fields)` accepts `ts` as a Unix timestamp float and `fields` as a
plain `dict[str, float | int | str]`. Do not include `ts` in `fields` — the store
handles it as the row index.

---

## 3 ParquetStore Factory Methods

Call these on `pipeline_context.parquet_store` to get a writer:

| Factory | Layer | Use for |
|---------|-------|---------|
| `create_single_market("DatasetName", market)` | silver | any dataset partitioned only by exchange/symbol |
| `create_single_market("DatasetName", market, layer=Layer.SILVER)` | silver | explicit layer |
| `create_execution_price(market, execution_size_str)` | silver | datasets partitioned by execution_size (MarketPrice pattern) |
| `create_chart(bar_type, market, param_id)` | gold | Chart datasets |
| `create_measured(dataset, market, execution_size_str, param_id)` | gold | Spread, Volatility (Hase-written, read only) |
| `create_lead_lag(market_lead, market_lag, param_id)` | gold | LeadLag |

Import `Layer` if needed: `from makalu.storage.parquet_store import Layer`

---

## 4 Full Example: OrderBook-derived Silver Metric

**File:** `src/pipeline/order_book_derived_steps.py` (add to existing file)

```python
class PopulateMyMetricStep(Step):
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        store = pipeline_context.parquet_store.create_single_market(
            "MyMetric", stage_context.market
        )
        count = 0
        for ts, fields in self._populate(stage_context):
            store.add(ts, fields)
            count += 1
        store.close()

    def _populate(self, stage_context: Context):
        for ob in stage_context.order_books:
            if not stage_context.populate_dt_range.is_within(ob.get_ts()):
                continue
            if not ob.best_bid or not ob.best_ask:
                continue
            yield (ob.get_ts(), {"my_signal": ob.mid})
```

---

## 5 Register in populate.py

`src/makalu/cli/populate.py` — add an `elif` branch in `execute_single_market`:

```python
from pipeline.order_book_derived_steps import PopulateMyMetricStep

elif dataset == "MyMetric":
    pipeline.stage(
        OrderBookDerivedStage(market).step(PopulateMyMetricStep(k=5))
    )
```

The CLI automatically accepts the new dataset name:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py populate MyMetric \
  --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07
```

---

## 6 Document the New Dataset

### 6.1 doc/catalog.yaml

```yaml
- dataset_name: MyMetric
  layer: silver
  description: "One-line description of what this measures."
  grain: per order book update
  primary_timestamp: ts
  partition_columns: [exchange, symbol]
  source_dataset: OrderBook
  path_template: /panda/makalu-data/silver/dataset=MyMetric/exchange={exchange}/symbol={symbol}/date={date}/data.parquet
```

### 6.2 doc/dataset_definitions.md

Add a section under the appropriate layer:

```markdown
### dataset=MyMetric

Source: OrderBook
Grain: per order book update

| Column | Definition |
|--------|------------|
| `ts`   | Unix timestamp (seconds, float) |
| `my_signal` | What this column means |
```

---

## 7 End-to-End Workflow

```bash
# 1. Ensure bronze data exists
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset OrderBook --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-01-07

# 2. Populate the new metric
python /home/lulurun/workspace/Makalu/src/cli.py populate MyMetric \
  --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07

# 3. Register in manifest
python /home/lulurun/workspace/Makalu/src/cli.py scan

# 4. Query to verify
python /home/lulurun/workspace/Makalu/src/cli.py query \
  --sql "SELECT * FROM mymetric WHERE exchange='GMO' AND symbol='BTC_JPY' AND date='2026-01-01' LIMIT 10"
```

Note: the SQL view name is the **lowercase** dataset name (`MyMetric` → `mymetric`).

---

## 8 Key Rules

- Always call `store.close()` — it flushes the parquet buffer
- Skip rows where `stage_context.populate_dt_range.is_within(ts)` is False — the stage
  loads 1 hour of extra data before the range for stateful metrics (e.g. rolling windows)
- The Stage loads data once per 3-day chunk; keep per-row computation cheap
- For metrics that need warm-up state (rolling windows, EWM), initialise the state
  object outside the loop and feed rows in order — the stage guarantees chronological order
- If the metric needs a `param_id` partition (e.g. configurable window sizes), use
  `create_single_market("MyMetric_paramid", market)` and encode the param in the dataset name,
  or add a custom Store class following the `MeasuredStore` pattern in `qate/store/parquet_store.py`
