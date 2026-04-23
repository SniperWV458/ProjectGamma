# GEX-Fragility × Sentiment Long-Short Backtest

## Strategy Story

Options dealers (market-makers) carry large gamma positions. Whether their hedging activity
stabilizes or destabilizes equity prices depends on the sign and shape of their aggregate
gamma exposure (GEX).

When a stock's spot price is **below its gamma flip level**, dealers are net-short gamma:
to delta-hedge, they must buy into rallies and sell into dips — the opposite of a stabilizing
force. This is the *fragile* regime. When options open interest is skewed toward puts (high
put dominance, D4), it means investors are buying downside protection, which further loads
the dealer's short-gamma book.

The strategy exploits this regime asymmetry:

- **Short leg**: Stocks caught in the fragile regime (spot < gamma flip, put-dominant OI)
  where retail/institutional sentiment is *still optimistic* (elevated sentiment z-score).
  These are mis-priced — the options structure signals fragility, but sentiment has not yet
  repriced the risk.

- **Long leg**: Stocks in the *stable* regime (spot > gamma flip, dealer-dampening) with
  positive sentiment momentum. Dealer hedging reinforces upward moves; sentiment confirms.

The signal is forward-looking by design: all factors are lagged by one trading day before
use, and positions use a lifecycle entry-exit with a time stop to prevent stale exposure.

---

## Factor Definitions (D1–D6)

| Factor | Definition | Data source |
|--------|-----------|-------------|
| **D1** | Net GEX sign: `d1_neg=1` when net GEX < 0 (destabilizing regime) | Panel aggregate GEX |
| **D2** | HHI of GEX concentration across strikes (high = gamma concentrated at one strike) | contract_gex |
| **D3** | Short-term gamma ratio: `|GEX from ≤21 DTE| / |total GEX|` | contract_gex |
| **D4** | Put dominance: `|put GEX| / (|call GEX| + |put GEX|)` | Panel aggregate GEX |
| **D5** | `d5_below_flip=1` when spot < gamma flip level | contract_gex (gamma flip) |
| **D6** | Regime switch: D1 flipped negative in the last N days | Panel aggregate GEX |

D1, D4, D6 are available for every stock in the panel (aggregate GEX data).
D2, D3, D5 require per-strike contract data from `contract_gex.parquet`.

### Fragility Gates

| Gate | Condition | Scope |
|------|-----------|-------|
| `simple` | D1_neg & D4_top_tercile | All panel stocks |
| `full` | D1_neg & D5_below_flip & (D3_top \| D4_top) | Contract-data stocks only |
| `v2` | D5_below_flip & D4_top_tercile | Contract-data stocks only (**recommended**) |
| `v3` | D5_below_flip | Contract-data stocks only (broadest) |

For large-cap stocks, `full` rarely fires because large-caps have positive net GEX ~93% of
the time. Use `v2` or `v3` for actionable signal frequency.

---

## Code Architecture

```
backtest/
├── backtest_framework.py        Base framework: BacktestConfig, BaseSignal, BaseStrategy,
│                                run_backtest(), performance_metrics()
├── gex_factors_builder.py       Pre-computes D2/D3/D5 from contract_gex.parquet via DuckDB.
│                                Run once; outputs data/gex_contract_factors.parquet.
├── sentiment_gex_strategy.py    FragilityGEXSignal + GEXSentimentStrategy classes.
├── run_sentiment_gex.py         Orchestrates all configs, prints performance tables, saves NAV plot.
└── example_sentiment_portfolios.py  Standalone example showing framework usage.
```

### Data Flow

```
contract_gex.parquet
        │
        ▼ (gex_factors_builder.py — run once)
data/gex_contract_factors.parquet   ←── D2 (HHI), D3 (term ratio), D5 (gamma flip level)
        │
        ├── backtest_panel_slim.parquet  ←── 14-column slim panel (created on first run)
        │
        ▼ (FragilityGEXSignal.compute)
panel + D1-D6 + sentiment_z + fragility_gate_*   (all lagged 1 day)
        │
        ▼ (GEXSentimentStrategy.target_weights)
weights DataFrame  →  run_backtest()  →  BacktestResult
```

---

## Strategy Variants

| Parameter | Options | Description |
|-----------|---------|-------------|
| `variant` | `A`, `B` | A: lifecycle entry on gate + sentiment threshold; B: daily top-N by sentiment within gate |
| `gate` | `simple`, `full`, `v2`, `v3` | Fragility gate flavour |
| `neutrality` | `dollar`, `beta` | Dollar-neutral or beta-neutral construction |
| `long_leg` | `momentum`, `none` | Include momentum long leg or short-only book |
| `n_per_leg` | int | Max simultaneous positions per leg |
| `time_stop_days` | int | Force-exit after this many consecutive holding days |

### Position Lifecycle

Positions follow an explicit entry-exit state machine (per stock):
1. **Enter** on the first day the candidate condition (A or B) is True.
2. **Hold** while the condition remains True and `days_held < time_stop_days`.
3. **Exit** when condition turns False (regime flip / sentiment cooling) **or** time stop reached.
4. After a time-stop exit, require at least one False day before re-entering.

This prevents the strategy from churning in and out on noisy daily signals.

---

## How to Run

### Prerequisites

```bash
pip install duckdb pandas pyarrow numpy matplotlib
```

DuckDB is required for `gex_factors_builder.py`. On Windows, use Python 3.11 if DuckDB
is not available for your default interpreter:

```bash
C:/Users/.../Python311/python.exe backtest/gex_factors_builder.py
```

### Step 1 — Build contract-level GEX factors (run once)

```bash
python backtest/gex_factors_builder.py
```

Reads `data/contract_gex.parquet`, outputs `data/gex_contract_factors.parquet`.
Uses DuckDB window functions throughout — scales to arbitrarily large contract_gex files
without loading per-strike data into Python memory.

To rebuild after updating `contract_gex.parquet`:

```bash
python backtest/run_sentiment_gex.py --rebuild-factors
```

### Step 2 — Run the strategy backtest

```bash
python backtest/run_sentiment_gex.py
```

On first run, creates `data/backtest_panel_slim.parquet` (14-column subset) to reduce memory.
Subsequent runs reuse the slim panel.

### Outputs

| File | Description |
|------|-------------|
| `data/gex_contract_factors.parquet` | D2/D3/gamma_flip_level per (secid, date) |
| `data/backtest_panel_slim.parquet` | 14-column panel used by the backtest |
| `backtest/sentiment_gex_nav.png` | NAV and drawdown chart for top configs |
| stdout | Performance tables at 50 bps and 200 bps/yr borrow cost |

---

## Scaling to a Larger Universe

The current default configs use `n_per_leg=3`, calibrated for the 19-stock contract universe.
When a larger `contract_gex.parquet` is available (more stocks), scale positions proportionally:

```python
n_leg = max(3, n_contract_stocks // 6)
```

All code is universe-agnostic: factors are computed for whichever stocks appear in
`contract_gex.parquet`; all other panel stocks receive NaN for D2/D3/D5 and are
automatically excluded from contract-data gates (`v2`, `v3`, `full`).

---

## Key Results (231-stock universe, 2018-06-01 to 2024, borrow = 50 bps/yr)

| config | sharpe | ann_ret% | cum_ret% | ann_vol% | max_dd% | avg_pos |
|--------|-------:|---------:|---------:|---------:|--------:|--------:|
| v2-A-beta-5d-LS | 0.987 | 27.0 | 382.2 | 28.3 | 32.2 | 6.9 |
| v2-A-dollar-5d-LS | 0.945 | 26.0 | 357.8 | 28.9 | 32.2 | 6.9 |
| v2-B-dollar-5d-LS | 0.902 | 26.5 | 368.5 | 31.7 | 38.5 | 8.4 |
| v2-A-beta-10d-LS | 0.890 | 23.3 | 295.9 | 27.9 | 32.2 | 6.9 |
| v2-A-beta-21d-LS | 0.887 | 23.2 | 293.8 | 27.9 | 32.2 | 6.9 |
| v3-A-dollar-5d-LS | 0.481 | 9.2 | 78.5 | 24.6 | 31.3 | 9.7 |
| EW benchmark | 0.831 | 19.2 | — | — | 38.8 | — |

Best config: **v2-A-beta-5d-LS** (gate v2, Variant A, beta-neutral, 5-day time stop, momentum long leg).

The strategy beats the equal-weight benchmark on Sharpe (0.99 vs 0.83), annualised return
(27% vs 19.2%), and max drawdown (32% vs 38.8%) at 50 bps borrow cost.

Note: earlier results with the 19-stock concentrated universe showed higher Sharpe (~1.38)
but were driven by idiosyncratic large-cap tech momentum. The 231-stock results are more
representative of the strategy's systematic edge.

---

## Limitations

- **Short-only requires dollar neutrality**: beta-neutral short-only weights become zero
  (no long leg means zero beta to offset). Use `neutrality="dollar"` for short-only books.
- **Transaction costs**: 5 bps/trade assumed; real short-book costs include locate fees
  and borrow, stress-tested at 50–200 bps/yr.
- **Borrow cost model**: Applied post-hoc as a flat rate on gross short exposure.
- **Time-stop sensitivity**: 5-day stop is significantly better than 21-day for v2 gate,
  suggesting the fragility signal decays quickly once identified.
  Actual borrow costs are higher during dislocations — exactly when the short book fires.
