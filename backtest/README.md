# GEX-Fragility × Sentiment Long-Short Backtest

## Strategy Story

Options dealers (market-makers) carry large gamma positions. Whether their hedging activity
stabilizes or destabilizes equity prices depends on the sign and shape of their aggregate
gamma exposure (GEX).

When a stock's spot price is **below its gamma flip level**, dealers are net-short gamma:
to delta-hedge, they must sell into dips and buy into rallies — a destabilizing feedback
loop. This is the *fragile* regime. When open interest is also skewed toward puts (high put
dominance), investors are buying downside protection, which concentrates more short-gamma
exposure on the dealer's book and amplifies the feedback on downside moves.

The strategy exploits this regime asymmetry:

- **Short leg**: Stocks in the fragile regime (spot < gamma flip, put-dominant OI) where
  sentiment is *still elevated* — mis-priced, because the options structure signals
  fragility but sentiment has not yet repriced the risk.
- **Long leg**: Stocks in the stable regime (spot > gamma flip, dealer-dampening) with
  positive sentiment momentum. Dealer hedging reinforces upward moves; sentiment confirms.

All factors are lagged one trading day before use. Positions follow an explicit lifecycle
entry-exit to prevent stale exposure after the regime signal weakens.

---

## Factor Definitions (D1–D6)

| Factor | Definition | Data source |
|--------|-----------|-------------|
| **D1** | Net GEX sign: `d1_neg=1` when net GEX < 0 (destabilizing) | Panel aggregate GEX |
| **D2** | HHI of GEX concentration across strikes | contract_gex |
| **D3** | Short-term gamma ratio: `\|GEX ≤21 DTE\| / \|total GEX\|` | contract_gex |
| **D4** | Put dominance: `\|put GEX\| / (\|call GEX\| + \|put GEX\|)` | Panel aggregate GEX |
| **D5** | `d5_below_flip=1` when spot < gamma flip level | contract_gex |
| **D6** | Regime switch: D1 flipped negative within last N days | Panel aggregate GEX |

D1, D4, D6 are available for every stock in the panel (aggregate GEX).
D2, D3, D5 require per-strike contract data and are merged from `gex_contract_factors.parquet`;
stocks without contract data receive NaN and are excluded from gates that use D5.

### Fragility Gates

| Gate | Condition | Notes |
|------|-----------|-------|
| `simple` | D1\_neg & D4\_top\_tercile | All panel stocks; fires broadly |
| `full` | D1\_neg & D5\_below\_flip & (D3\_top \| D4\_top) | Rarely fires for large-caps |
| `v2` | D5\_below\_flip & D4\_top\_tercile | **Recommended** — ~14% hit rate |
| `v3` | D5\_below\_flip | Broadest — ~43% hit rate |

---

## Code Architecture

```
backtest/
├── backtest_framework.py        Core engine: BacktestConfig, BaseSignal, BaseStrategy,
│                                BaseEntryExitRule, run_backtest(), performance_metrics()
├── gex_factors_builder.py       Builds D2/D3/D5 from contract_gex.parquet via DuckDB.
│                                Run once; writes data/gex_contract_factors.parquet.
├── sentiment_gex_strategy.py    FragilityGEXSignal (computes D1–D6, gates, sentiment_z)
│                                + GEXSentimentStrategy (target weights with lifecycle).
├── ml_exit.py                   ML-based exit rule (GradientBoosting walk-forward).
│                                Standalone evaluation + MLExitRule(BaseEntryExitRule).
├── run_sentiment_gex.py         Runs all configs, vol-targeting variants, saves outputs.
└── example_sentiment_portfolios.py  Standalone framework usage example.
```

### Data Flow

```
contract_gex.parquet
        │
        ▼  gex_factors_builder.py  (run once)
data/gex_contract_factors.parquet    — D2, D3, gamma_flip_level per (secid, date)
        │
        ├── data/backtest_panel_slim.parquet   — 14-col panel (auto-created on first run)
        │
        ▼  FragilityGEXSignal.compute()
panel + D1–D6 + sentiment_z + fragility_gate_*   (all shifted 1 day per permno)
        │
        ▼  GEXSentimentStrategy.target_weights()
weights DataFrame
        │
        ├──[optional]── MLExitRule.apply()   — drop positions the model predicts to lose
        │
        ▼  run_backtest()  →  BacktestResult
```

---

## Strategy Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `variant` | `A`, `B` | A: lifecycle entry when gate + sentiment > threshold; B: daily top-N ranked by sentiment within gate |
| `gate` | `simple`, `full`, `v2`, `v3` | Fragility gate condition |
| `neutrality` | `dollar`, `beta` | Dollar-neutral or beta-neutral. Short-only books must use `dollar`. |
| `long_leg` | `momentum`, `none` | Include stable-regime momentum long, or short-only |
| `n_per_leg` | int | Max simultaneous positions per leg per day |
| `time_stop_days` | int | Force-exit after this many consecutive holding days |

### Position Lifecycle (per stock)

1. **Enter** on the first day the candidate condition is True.
2. **Hold** while condition stays True and `days_held < time_stop_days`.
3. **Exit** when condition turns False (regime flip / sentiment cooling) or time stop reached.
4. Require at least one False day before re-entering after a time-stop exit.

---

## How to Run

### Prerequisites

```bash
pip install duckdb pandas pyarrow numpy matplotlib scikit-learn
```

On Windows, use Python 3.11 if DuckDB is unavailable for the default interpreter:

```bash
C:/Users/.../Python311/python.exe backtest/gex_factors_builder.py
```

### Step 1 — Build contract-level GEX factors (run once)

```bash
python backtest/gex_factors_builder.py
```

Reads `data/contract_gex.parquet`, writes `data/gex_contract_factors.parquet`.
All computation runs inside DuckDB — no per-strike data is loaded into Python memory,
so it scales to arbitrarily large contract files.

To rebuild after updating `contract_gex.parquet`:

```bash
python backtest/run_sentiment_gex.py --rebuild-factors
```

### Step 2 — Run strategy backtest and parameter configs

```bash
python backtest/run_sentiment_gex.py
```

On first run, creates `data/backtest_panel_slim.parquet` (14-column subset).
Runs base configs + vol-targeted variants, prints performance tables, saves all outputs.

### Step 3 — Run ML exit evaluation (optional)

```bash
python backtest/ml_exit.py
```

Runs base strategy vs ML-exit strategy side by side, prints Sharpe comparison and
feature importances, saves NAV chart to `backtest/output/ml_exit_nav.png`.

### Outputs

| File | Description |
|------|-------------|
| `data/gex_contract_factors.parquet` | D2/D3/gamma_flip_level per (secid, date) |
| `data/backtest_panel_slim.parquet` | 14-column slim panel used by the backtest |
| `backtest/sentiment_gex_nav.png` | NAV + drawdown chart for base configs |
| `backtest/output/positions.parquet` | All position weights with config label |
| `backtest/output/daily_returns.csv` | Per-config daily return series |
| `backtest/output/performance_summary.csv` | Full performance table (50 bps borrow) |
| `backtest/output/ml_exit_nav.png` | Base vs ML-exit NAV comparison chart |

---

## Key Results (2018-06-01 to 2024, borrow = 50 bps/yr, txcost = 5 bps)

### Base Strategy — Parameter Sweep

| Config | Sharpe | Ann Ret | Cum Ret | Ann Vol | Max DD | Avg Pos |
|--------|-------:|--------:|--------:|--------:|-------:|--------:|
| v2-A-beta-5d-LS | **0.987** | 27.0% | 382% | 28.3% | 32.2% | 6.9 |
| v2-A-dollar-5d-LS | 0.945 | 26.0% | 358% | 28.9% | 32.2% | 6.9 |
| v2-B-dollar-5d-LS | 0.902 | 26.5% | 369% | 31.7% | 38.5% | 8.4 |
| v2-A-beta-10d-LS | 0.890 | 23.3% | 296% | 27.9% | 32.2% | 6.9 |
| v3-A-dollar-5d-LS | 0.481 | 9.2% | 79% | 24.6% | 31.3% | 9.7 |
| **EW benchmark** | 0.831 | 19.2% | — | — | 38.8% | — |

Best base config: **v2-A-beta-5d-LS** — gate v2, Variant A, beta-neutral, 5-day time stop,
momentum long leg. Beats the equal-weight benchmark on Sharpe, return, and drawdown.

Key finding from the sweep: a **5-day time stop consistently outperforms 21-day** across
all gate/variant combinations. The fragility signal is fast-decaying — once the gate fires,
the trade window is short.

### Vol-Targeting Variants (best base config)

Vol-targeting scales daily position sizes so rolling 20-day realized vol tracks a target,
reducing exposure during high-volatility periods (e.g. 2020 COVID).

| Config | Sharpe | Ann Ret | Ann Vol | Max DD |
|--------|-------:|--------:|--------:|-------:|
| Unscaled (base) | 0.987 | 27.0% | 28.3% | 32.2% |
| Vol-target 20% | 0.807 | 15.4% | 20.3% | 17.5% |
| Vol-target 15% | 0.716 | 11.0% | 16.5% | 17.0% |
| Vol-target 10% | 0.615 | 6.6% | 11.5% | 16.1% |

Vol-targeting halves the max drawdown (32% → 17%) at the cost of lower Sharpe, because it
scales down aggressively after volatile periods and misses the subsequent recovery. Choose
based on drawdown tolerance: unscaled for best Sharpe, vol-targeted for smoother NAV.

---

## ML Exit Evaluation

The fixed time-stop is replaced by a **walk-forward GradientBoosting classifier** that
decides, each day a position is held, whether to exit or continue.

**Setup:**
- Training window: rolling 2 years (504 trading days), retrained quarterly
- Features: `days_held`, `sentiment_z`, `net_gex_z`, `d4_put_dom`, `d5_below_flip`,
  `stock_ret_1d`, `stock_ret_5d`, `pos_return_sofar`, `market_ret_5d`, `market_ret_20d`
- Label: exit = 1 if holding for another 5 days would lose money (adjusted for side)

**Results:**

| | Sharpe | Ann Ret | Ann Vol | Max DD |
|--|-------:|--------:|--------:|-------:|
| Base (time-stop 5d) | 0.999 | 27.5% | 28.3% | 32.2% |
| **ML exit** | **1.441** | **41.3%** | **26.4%** | **32.2%** |

The ML exit improves Sharpe from 1.00 → **1.44** and annual return from 27.5% → **41.3%**,
while keeping the same max drawdown. It achieves this by holding profitable positions longer
(staying in when the regime remains intact) and cutting losses earlier (exiting when momentum
reverses), rather than applying a rigid calendar stop.

**Feature importances from the final model:**

| Feature | Importance | Interpretation |
|---------|----------:|----------------|
| `d4_put_dom` | 0.332 | Put dominance dropping → fragile regime easing → exit |
| `stock_ret_5d` | 0.199 | Recent momentum reversal predicts position loss |
| `d5_below_flip` | 0.154 | Stock moving above gamma flip → regime shift → exit |
| `stock_ret_1d` | 0.119 | Yesterday's return as short-term reversal signal |
| `market_ret_5d` | 0.064 | Broad market rally → short book at risk |
| `net_gex_z` | 0.037 | GEX regime intensity |
| `pos_return_sofar` | 0.033 | Profit-take / stop-loss trigger |

The model largely recovers the fundamental thesis from data: **stay in the short as long as
put dominance is elevated and the stock remains below the gamma flip; exit when either
condition weakens.**

Run standalone:
```bash
python backtest/ml_exit.py
```

---

## Limitations

- **Short-only must use dollar neutrality**: beta-neutral short-only weights become zero
  because `w_short = -beta_long_total / (n_short * beta_short)` and `beta_long_total = 0`
  when there is no long leg.
- **Borrow cost model**: applied post-hoc as a flat rate on gross short exposure.
  Real borrow is higher during dislocations — precisely when the short book is most active.
- **Transaction costs**: 5 bps/trade assumed; real round-trip costs for liquid names are
  typically 2–10 bps depending on size and venue.
- **ML exit lookahead risk**: walk-forward training avoids in-sample bias, but the initial
  2-year training period (2018–2020) is short and includes COVID — a regime not seen before.
  Out-of-sample improvement should be verified on data beyond 2024.
