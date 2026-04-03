# `factor_analysis_v0.py`

This document explains the end-to-end factor analysis pipeline implemented in `factor_analysis_v0.py`. The code is organized as a configurable experiment engine built around the `GEXCollaborativeEffectExperiment` class. It starts from a daily stock-level panel that merges GEX data with CRSP data, attaches external factor files, standardizes the panel, constructs forward-looking targets, and then runs a sequence of seven analysis phases.

The main design idea is:

1. Build a clean daily `permno`-`date` panel.
2. Express GEX and factor signals in a comparable cross-sectional format.
3. Test whether GEX behaves like a market fragility regime variable.
4. Measure how factor efficacy changes across GEX regimes.
5. Train classification models for tail events.
6. Use those outputs to scale a factor portfolio overlay.

## 1. What The Code Does

The file contains:

- Dataclass-based configuration objects for data loading, preprocessing, targets, regressions, classification, portfolio overlays, and multivariate modeling.
- A main experiment class, `GEXCollaborativeEffectExperiment`.
- Phase runners:
  - `run_phase1()`
  - `run_phase2()`
  - `run_phase3()`
  - `run_phase4()`
  - `run_phase5_random_forest()`
  - `run_phase6_portfolio_overlay()`
  - `run_phase7_multivariate_model_training()`

The pipeline is stateful: each phase reads from `self.panel` and writes results into `self.artifacts`.

## 2. Data Model

The core panel is keyed by:

- `permno`
- `date`

The base panel is formed by merging:

- an underlying GEX daily file
- a CRSP daily file

Important default columns:

- `net_gex_1pct`: main GEX signal
- `ret`: stock return used for forward target construction
- `spot`: underlying reference price from GEX-side data
- `prc`: CRSP price
- `prc_abs`: absolute price derived from `prc`
- `market_equity_crsp`: constructed as absolute price times shares outstanding when available
- `price_underlying`: alias of `spot`
- `price_stock`: alias of `abs(prc)` when available

## 3. Configuration Objects

The file uses small dataclasses to separate concerns.

### `FactorSpec`

Describes one factor source to merge:

- source path
- source id column
- date column
- factor columns
- frequency: `daily` or `monthly`
- merge direction
- suffixing
- identifier type: `permno`, `ticker`, or `secid`
- optional lags

### `IdentifierConfig`

Controls mapping from non-`permno` identifiers into `permno`, including:

- mapping file path
- ticker normalization
- duplicate resolution policy

### `DataConfig`

Controls the anchor inputs and global sample filters:

- GEX path
- CRSP path
- optional `start_date` and `end_date`
- minimum absolute price filter

### `ColumnConfig`

Defines canonical column names for:

- id and date
- GEX column
- return column
- price columns
- optional controls

### `PreprocessConfig`

Controls panel cleaning:

- winsorization
- cross-sectional z-scoring
- minimum cross-sectional size
- missing-value filtering
- fill method

### `TargetConfig`

Controls forward outcome construction:

- forecast horizons
- return type
- realized volatility
- downside semivariance
- tail indicators

### `RegressionConfig`

Controls Phase 2 and Phase 3 regression/sorting logic:

- Fama-MacBeth settings
- Newey-West lags
- control variables
- bucket counts

### `ClassificationConfig`

Controls Phase 4 logistic models:

- target columns
- date-based train/test split
- model names
- regularization strength
- class weighting

### `RandomForestPhase5Config`

Controls Phase 5 nonlinear binary classification:

- targets
- feature-set variants
- explicit date split
- random forest hyperparameters

### `Phase6OverlayConfig`

Controls Phase 6 portfolio construction and overlays:

- bucket portfolio settings
- long/short sides
- equal- or value-weighting
- GEX-based scaling
- Phase 5 probability scaling
- transaction-cost assumption

### `Phase7MultivariateConfig`

Controls Phase 7 curated multivariate modeling:

- curated factor list
- GEX inclusion
- GEX interactions
- elastic-net logistic regression settings
- histogram gradient boosting settings
- permutation importance

## 4. Pipeline Overview

The intended execution order is:

1. `run_phase1()`: build the research panel.
2. `run_phase2()`: validate GEX and estimate interaction regressions.
3. `run_phase3()`: compare factor efficacy across regimes and run double sorts.
4. `run_phase4()`: classify tail outcomes with simple linear probabilistic models.
5. `run_phase5_random_forest()`: add nonlinear tail prediction.
6. `run_phase6_portfolio_overlay()`: backtest overlay-scaled factor portfolios.
7. `run_phase7_multivariate_model_training()`: train curated multivariate predictive models.

## 5. Phase 1: Build The Research Panel

`run_phase1()` does six things:

1. Load the anchor panel.
2. Merge factor sources.
3. Preprocess the panel.
4. Build forward targets.
5. Build GEX regimes.
6. Save the panel snapshot and metadata.

### 5.1 Load The Anchor Panel

`load_base_panel()`:

- reads the GEX file and CRSP file
- standardizes dates
- optionally drops rows with missing `permno`
- creates `prc_abs = |prc|` if configured
- applies date filters
- optionally applies a minimum stock price filter
- deduplicates by `permno` and `date`
- left-merges CRSP into the GEX panel

Derived values:

- Absolute price:

```text
prc_abs = |prc|
```

- Market equity when `shrout` is present:

```text
market_equity_crsp = |prc| * shrout
```

### 5.2 Load And Merge Factor Sources

`load_factor_source()` and `merge_factors()` support:

- daily factors merged on `permno` and `date`
- monthly factors merged on `permno` and year-month
- factor files already in the base panel
- source ids in `permno`, `ticker`, or `secid`

#### Identifier Mapping

When a factor file is not keyed by `permno`, the code maps it through an identifier table.

Examples:

- `ticker -> permno`
- `secid -> permno`

Ambiguous mappings can be:

- kept as first match
- kept as last match
- dropped entirely

#### Daily Merge

For daily factors:

```text
panel_{t,i} <- merge(panel_{t,i}, factor_{t,i})
```

If `lag_days != 0`, factor dates are shifted before merging:

```text
date_effective = date_source + lag_days
```

#### Monthly Merge

For monthly factors, both panel and factor dates are converted to year-month keys:

```text
ym_panel  = period_month(date_panel)
ym_factor = period_month(date_factor) + lag_months
```

Then the merge key is:

```text
(permno, ym)
```

### 5.3 Preprocess The Panel

`preprocess_panel()` performs:

- infinity cleanup
- numeric coercion
- missingness filtering
- optional within-stock forward/backward filling
- cross-sectional size filtering
- date-wise winsorization
- date-wise z-scoring

#### Missingness Filter

For each active factor column `f`, the missing rate is:

```text
missing_rate(f) = (# of missing observations in f) / (# of rows)
```

If `missing_rate(f)` exceeds `missing_threshold`, the factor can be dropped.

#### Cross-Sectional Size Filter

For each date `t`, let `N_t` be the number of names in the cross section. The date is kept only if:

```text
N_t >= min_cross_section_size
```

#### Winsorization By Date

For each date `t` and variable `x`:

```text
lo_t = Quantile_t(x, q = winsor_lower)
hi_t = Quantile_t(x, q = winsor_upper)
x_winsor,t = min(max(x_t, lo_t), hi_t)
```

This is done independently for each date, which preserves cross-sectional comparability while reducing outlier influence.

#### Cross-Sectional Z-Score By Date

For each date `t` and signal `x`:

```text
z_{i,t} = (x_{i,t} - mean_t(x)) / std_t(x)
```

The implementation uses population standard deviation with `ddof=0`.

If the date-level standard deviation is zero, the z-score is set to missing.

## 6. Target Construction

`build_targets()` creates forward-looking targets from the return column, usually `ret`.

Suppose daily return is `r_{i,t}` and horizon is `h`.

### 6.1 Forward Signed Return

If `return_type = "simple"`:

```text
ret_fwd_{h}d(i,t) = prod_{k=1..h} (1 + r_{i,t+k}) - 1
```

If `return_type = "log"`:

```text
ret_fwd_{h}d(i,t) = exp(sum_{k=1..h} log(1 + r_{i,t+k})) - 1
```

The window begins at `t+1`, so the target is forward-looking.

### 6.2 Absolute Return

```text
abs_ret_fwd_{h}d = |ret_fwd_{h}d|
```

### 6.3 Squared Return

```text
sq_ret_fwd_{h}d = (ret_fwd_{h}d)^2
```

### 6.4 Forward Realized Volatility

The code uses root-mean-square future return magnitude:

```text
rv_fwd_{h}d(i,t) = sqrt( (1/h) * sum_{k=1..h} r_{i,t+k}^2 )
```

This is not a de-meaned standard deviation. It is a horizon-level realized volatility proxy based on future return energy.

### 6.5 Forward Downside Semivariance

Let:

```text
r^-_{i,t+k} = min(r_{i,t+k}, 0)
```

Then:

```text
downside_semivar_fwd_{h}d(i,t) = (1/h) * sum_{k=1..h} (r^-_{i,t+k})^2
```

### 6.6 Tail Indicators

After a forward return column is built, the code creates binary indicators:

- `tail_left`
- `tail_right`
- `tail_abs`

Let `y` be the forward return column and let `q_L`, `q_U` be lower and upper tail thresholds.

Definitions:

```text
tail_left  = 1[y <= q_L]
tail_right = 1[y >= q_U]
tail_abs   = 1[y <= q_L or y >= q_U]
```

The thresholds can be computed:

- on the full sample
- by date
- by stock

So if `tail_groupby = "by_date"`, for example:

```text
q_L(t) = Quantile(y_{.,t}, lower_q)
q_U(t) = Quantile(y_{.,t}, upper_q)
```

and the binary indicator is evaluated relative to that date-specific threshold.

## 7. GEX Regime Construction

`build_gex_regimes()` defines regime labels from the GEX signal.

### 7.1 Sign-Based Regime

```text
neg_gex_flag = 1[ GEX < 0 ]
pos_gex_flag = 1[ GEX >= 0 ]
gex_sign_regime = "neg" if GEX < 0 else "pos"
```

### 7.2 Quantile Regime

The code bucketizes either the z-scored GEX signal or raw GEX by date:

```text
gex_q = qcut_by_date(GEX or GEX_z, n_buckets)
```

### 7.3 Extreme Negative Regime Proxy

The most negative GEX bucket is flagged as:

```text
extreme_neg_gex_flag = 1[ gex_q = 1 ]
```

This acts as a simple fragility regime proxy.

## 8. Phase 2: GEX Validation And Interaction Regressions

`run_phase2()` contains two modules.

## 8.1 Regime Validation

`run_regime_validation()` asks whether GEX behaves more like a risk or fragility state variable than a directional alpha signal.

For each outcome `Y`:

1. Sort names into GEX buckets by date.
2. Compute bucket-level average outcome.
3. Run Fama-MacBeth regressions of `Y` on GEX and controls.

### Bucket Sort Table

For each date `t`, names are assigned to GEX quantile buckets. Then the date-level bucket mean is:

```text
mu_{b,t} = mean( Y_{i,t} | bucket_{i,t} = b )
```

The final table reports time-series averages by bucket:

```text
mean_b = mean_t(mu_{b,t})
std_b  = std_t(mu_{b,t})
```

The reported spread is:

```text
top_minus_bottom_t = mu_{B,t} - mu_{1,t}
```

where `B = n_buckets`.

### Fama-MacBeth Regression

For each date `t`, the code runs a cross-sectional OLS:

```text
Y_{i,t} = a_t + beta_t * GEX_{i,t} + gamma_t' Controls_{i,t} + eps_{i,t}
```

or uses `GEX_z` if available.

Then it averages the date-level slopes:

```text
beta_bar = (1/T) * sum_{t=1..T} beta_t
```

To assess significance, the code runs a mean-only regression on the time series of daily slopes and computes a Newey-West HAC standard error:

```text
t(beta_bar) = beta_bar / SE_HAC(beta_t)
```

The reported p-value is based on a Student-t approximation.

## 8.2 Interaction Regression

`run_interaction_regression()` tests whether factor efficacy depends on the GEX regime.

For each factor `F` and outcome `Y`, the model is:

```text
Y_{i,t} = a_t + b_t * GEX_z{i,t} + c_t * F_z{i,t} + d_t * (GEX_z{i,t} * F_z{i,t}) + gamma_t' Controls_{i,t} + eps_{i,t}
```

The main coefficient of interest is:

```text
d_bar = mean_t(d_t)
```

Interpretation:

- `d_bar > 0`: the factor effect strengthens as GEX increases.
- `d_bar < 0`: the factor effect strengthens when GEX is lower or more negative.
- large `|t(d_bar)|`: stronger evidence of a regime-dependent factor slope.

The output summary includes:

- average GEX slope
- average factor slope
- average interaction slope
- `t_interaction`
- `p_interaction`
- average cross-sectional `R^2`

## 9. Phase 3: Regime-Split Factor Tests And Double Sorts

`run_phase3()` has two parts.

## 9.1 Regime-Split Factor Test

`run_regime_split_factor_test()` evaluates each factor separately inside:

- negative GEX regime
- positive GEX regime

For a factor signal `S`, the code creates date-level factor buckets and computes a daily long-short spread:

```text
spread_t = mean(Y_{i,t} | S bucket = top) - mean(Y_{i,t} | S bucket = bottom)
```

### Performance Metrics

For the spread time series `spread_t`, the code computes:

- Mean spread return:

```text
mean_spread_ret = mean_t(spread_t)
```

- Spread volatility:

```text
spread_vol = std_t(spread_t)
```

- Spread Sharpe:

```text
spread_sharpe = mean_t(spread_t) / std_t(spread_t)
```

- Tail frequency:

```text
tail_freq = mean_t( 1[spread_t < 0] )
```

- Max drawdown:

Let cumulative curve be:

```text
V_t = prod_{s <= t} (1 + spread_s)
```

Then drawdown is:

```text
DD_t = V_t / max_{u <= t}(V_u) - 1
max_drawdown = min_t(DD_t)
```

### Information Coefficient

For each date `t`, the code computes:

- Pearson IC:

```text
IC_t = Corr_Pearson(S_{.,t}, Y_{.,t})
```

- Spearman Rank IC:

```text
RankIC_t = Corr_Spearman(S_{.,t}, Y_{.,t})
```

It then reports:

```text
ic_mean = mean_t(IC_t)
rank_ic_mean = mean_t(RankIC_t)
```

The regime comparison table reports:

```text
difference = metric_neg_gex - metric_pos_gex
```

## 9.2 Double-Sort Analysis

`run_double_sort_analysis()` runs two sequential sorts:

- A: factor first, then GEX
- B: GEX first, then factor

For each date:

1. Bucket on the first signal.
2. Within each first-bucket subgroup, bucket on the second signal.
3. Compute the cell mean of `Y`.

If the first bucket is `a` and the second bucket is `b`, the cell mean is:

```text
cell_mean(a,b) = mean_t mean(Y_{i,t} | first_bucket=a, second_bucket=b)
```

### Spread-In-Spread

For the highest and lowest first-signal buckets:

```text
high_mean = mean_t( top_second - bottom_second | first = high )
low_mean  = mean_t( top_second - bottom_second | first = low )
```

Then:

```text
spread_in_spread = high_mean - low_mean
```

This tests whether the second signal matters more when the first signal is already extreme.

## 10. Phase 4: Tail Classification With Logistic Models

`run_phase4()` calls `run_tail_classification()`.

For each target and factor, the code builds feature sets:

- `factor_only`
- `gex_only`
- `factor_plus_gex`
- `factor_plus_gex_plus_interaction`

The interaction feature is:

```text
interaction = factor_signal * gex_signal
```

### Train/Test Split

The split is date-based, not random, to avoid leakage across time.

If explicit ranges are provided:

```text
train = [train_start, train_end]
test  = [test_start, test_end]
```

Otherwise the code uses a chronological fraction split on unique dates.

### Models

The phase currently supports:

- logistic regression with L2-type setup
- elastic-net logistic regression

The pipeline uses:

- median imputation by default
- feature standardization
- class weighting

### Logistic Model Form

The conceptual probability model is:

```text
P(Y_{i,t}=1 | X_{i,t}) = sigma( alpha + X'_{i,t} beta )
```

where:

```text
sigma(z) = 1 / (1 + e^{-z})
```

### Classification Metrics

For predicted probabilities `p_i` and thresholded labels `yhat_i = 1[p_i >= 0.5]`, the code reports:

- Accuracy:

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- Precision:

```text
precision = TP / (TP + FP)
```

- Recall:

```text
recall = TP / (TP + FN)
```

- F1:

```text
F1 = 2 * precision * recall / (precision + recall)
```

- Brier score:

```text
brier = mean_i( (y_i - p_i)^2 )
```

- ROC AUC:

Area under the ROC curve computed from predicted probabilities.

Outputs include per-specification score tables, prediction tables, ROC plots, and top-AUC bar charts.

## 11. Phase 5: Nonlinear Tail Classification With Random Forest

`run_phase5_random_forest()` repeats the Phase 4 idea with nonlinear tree ensembles.

For each:

- target
- factor
- feature set

the code:

1. builds the feature matrix
2. applies an explicit date split
3. trains a `RandomForestClassifier`
4. scores train and test sets
5. stores feature importance

### Feature Sets

The same four variants are used:

- `factor_only`
- `gex_only`
- `factor_plus_gex`
- `factor_plus_gex_plus_interaction`

with:

```text
interaction = factor_signal * gex_signal
```

### Probability Output

The model produces:

```text
pred_prob = P_hat(Y = 1 | X)
```

These probabilities later feed Phase 6 fragility scaling.

### Feature Importance

The code records the model's built-in random forest feature importance values:

```text
importance_j
```

These come directly from `model.feature_importances_`.

The same binary classification metrics from Phase 4 are reported on train and test sets.

## 12. Phase 6: Portfolio Overlay Backtest

`run_phase6_portfolio_overlay()` converts factor signals into long-short portfolios and then scales those portfolios using GEX-based or model-based overlays.

### 12.1 Base Portfolio Construction

For each date, the factor signal is bucketed into `n_buckets`. The portfolio goes:

- long top bucket
- short bottom bucket

depending on configuration.

#### Equal Weighting

If equal-weighted:

```text
w_i^L = 1 / N_long
w_i^S = 1 / N_short
```

and short weights are assigned negative sign:

```text
w_i = +w_i^L  for longs
w_i = -w_i^S  for shorts
```

#### Value Weighting

If value-weighted using `weight_col`:

```text
w_i^L = cap_i / sum_{j in long} cap_j
w_i^S = cap_i / sum_{j in short} cap_j
```

with short-side portfolio weights applied as negative values.

#### Daily Portfolio Return

For each date:

```text
portfolio_ret_t = sum_i w_{i,t} * ret_{i,t}
```

The code also tracks gross exposure:

```text
gross_exposure_t = sum_i |w_{i,t}|
```

### 12.2 GEX Sign Overlay

The code computes daily aggregate GEX statistics:

- cross-sectional median GEX
- cross-sectional mean GEX
- share of names with negative GEX
- share in extreme negative GEX bucket

The sign overlay scale is:

```text
scale_gex_sign,t =
  neg_gex_scale, if median_cross_sectional_GEX_t < 0
  1.0, otherwise
```

Then:

```text
portfolio_ret_scaled,t = portfolio_ret_t * scale_gex_sign,t
```

### 12.3 GEX Quantile Overlay

The quantile overlay is more aggressive when negative GEX breadth is high:

```text
scale_gex_quantile,t =
  extreme_neg_gex_scale, if neg_gex_share_t >= 0.5
  1.0, otherwise
```

Again:

```text
portfolio_ret_scaled,t = portfolio_ret_t * scale_t
```

### 12.4 Phase 5 Probability Overlay

For each factor, the code selects Phase 5 predictions for a chosen target and preferred feature set, then aggregates predicted fragility probability by date:

```text
mean_pred_prob_t = mean_i(pred_prob_{i,t})
```

The exposure scale is:

```text
scale_phase5_prob,t = clip(1 - prob_scale_multiplier * mean_pred_prob_t, min_scale, max_scale)
```

So higher predicted fragility leads to smaller exposure.

### 12.5 Approximate Overlay Turnover And Cost

The code estimates overlay turnover from changes in scale:

```text
overlay_turnover_t = |scale_t - scale_{t-1}|
```

If transaction cost is `tc_bps`, then:

```text
tc_rate = tc_bps / 10000
overlay_cost_t = overlay_turnover_t * tc_rate
```

Net return is:

```text
portfolio_ret_net,t = portfolio_ret_scaled,t - overlay_cost_t
```

This is explicitly an overlay-level approximation, not a full holdings-based turnover model.

### 12.6 Portfolio Performance Metrics

For a daily return series `r_t`, the code computes:

- Mean daily return:

```text
mean_daily_ret = mean_t(r_t)
```

- Annualized return:

```text
ann_ret = (1 + mean_daily_ret)^252 - 1
```

- Annualized volatility:

```text
ann_vol = std(r_t) * sqrt(252)
```

- Sharpe ratio:

```text
sharpe = (mean_daily_ret / std(r_t)) * sqrt(252)
```

- Downside deviation:

Let `r_t^-` be returns where `r_t < 0`. Then:

```text
downside_dev = std(r_t^-) * sqrt(252)
```

- Sortino ratio:

```text
sortino = (mean_daily_ret * 252) / downside_dev
```

- Max drawdown:

```text
equity_t = prod_{s <= t}(1 + r_s)
DD_t = equity_t / max_{u <= t}(equity_u) - 1
max_drawdown = min_t(DD_t)
```

- Expected shortfall at 5%:

```text
VaR_5 = Quantile(r_t, 0.05)
ES_5  = mean(r_t | r_t <= VaR_5)
```

- Hit rate:

```text
hit_rate = mean_t(1[r_t > 0])
```

Outputs include summary tables, full time series, and date-level scaling records.

## 13. Phase 7: Curated Multivariate Model Training

`run_phase7_multivariate_model_training()` trains multi-feature binary classifiers using a curated factor list plus GEX and optional interactions.

### Feature Construction

The feature set may include:

- z-scored GEX
- selected z-scored factor columns
- factor-by-GEX interactions

For each factor `F`:

```text
interaction_F = F * GEX
```

### Models

The phase can run:

- elastic-net logistic regression
- histogram gradient boosting classifier

### Evaluation

The same test metrics are reported:

- AUC
- accuracy
- precision
- recall
- F1
- Brier score

### Permutation Importance

If enabled, the code computes test-set permutation importance for the best model:

```text
importance_j = performance(X_test, y_test) - performance(X_test with feature j permuted, y_test)
```

averaged across repeated shuffles.

The implementation stores:

- `importance_mean`
- `importance_std`

for each feature.

## 14. Outputs And Artifacts

The experiment writes outputs into `output_dir`.

Common outputs include:

- `phase1_panel.parquet`
- `phase1_metadata.json`
- Phase 2 CSV tables and plots
- Phase 3 regime-split and double-sort summaries
- Phase 4 classification score tables and ROC/bar plots
- `phase5_rf_scores_all.csv`
- `phase5_rf_feature_importance_all.csv`
- `phase5_rf_predictions_all.csv` when enabled
- `phase6_overlay_summary.csv`
- `phase6_overlay_timeseries.csv`
- `phase6_overlay_scaling_by_date.csv`
- `phase7_multivariate_scores.csv`
- `phase7_multivariate_predictions.csv`
- `phase7_multivariate_permutation_importance.csv`

Results are also stored in `self.artifacts` for immediate in-memory access.

## 15. Minimal Usage Pattern

Typical usage is:

```python
exp = GEXCollaborativeEffectExperiment(
    data_config=data_config,
    column_config=column_config,
    preprocess_config=preprocess_config,
    target_config=target_config,
    regression_config=regression_config,
    classification_config=classification_config,
    output_config=output_config,
    identifier_config=identifier_config,
    phase5_rf_config=phase5_rf_config,
    phase6_overlay_config=phase6_overlay_config,
    phase7_multivariate_config=phase7_multivariate_config,
)

exp.run_phase1(factor_specs=factor_specs, selected_factors=selected_factors)
exp.run_phase2()
exp.run_phase3()
exp.run_phase4()
exp.run_phase5_random_forest()
exp.run_phase6_portfolio_overlay()
exp.run_phase7_multivariate_model_training()
```

You can also run later phases selectively after `run_phase1()` if the panel already contains the required derived columns.

## 16. Interpretation Notes

This codebase is strongest when used to answer questions like:

- Does GEX behave like a cross-sectional risk regime variable?
- Are some factors more effective in negative GEX states?
- Are tail events more predictable when factor and GEX signals interact?
- Does reducing exposure in fragile regimes improve factor portfolio performance?

Important interpretation caveats:

- Forward targets depend on future returns and therefore must never leak into feature construction.
- Monthly merges rely on month-key alignment plus optional `lag_months`; users should confirm this timing matches their economic intent.
- The Phase 6 turnover model only captures changes in overlay scale, not security-level rebalancing cost.
- Realized volatility is a root-mean-square return measure, not a de-meaned realized standard deviation.
- Statistical significance in Fama-MacBeth summaries depends on enough usable dates and sufficiently large cross sections.

## 17. Recommended Reading Order In Code

If you want to inspect the implementation itself, read in this order:

1. config dataclasses
2. `run_phase1()`
3. `load_base_panel()`
4. `load_factor_source()` and `merge_factors()`
5. `preprocess_panel()`
6. `build_targets()`
7. `build_gex_regimes()`
8. `run_phase2()` and `run_fama_macbeth()`
9. `run_phase3()`
10. `run_phase4()`
11. `run_phase5_random_forest()`
12. `run_phase6_portfolio_overlay()`
13. `run_phase7_multivariate_model_training()`

## 18. Summary

`factor_analysis_v0.py` is not just a factor tester. It is a full research pipeline that:

- builds a stock-level panel around GEX and CRSP
- standardizes and enriches factor data
- creates forward return and tail-risk targets
- tests factor behavior conditionally on GEX regimes
- trains both simple and nonlinear predictive models
- translates those predictions into portfolio exposure overlays

That makes it useful both for empirical factor diagnostics and for turning regime information into an implementable portfolio scaling framework.
