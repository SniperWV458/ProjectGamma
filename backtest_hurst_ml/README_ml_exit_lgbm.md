# ML Exit LGBM: Detailed Training and Evaluation Walkthrough

This document explains `backtest_hurst_ml/ml_exit_lgbm.py` in detail: what each section does, how data flows through the pipeline, and why each component exists.

---

## 1) What this script is for

`ml_exit_lgbm.py` trains a binary hazard model (LightGBM classifier) to decide whether a currently held position is likely to experience an adverse move soon.

- Target idea: at date `t`, for an active position, predict whether a sufficiently bad move occurs within the next `k` days.
- If predicted risk is high, the policy exits that position early.
- The script evaluates both:
  - standard classification quality (ROC-AUC, PR-AUC, calibration, etc.)
  - strategy-style proxy uplift (does exiting high-risk names improve forward returns vs keep-all?).

This script is standalone and does not automatically wire into `run_sentiment_gex.py`.

---

## 2) File structure at a glance

The file has five main layers:

1. Configuration and feature definitions (`CONFIG`, `LGBM_PARAMS`, feature lists)
2. Data preparation / feature engineering / labels (position feature builder, cross-sectional transforms, split logic)
3. Model training + policy tuning + split evaluation (fit model, select policy on validation, compute metrics)
4. Walk-forward inference rule class (`LGBMExitRule`) for backtest-time incremental retraining + exits
5. Main experiment driver (`main()`) orchestrates benchmark run, training data construction, experiments, outputs

---

## 3) Imports and dependency behavior

### Core dependencies

- `pandas`, `numpy` for data handling
- `scikit-learn` metrics + calibration utilities
- `lightgbm` imported lazily in `_fit_lgbm_classifier(...)`
- plotting via `matplotlib` if available (`Agg` backend for file output only)

### Optional behavior

If `matplotlib` is unavailable, the script still runs and writes CSV outputs; PNG plotting is skipped with a printed message.

If `lightgbm` is missing, `_fit_lgbm_classifier` raises an explicit install hint:
`pip install lightgbm`.

---

## 4) Global config and what each group controls

### `CONFIG` (experiment/runtime behavior)

#### Data and run paths

- `panel_path`: input panel parquet
- `start_date`: filter start for backtest/evaluation universe
- `run_name`, `output_root`: timestamped output directory naming

#### Label definition

- `hazard_horizon_days`: forward lookahead window `k`
- `adverse_return_threshold`: event threshold for hazard (for example `-0.5%`)

#### Time split

- `train_frac`, `val_frac`, `test_frac`
- `purge_days`: embargo gap inserted between split blocks

#### Exit policy tuning

- `policy_mode`: `"threshold"` or `"top_quantile"`
- `policy_threshold_default`, `policy_threshold_grid`
- `policy_quantile_default`, `policy_quantile_grid`
- `policy_sides`: which side(s) policy applies to (default short-only)

#### Strategy generation for training data breadth

- `benchmark_n_per_leg`: narrow baseline strategy size
- `training_n_per_leg`: larger candidate generation size
- `training_variants`, `training_gates`, `training_neutralities`: cross-product of strategy settings used to produce broader supervised training examples

#### Experiment extras

- `side_experiments`: optional side-specific ablations (`long`, `short`)
- early-stopping flags, random seed, minimum sample gate, plot dpi

### `LGBM_PARAMS` (model hyperparameters)

Contains objective, tree count, learning rate, regularization, tree constraints, and sampling controls used to instantiate `lgb.LGBMClassifier`.

---

## 5) Feature list design

Three feature groups are defined:

1. `BASE_FEATURE_COLS`  
   compact core signals/state (holding age, sentiment/GEX z-scores, return history, etc.)
2. `OPTIONAL_FEATURE_COLS`  
   richer optional diagnostics or options-derived columns used only if present in panel
3. `CS_FEATURE_SOURCE_COLS`  
   source columns used to derive same-date cross-sectional transforms (rank/z style), improving relative-signal expressiveness across names on each date

---

## 6) Utility functions and why they exist

### `_future_sum(series, horizon)`

Computes forward aggregated returns over a fixed horizon. Used to create `signed_fwd_k` style proxy outcomes.

### `add_cross_sectional_features(panel, source_cols)`

Adds per-date cross-sectional normalized features (for example z/rank-like transforms).  
Important: only same-date information is used (no lookahead), which preserves temporal validity.

### `_safe_auc`, `_safe_pr_auc`, `_safe_log_loss`

Defensive wrappers around sklearn metrics:

- avoid exceptions when only one class appears
- clip probabilities where needed
- return `NaN` gracefully in degenerate cases

### `chronological_split_with_purge(...)`

Strictly time-based split:

- partitions by unique dates
- inserts purge windows between train->val and val->test
- returns split DataFrames plus `meta` summary  
  This reduces leakage from near-adjacent observations/horizons.

---

## 7) Position-level dataset construction

### `build_lgbm_position_features(weights, panel, ...)`

This is the most important data-prep function.

It takes strategy weights and signal-enriched panel data and builds row-level supervised samples for active positions, including:

- identity fields: `permno`, `date`, side
- spell tracking: `_pos_id`, holding age (`days_held`)
- one-step signed return proxy (`signed_ret_t1`)
- forward-horizon signed return proxy (`signed_fwd_k`)
- engineered features (base + optional + cross-sectional)
- binary `hazard_label`

### Hazard label mechanics

For each row/date in a position spell:

- evaluate forward signed returns for horizons `1..k`
- set `hazard_label = 1` if any forward horizon breaches adverse threshold
- otherwise `0`

This is a discrete-time event-in-horizon formulation rather than exact event-day survival modeling.

---

## 8) Training data expansion logic

### `collect_training_candidate_weights(...)`

Runs many strategy variants to generate broader labeled training rows than a single baseline strategy would provide.

Process:

1. Loop over configured variant/gate/neutrality combinations
2. Run backtest to obtain weights
3. Concatenate and deduplicate by `(permno, date, side)` logic
4. Return merged candidate set

Why: increases sample count/diversity for classifier fitting and reduces overfitting to one narrow portfolio construction.

---

## 9) Pre-fit diagnostics

### `export_prefit_diagnostics(...)`

Writes sanity-check tables before model fitting:

- label counts by split and side
- label-rate over time (month)
- rows per date by split
- feature coverage after split

Use these to detect:

- class collapse
- severe drift
- sparse/missing feature problems
- unstable date-level sample density

---

## 10) Model fit summary and classifier fit

### `model_fit_summary(...)`

Collects compact fit metadata:

- number of features
- tree count / best iteration
- total gain
- prediction dispersion / uniqueness

Used to flag weak or near-constant models.

### `_fit_lgbm_classifier(...)`

Training procedure:

1. import LightGBM
2. copy base params + set random state
3. compute imbalance correction via `scale_pos_weight = n_neg / n_pos` when positives exist
4. build callbacks (`log_evaluation`, optional `early_stopping`)
5. fit with eval set and metrics `auc`, `binary_logloss`

---

## 11) Policy engine and uplift metrics

### `_eligible_policy_index(...)`

Filters rows eligible for policy action by side restrictions.

### `_policy_exit_mask(...)`

Converts risk predictions into binary exits:

- threshold mode: `proba >= threshold`
- top_quantile mode: per date, exit top `ceil(n * q)` risky rows

### `_same_count_baseline_returns(...)`

Builds same-exit-count comparators:

- oracle baseline: exits worst realized forward returns (best-case hindsight comparator)
- random baseline: exits random eligible names

### `_policy_proxy_metrics(...)`

Computes policy-level proxy performance:

- exit rate
- keep-all vs model-adjusted forward mean
- uplift vs keep-all
- oracle/random same-count references

This is the key strategy-like evaluation layer.

---

## 12) Validation tuning and split evaluation

### `tune_policy_on_validation(...)`

Grid-searches policy parameter on validation set:

- threshold grid or quantile grid depending on mode
- objective = maximize `proxy_uplift_vs_keep_all`
- returns selected policy dict + full sweep table

### `evaluate_split(...)`

For a given split:

1. predict hazard probabilities
2. compute classifier metrics at 0.50 decision threshold
3. apply selected policy and compute proxy metrics
4. produce calibration table (when both classes exist)
5. return metrics + prediction table + calibration table

---

## 13) Experiment runner

### `run_model_experiment(...)`

Runs one named experiment (`core`, `rich`, `short_rich`, etc.) end-to-end:

1. fit model on train (validated on val during fit)
2. tune policy on validation predictions
3. evaluate validation and test with selected policy
4. write experiment-prefixed CSV outputs:
   - metrics
   - policy sweep
   - calibration
   - predictions
   - prediction distribution
   - fit summary
   - feature importance gain
5. return all artifacts in-memory for orchestration

---

## 14) `LGBMExitRule`: walk-forward deployment-style rule

`LGBMExitRule(BaseEntryExitRule)` is a reusable exit rule for `run_backtest(...)`.

What it does:

- optionally enriches panel with optional features from external parquet
- optionally expands training weights via strategy combinations
- builds train/pred feature tables
- performs rolling walk-forward retraining:
  - trailing train window
  - retrain every `retrain_freq_days`
  - horizon embargo before prediction date
- predicts daily risk and applies policy exits
- removes rows after first exit per position spell

This class approximates a production-like retraining workflow rather than single static split training.

---

## 15) Plotting and reporting helpers

### `make_naive_proxy_nav(pred_df)`

Builds proxy NAV curves:

- no-exit baseline NAV
- model-policy NAV

### `plot_diagnostics(...)`

Writes standard visual diagnostics:

- ROC and PR (val/test)
- calibration
- policy sweep uplift curve
- naive proxy NAV

---

## 16) `main()` orchestration in exact order

1. clone config and create timestamped run folder
2. load panel and benchmark daily returns
3. run benchmark narrow strategy backtest
4. collect expanded training candidate weights
5. build feature tables for:
   - expanded training candidates
   - benchmark strategy weights (for later comparison)
6. export initial feature coverage
7. drop rows missing label/proxy-required fields
8. chronological split + purge
9. export pre-fit diagnostics
10. guardrails:
    - min train samples
    - both classes in train
11. run experiments:
    - `core`
    - `rich`
    - side-specific rich (if valid)
12. build `ablation_summary.csv`
13. choose `rich` as selected experiment for primary reporting
14. evaluate unchanged benchmark strategy using selected model/policy on test window
15. save all tables and plots
16. write `run_config.json` with chosen policy and params
17. print key test metrics and uplift summary

---

## 17) Output files and what they mean

### Split/diagnostic files

- `split_meta.csv`: date ranges and row counts per split
- `prefit_*`: class balance and temporal diagnostics
- `feature_coverage*.csv`: missingness/coverage snapshots

### Experiment files (`core_*`, `rich_*`, side-specific)

- `*_model_metrics_validation_test.csv`
- `*_validation_policy_sweep.csv`
- `*_calibration_validation_test.csv`
- `*_predictions_validation_test.csv`
- `*_prediction_distribution.csv`
- `*_model_fit_summary.csv`
- `*_feature_importance_gain.csv`

### Selected rich + benchmark comparisons

- `model_metrics_validation_test.csv`
- `validation_policy_sweep.csv`
- `predictions_validation_test.csv`
- `benchmark_strategy_test_metrics.csv`
- `benchmark_strategy_test_predictions.csv`
- `naive_proxy_nav_test.csv`
- `benchmark_strategy_proxy_nav_test.csv`
- `ablation_summary.csv`

### Plots

- `roc_pr_validation_test.png`
- `calibration_validation_test.png`
- `validation_policy_sweep.png`
- `naive_proxy_nav_test.png`

### Metadata

- `run_config.json`: config, model params, selected policy, selected experiment

---

## 18) Key assumptions and caveats

1. Objective mismatch  
   Policy selection uses proxy uplift metric, not a full transaction-cost-aware portfolio objective.
2. Training data mixture  
   Expanded candidate generation improves coverage but may shift distribution versus deployment strategy.
3. Class stability risk  
   Hazard event rate is sensitive to horizon and threshold; can collapse in some regimes.
4. Leakage controls are partial but intentional  
   Chronological split + purge helps; walk-forward rule also uses horizon embargo.
5. Selection policy hardcoded to rich experiment  
   Current script explicitly chooses `rich` for final reporting.

---

## 19) How to customize safely

Most impactful knobs:

- Labeling: `hazard_horizon_days`, `adverse_return_threshold`
- Leakage robustness: `purge_days`
- Policy aggressiveness: threshold/quantile grids and `policy_sides`
- Data breadth: training strategy combination lists
- Model complexity: `LGBM_PARAMS`

Recommended iteration pattern:

1. adjust one group of knobs
2. compare `ablation_summary.csv` + calibration + uplift curves
3. inspect benchmark comparison stability before adopting changes

---

## 20) Minimal run command

```bash
python backtest_hurst_ml/ml_exit_lgbm.py
```

Outputs appear in:
`backtest_hurst_ml/output_ml/ml_exit_lgbm/ml_exit_lgbm_<timestamp>/`
