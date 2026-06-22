# Mechanistic Misspecification Correction for CT Nodule Trajectories

This repository contains the code, processed tables, and manuscript-facing
outputs for the study:

**Uncertainty-Aware Mechanistic Misspecification Correction for Sparse
Longitudinal CT-Derived Lung Nodule Trajectory Prediction**

The current manuscript evaluates sparse three-scan lung nodule trajectory
prediction using CT-derived measurements from the NLSTt dataset. The proposed
framework uses a Bayesian Gompertz model as an interpretable mechanistic
backbone, learns a regularized population-level mechanistic deviation
correction from pre-holdout trajectory features, and applies conformal
calibration to improve predictive interval reliability.

## What Is Included

- NLSTt three-scan cohort construction utilities and processed modeling tables
- Bayesian Gompertz MAP/HMC fitting and posterior predictive utilities
- Regularized mechanistic deviation correction scripts
- Simple, data-driven, mechanistic, and neural baseline scripts
- Conformal calibration and uncertainty evaluation scripts
- Manuscript-facing figures and tables for the current BSPC submission
- MAP-based robustness outputs on all 8603 eligible three-scan NLSTt
  trajectories
- Legacy Bayesian PINN scripts retained for provenance and comparison

## Repository Layout

- `build_nlstt_volume_dataset.py`  
  Builds the three-scan NLSTt nodule-volume table from the NLSTt registration
  coordinate file using an ellipsoidal CT-derived volume approximation.

- `nlstt_population_residual_pipeline.py`  
  Main grouped cross-validation pipeline for Bayesian Gompertz-guided
  population-level mechanistic deviation correction.

- `generate_nlstt_population_residual_paper_outputs.py`  
  Exports manuscript-ready tables and figures from the NLSTt-300 evaluation.

- `generate_additional_results_effect_error_lambda.py`  
  Generates residual diagnostics, effect-size tables, error summaries, and
  shrinkage-sensitivity outputs used in the revised manuscript.

- `compute_naive_baselines.py`, `compute_population_baselines_300.py`,
  `compute_enhanced_baseline_stats.py`  
  Baseline and paired-statistics utilities for simple longitudinal
  extrapolation, Gaussian process regression, NLS Gompertz, and neural
  baselines.

- `run_nlstt_8603_map_robustness.py`  
  MAP-based robustness experiment on all 8603 eligible NLSTt three-scan
  trajectories.

- `plot_gompertz_residual_diagnostics.py`,
  `plot_gompertz_residual_structure.py`,
  `plot_prediction_error_ecdf.py`  
  Diagnostic plotting scripts for residual structure and prediction-error
  analyses.

- `data/nlstt_processed/`  
  Processed NLSTt three-scan tables and cohort-construction metadata.

- `outputs_nlstt_validation/`  
  Validation outputs for the NLSTt-300 cohort.

- `outputs_nlstt_validation/paper_outputs_population_residual/`  
  Main manuscript-facing tables and figures for the current paper.

- `outputs_nlstt_8603_map_robustness/`  
  Supplemental MAP-based robustness experiment on all eligible three-scan
  trajectories.

- `outputs_population_baselines_300/`,
  `outputs_naive_baselines/`,
  `outputs_enhanced_baseline_stats/`  
  Baseline prediction files and paired statistical summaries.

- `docs/`  
  Method notes, release notes, and manuscript-output descriptions.

## Main Manuscript Outputs

The primary ready-to-use tables and figures are in:

```text
outputs_nlstt_validation/paper_outputs_population_residual/
```

Important files include:

- `table1_ablation_rmse_mae_for_paper.csv/.tex`
- `table2_methods_with_stats_for_paper.csv/.tex`
- `table_effect_size_paired_tests.csv/.tex`
- `table_lambda_sensitivity.csv/.tex`
- `table_hard_case_subgroup_for_paper.csv`
- `fig_case_examples_log.pdf/.png`
- `fig_gompertz_residual_diagnostics_refined.pdf/.png`
- `fig_lambda_sensitivity.pdf/.png`
- `fig_calibration_interval_score.pdf/.png`

## Key Results Preserved in This Repository

On the stratified NLSTt-300 three-scan evaluation cohort, the Bayesian
Gompertz backbone achieved a log-volume RMSE of `0.5662` and a
volume-space MAE of `0.8551 cm^3`. The proposed mechanistic correction
framework achieved a log-volume RMSE of `0.5279` and a volume-space MAE
of `0.7636 cm^3`, corresponding to relative improvements of `6.8%` and
`10.7%`, respectively.

After conformal calibration, empirical 95% coverage improved from `0.600`
for the original Bayesian Gompertz intervals to `0.950` for the proposed
calibrated intervals, with interval score decreasing from `12.1614` to
`8.4727`.

The supplemental MAP-based robustness experiment on all `8603` eligible
three-scan trajectories is stored in:

```text
outputs_nlstt_8603_map_robustness/
```

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Main packages:

- `numpy`
- `pandas`
- `scipy`
- `torch`
- `matplotlib`
- `pydicom`
- `pynrrd`

## Reproducibility Notes

The public repository includes processed NLSTt-derived tables used for the
manuscript analyses. The CT-derived volumes are approximate ellipsoidal
volumes computed from NLSTt nodule extents, not pathologically confirmed
malignant tumor burden.

Some legacy scripts retained from the earlier Bayesian PINN manuscript still
use the original absolute default path
`/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv`. These scripts are kept
for provenance. For fresh local runs, update their `csv_path` configuration to
point to the desired modeling table.

The current NLSTt manuscript scripts added in this release use repository
relative paths where possible.

## Public Release Scope

This repository is intended to accompany manuscript review and public release.
Large intermediate model checkpoints, local virtual environments, IDE files,
and raw CT image data are intentionally excluded. The included outputs are the
compact manuscript-facing tables and figures needed to inspect and reproduce
the reported analyses.
