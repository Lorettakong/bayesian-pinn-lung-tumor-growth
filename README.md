# Uncertainty-Aware Mechanistic Misspecification Correction for Sparse Longitudinal CT-Derived Lung Nodule Trajectory Prediction

This repository contains the code, processed tables, and manuscript-facing
outputs for the study.

The study evaluates sparse three-scan lung nodule trajectory prediction using
CT-derived measurements from the NLSTt dataset. The proposed framework uses a
Bayesian Gompertz model as an interpretable mechanistic backbone, learns a
regularized population-level mechanistic deviation correction from pre-holdout
trajectory features, and applies conformal calibration to improve predictive
interval reliability.

## What Is Included

- NLSTt three-scan cohort construction utilities and processed modeling tables
- Bayesian Gompertz-guided mechanistic deviation correction code
- Simple longitudinal, data-driven, mechanistic, and residual-neural baseline
  comparison outputs
- Conformal calibration and uncertainty evaluation outputs
- Manuscript-facing figures and tables for the BSPC submission
- MAP-based robustness outputs on all 8603 eligible three-scan NLSTt
  trajectories

## Repository Layout

- `build_nlstt_volume_dataset.py`  
  Builds the three-scan NLSTt nodule-volume table from the NLSTt registration
  coordinate file using an ellipsoidal CT-derived volume approximation.

- `nlstt_population_residual_pipeline.py`  
  Main grouped cross-validation pipeline for Bayesian Gompertz-guided
  mechanistic deviation correction.

- `generate_nlstt_population_residual_paper_outputs.py`  
  Exports manuscript-ready tables and figures from the NLSTt-300 evaluation.

- `generate_additional_results_effect_error_lambda.py`  
  Generates residual diagnostics, effect-size tables, error summaries, and
  shrinkage-sensitivity outputs used in the manuscript.

- `compute_naive_baselines.py`, `compute_population_baselines_300.py`,
  `compute_enhanced_baseline_stats.py`  
  Baseline and paired-statistics utilities for simple longitudinal
  extrapolation, Gaussian process regression, NLS Gompertz, and residual-neural
  baselines.

- `run_nlstt_8603_map_robustness.py`  
  MAP-based robustness experiment on all 8603 eligible NLSTt three-scan
  trajectories.

- `plot_gompertz_residual_diagnostics.py`  
  Generates the refined residual-structure diagnostic figure used in the
  Results section.

- `data/nlstt_processed/`  
  Processed NLSTt three-scan tables and cohort-construction metadata.

- `outputs_nlstt_validation/paper_outputs_population_residual/`  
  Main manuscript-facing tables and figures.

- `outputs_nlstt_8603_map_robustness/`  
  Supplemental MAP-based robustness experiment on all eligible three-scan
  trajectories.

- `outputs_population_baselines_300/`,
  `outputs_naive_baselines/`,
  `outputs_enhanced_baseline_stats/`  
  Baseline prediction files and paired statistical summaries.

- `docs/NLSTT_MANUSCRIPT_OUTPUTS.md`  
  Description of the manuscript output files.

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

## Key Results

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
- `matplotlib`

## Running the Main Analyses

A typical workflow for reproducing the manuscript results is:

```bash
python build_nlstt_volume_dataset.py
python nlstt_population_residual_pipeline.py
python generate_nlstt_population_residual_paper_outputs.py
python generate_additional_results_effect_error_lambda.py
python compute_naive_baselines.py
python compute_population_baselines_300.py
python compute_enhanced_baseline_stats.py
python run_nlstt_8603_map_robustness.py
```

The first three scripts rebuild the processed three-scan cohort, run the main
NLSTt-300 grouped cross-validation experiment, and export the manuscript-ready
figures and tables. The additional baseline and robustness scripts reproduce
the supplementary comparison outputs used in the manuscript.

## Reproducibility Notes

The repository includes processed NLSTt-derived tables used for the manuscript
analyses. The CT-derived volumes are approximate ellipsoidal volumes computed
from NLSTt nodule extents, not pathologically confirmed malignant tumor
burden.

Raw NLST CT images are not included. The processed tables and manuscript-facing
outputs are provided so readers can inspect the cohort construction,
prediction results, statistical comparisons, and figures reported in the
paper.
