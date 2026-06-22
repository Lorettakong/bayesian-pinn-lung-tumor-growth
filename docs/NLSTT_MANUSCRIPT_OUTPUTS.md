# NLSTt Manuscript Outputs

This document summarizes the output folders added for the revised
mechanistic misspecification correction manuscript.

## Main Paper Outputs

Folder:

```text
outputs_nlstt_validation/paper_outputs_population_residual/
```

This folder contains manuscript-ready figures, tables, and supporting CSV files
for the NLSTt-300 evaluation cohort.

Key outputs:

- `fig_case_examples_log.pdf/.png`  
  Log-volume single-trajectory examples with held-out error annotations.

- `fig_gompertz_residual_diagnostics_refined.pdf/.png`  
  Residual-structure diagnostics supporting the mechanistic deviation
  correction module.

- `fig_lambda_sensitivity.pdf/.png`  
  Sensitivity of held-out prediction accuracy to the residual shrinkage
  coefficient.

- `table1_ablation_rmse_mae_for_paper.csv/.tex`  
  Ablation of correction features.

- `table2_methods_with_stats_for_paper.csv/.tex`  
  Main baseline comparison table.

- `table_effect_size_paired_tests.csv/.tex`  
  Paired Wilcoxon tests and rank-biserial effect sizes.

- `table_hard_case_subgroup_for_paper.csv`  
  Pre-defined challenging-case subgroup results.

- `table_lambda_sensitivity.csv/.tex`  
  Shrinkage coefficient sensitivity table.

## Supplemental Robustness Outputs

Folder:

```text
outputs_nlstt_8603_map_robustness/
```

This folder contains the MAP-based robustness experiment on all 8603 eligible
three-scan NLSTt trajectories. It is included to support the representativeness
of the stratified NLSTt-300 primary evaluation cohort.

Key outputs:

- `nlstt_8603_map_summary.csv`
- `nlstt_8603_map_patient_level.csv`
- `nlstt_8603_map_subgroup_summary.csv`
- `nlstt_8603_map_paired_stats.csv`

## Baseline Outputs

Folders:

```text
outputs_naive_baselines/
outputs_population_baselines_300/
outputs_enhanced_baseline_stats/
```

These folders contain simple longitudinal extrapolation, Gaussian process,
mechanistic, and neural baseline comparisons used in the revised results
section.
