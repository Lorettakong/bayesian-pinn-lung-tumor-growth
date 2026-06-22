NLSTt mechanistic correction manuscript outputs

This folder contains manuscript-ready outputs generated from the leakage-free
NLSTt-300 evaluation pipeline.

Main files:
- table1_ablation_rmse_mae_for_paper.tex/csv: ablation of correction features.
- table2_methods_with_stats_for_paper.tex/csv: baseline comparison table.
- table_effect_size_paired_tests.tex/csv: paired Wilcoxon tests and effect sizes.
- table_hard_case_subgroup_for_paper.csv: challenging-case subgroup results.
- table_lambda_sensitivity.tex/csv: residual shrinkage sensitivity.
- fig_case_examples_log.png/pdf: illustrative log-volume single-trajectory examples.
- fig_gompertz_residual_diagnostics_refined.png/pdf: residual-structure diagnostics.
- fig_lambda_sensitivity.png/pdf: shrinkage sensitivity figure.
- fig_calibration_interval_score.png/pdf: coverage and interval score after conformal calibration.

Important leakage note:
The proposed residual learner uses only features available before observing the held-out third CT scan:
baseline/follow-up volumes, first-interval growth descriptors, and Bayesian Gompertz predictions/uncertainty
computed from the first two scans. Held-out third-scan observations are used only as training targets
inside the appropriate training folds and for final evaluation.
