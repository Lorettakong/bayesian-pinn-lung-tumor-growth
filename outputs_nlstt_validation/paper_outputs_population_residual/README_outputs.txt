NLSTt population residual paper outputs

This folder contains paper-ready outputs generated from the leakage-free pipeline.

Main files:
- table_main_population_residual.tex/csv: main performance table.
- table_feature_ablation_no_leakage.tex/csv: feature ablation and sanity evidence.
- table_paired_tests_vs_bayesian_gompertz.tex/csv: paired Wilcoxon tests.
- fig_method_comparison.png/pdf: cohort-level error comparison.
- fig_paired_improvement.png/pdf: patient-wise improvement over Bayesian Gompertz.
- fig_calibration_interval_score.png/pdf: coverage and interval score after conformal calibration.
- fig_subgroup_growth_class_rmse.png/pdf: subgroup RMSE by growth class.
- fig_case_examples.png/pdf: illustrative single-patient examples.
- fig_permutation_sanity_check.png/pdf: target permutation sanity check.

Important leakage note:
The proposed residual learner uses only features available before observing the held-out third CT scan:
baseline/follow-up volumes, first-interval growth descriptors, and Bayesian Gompertz predictions/uncertainty
computed from the first two scans. The previous holdout-normalized relative interval width was removed.
