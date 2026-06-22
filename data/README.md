# Data

This repository contains two processed data groups.

## Legacy patient-level tumor-volume table

`volumetric_data.csv` is the compact patient-timepoint table used by the
earlier Bayesian PINN experiments, with columns such as:

- `patient_id`
- `t_rel`
- `V_obs`

Some legacy scripts still reference this table through their local
configuration blocks.

## NLSTt three-scan nodule-volume tables

`nlstt_processed/` contains the processed NLSTt-derived tables used by the
current mechanistic misspecification correction manuscript.

Main files:

- `nlstt_three_scan_volume_long_with_metadata.csv`  
  Long-format CT-derived nodule-volume table for the final stratified
  three-scan cohort.

- `nlstt_three_scan_trajectory_summary.csv`  
  Trajectory-level summary of the final stratified cohort.

- `nlstt_three_scan_all_eligible_summary.csv`  
  Trajectory-level summary for all eligible three-scan NLSTt trajectories.

- `nlstt_three_scan_selection_metadata.txt`  
  Cohort-construction counts and filtering notes.

The NLSTt volumes are approximate ellipsoidal volumes computed from nodule
extent measurements. They should be interpreted as CT-derived nodule or lesion
volume estimates rather than pathologically confirmed malignant tumor burden.
