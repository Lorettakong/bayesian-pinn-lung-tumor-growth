# Data

`volumetric_data.csv` is the processed modeling input table included with this public repository version.

This file is the compact patient-timepoint table used by the downstream modeling scripts, with columns such as:

- `patient_id`
- `t_rel`
- `V_obs`

The larger preprocessing metadata table from the original working directory was intentionally omitted because it contained local filesystem paths and was not needed for manuscript reproduction.
