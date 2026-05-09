# Bayesian PINN for Lung Tumor Growth Modeling

This repository is a cleaned public submission version of the codebase used for Bayesian physics-informed neural network experiments on longitudinal lung tumor growth data.

## What is included

- Core training, inference, evaluation, and plotting scripts used in the manuscript
- Processed modeling input table in `data/volumetric_data.csv`
- Manuscript-facing result tables and figures in `outputs_validation/`
- Supporting method notes in `docs/`
- Preprocessing utilities in `preprocessing/`

## What was intentionally excluded

The original working directory contained local IDE files, a virtual environment, Python caches, heavy intermediate outputs, model checkpoints, and early-generation artifacts. Those files were omitted from this public version so the repository stays readable and GitHub-friendly.

Excluded categories include:

- `.venv/`, `.idea/`, `__pycache__/`, `.DS_Store`
- patient-by-patient intermediate output folders such as `outputs/`, `outputs_hmc/`, `outputs_bpinn/`, `outputs_uq/`, `outputs_nls/`, and related checkpoint directories
- local-path metadata tables that were only used during earlier preprocessing

## Repository layout

- `Train_MAP.py`: deterministic MAP PINN training
- `Train_PINN_Deterministic.py`: deterministic PINN baseline
- `Train_PINN_Bayesian.py`: Bayesian PINN ablation without Gompertz physics
- `Train_BayesPINN_VI.py`: variational Bayesian PINN experiments
- `baseline_gompertz_nls.py`: classical Gompertz NLS baseline
- `HMC.py`: HMC inference for low-dimensional Bayesian Gompertz parameters
- `uq_predict.py`: posterior predictive summaries
- `Train_GP_Baselines.py`: Gaussian-process baselines
- `evaluate_holdout.py`: holdout evaluation and paired statistical tests
- `build_cohort_summary.py`: cohort summary exports and manuscript plots
- `plot_*.py`: figure-generation scripts
- `data/`: processed modeling inputs
- `outputs_validation/`: manuscript-facing tables and figures
- `docs/`: method notes
- `preprocessing/`: legacy preprocessing utilities kept for provenance

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Main packages:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `scipy`
- `pydicom`
- `pynrrd`

## Notes on reproducibility

The public cleanup preserves the original code logic, default hyperparameters, output filenames, and manuscript result files that were already generated in the source project.

Some scripts still use the original absolute default path `'/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'` in their configuration blocks. For a fresh environment, update `csv_path` to point to your local copy of the modeling table before rerunning the pipeline.
