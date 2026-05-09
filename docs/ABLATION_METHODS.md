# Ablation Method Definitions

## Gompertz + Bayesian

This ablation removes the PINN backbone and keeps only the low-dimensional
Bayesian Gompertz model.

- Forward model: closed-form Gompertz dynamics in log-volume space
- Parameters with posterior inference: `alpha`, `beta`, `y0`
- Inference: HMC after MAP-style initialization when available
- Code/results:
  - `HMC.py`
  - `uq_predict.py`
  - outputs in `outputs_hmc/` and `outputs_uq/`

## PINN + Bayesian

This ablation removes the Gompertz physics constraint and keeps only a Bayesian
neural network mapping time to `log V(t)`.

- Forward model: Bayesian MLP for `y(t) = log V(t)`
- Removed component: Gompertz residual / physics term
- Objective: data likelihood + initial-condition anchoring + variational KL
- Optional initialization: deterministic MAP PINN weights, used only as a warm start
- Code/results:
  - `Train_PINN_Bayesian.py`
  - outputs in `outputs_pinn_bayes/`

## Intended role relative to the proposed method

These two lines are ablations around the paper's proposed method.

- Proposed method: `Gompertz + PINN + Bayesian`
- Ablation 1: remove PINN -> `Gompertz + Bayesian`
- Ablation 2: remove Gompertz physics -> `PINN + Bayesian`
