# Proposed Method Design

This document defines the intended implementation of the paper's proposed method:

`Gompertz + PINN + Bayesian`

## 1. Method definition

The proposed method should contain three components at the same time:

1. A PINN backbone that predicts `y(t) = log V(t)` from time.
2. A Gompertz physics constraint through the residual
   `dy/dt = alpha - beta * y`.
3. A Bayesian layer that quantifies uncertainty for the proposed model itself,
   not only for a separate low-dimensional Gompertz model.

## 2. Practical inference strategy

Given the extremely small number of observations per patient, the practical
implementation should use a staged inference pipeline:

1. Train a deterministic MAP PINN on the training subset only.
2. Freeze the PINN backbone except for the final prediction layer.
3. Place Bayesian inference on:
   - the final layer parameters,
   - `alpha`,
   - `beta`,
   - an explicit initial-condition parameter if needed.
4. Draw posterior predictive samples from the resulting Bayesian head.

This keeps the method aligned with the paper while avoiding unstable
full-network Bayesian inference with only three observations per patient.

## 3. Required protocol fixes

Before the final proposed-method implementation is trusted, two protocol issues
must be fixed:

1. Persistent time normalization:
   all tensors (`t_data`, `t_phys`, `t_grid`, holdout time) must use the same
   normalization statistics derived from the training data.
2. True holdout evaluation:
   the final observation must be excluded from training and used only for
   evaluation.

## 4. Result files expected from the final method

For each patient, the final implementation should write:

- posterior predictive values at training and holdout times,
- posterior predictive intervals in log and volume spaces,
- a per-patient summary CSV,
- optional posterior samples for Bayesian head parameters.

At the cohort level, the pipeline should produce a distinct method line named:

`Bayesian PINN`

This line must remain separate from:

- `Gompertz + Bayesian`
- `Deterministic PINN (MAP)`
- `NLS Gompertz`

