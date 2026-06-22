#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_all_eligible_summary.csv"
OUT = ROOT / "outputs_nlstt_8603_map_robustness"


@dataclass(frozen=True)
class PriorConfig:
    sigma_d: float = 0.2
    mu_log_alpha: float = math.log(0.2)
    sd_log_alpha: float = 0.5
    mu_log_beta: float = math.log(0.05)
    sd_log_beta: float = 0.5
    y0_prior_sd: float = 0.5


def gompertz_y(t: np.ndarray, alpha: float, beta: float, y0: float, t0: float = 0.0) -> np.ndarray:
    beta = max(float(beta), 1e-12)
    y_inf = float(alpha) / beta
    return y_inf + (float(y0) - y_inf) * np.exp(-beta * (np.asarray(t, dtype=float) - t0))


def map_objective(z: np.ndarray, y_train: np.ndarray, prior: PriorConfig) -> float:
    log_alpha, log_beta, y0 = [float(v) for v in z]
    alpha = math.exp(log_alpha)
    beta = math.exp(log_beta)
    t_train = np.array([0.0, 1.0], dtype=float)
    y_hat = gompertz_y(t_train, alpha, beta, y0, 0.0)
    data = 0.5 * float(np.sum(((y_hat - y_train) / prior.sigma_d) ** 2))
    p_alpha = 0.5 * ((log_alpha - prior.mu_log_alpha) / prior.sd_log_alpha) ** 2
    p_beta = 0.5 * ((log_beta - prior.mu_log_beta) / prior.sd_log_beta) ** 2
    p_y0 = 0.5 * ((y0 - float(y_train[0])) / prior.y0_prior_sd) ** 2
    return data + p_alpha + p_beta + p_y0


def fit_map_row(row: pd.Series, prior: PriorConfig) -> dict:
    y0_obs = math.log(float(row["v0_cm3"]))
    y1_obs = math.log(float(row["v1_cm3"]))
    y2_obs = math.log(float(row["v2_cm3"]))
    y_train = np.array([y0_obs, y1_obs], dtype=float)
    x0 = np.array([prior.mu_log_alpha, prior.mu_log_beta, y0_obs], dtype=float)
    res = minimize(
        map_objective,
        x0=x0,
        args=(y_train, prior),
        method="L-BFGS-B",
        bounds=[
            (math.log(1e-8), math.log(10.0)),
            (math.log(1e-8), math.log(10.0)),
            (-20.0, 20.0),
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    log_alpha, log_beta, y0_hat = [float(v) for v in res.x]
    alpha = math.exp(log_alpha)
    beta = math.exp(log_beta)
    pred_y2 = float(gompertz_y(np.array([2.0]), alpha, beta, y0_hat, 0.0)[0])
    pred_v2 = math.exp(pred_y2)
    return {
        "trajectory_id": str(row["trajectory_id"]),
        "subject_id": str(row["patient_id"]),
        "growth_class": str(row["growth_class"]),
        "V0": float(row["v0_cm3"]),
        "V1": float(row["v1_cm3"]),
        "V2": float(row["v2_cm3"]),
        "logV0": y0_obs,
        "logV1": y1_obs,
        "holdout_y_obs": y2_obs,
        "holdout_V_obs": float(row["v2_cm3"]),
        "alpha_map": alpha,
        "beta_map": beta,
        "y0_map": y0_hat,
        "gompertz_map_pred_y": pred_y2,
        "gompertz_map_pred_V": pred_v2,
        "map_success": bool(res.success),
        "map_objective": float(res.fun),
    }


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x)))


def mae(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x)))


def make_outer_folds(groups: np.ndarray, k: int = 5) -> np.ndarray:
    unique = np.array(sorted(pd.unique(groups.astype(str))))
    group_to_fold = {g: i % k for i, g in enumerate(unique)}
    return np.array([group_to_fold[str(g)] for g in groups], dtype=int)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def poly2_expand(X: np.ndarray) -> np.ndarray:
    parts = [X]
    inter = []
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            inter.append((X[:, i] * X[:, j])[:, None])
    if inter:
        parts.append(np.hstack(inter))
    return np.hstack(parts)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]
    reg = np.eye(X1.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)


def ridge_predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ coef


def fit_predict_ridge(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    alpha: float,
    poly2: bool,
) -> np.ndarray:
    mu, sd = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, mu, sd)
    X_test = standardize_apply(X_test_raw, mu, sd)
    if poly2:
        X_train = poly2_expand(X_train)
        X_test = poly2_expand(X_test)
    coef = ridge_fit(X_train, y_train, alpha)
    return ridge_predict(X_test, coef)


def inner_folds(n: int, k: int = 4) -> np.ndarray:
    return np.arange(n) % k


def tune_ridge_and_lambda(X: np.ndarray, y_resid: np.ndarray, base_error: np.ndarray) -> dict:
    candidates = [(a, p) for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] for p in [False, True]]
    folds = inner_folds(len(X), 4)
    best: dict | None = None
    for alpha, poly2 in candidates:
        pred = np.zeros(len(X), dtype=float)
        ok = True
        for f in np.unique(folds):
            tr = folds != f
            va = folds == f
            try:
                pred[va] = fit_predict_ridge(X[tr], y_resid[tr], X[va], alpha, poly2)
            except np.linalg.LinAlgError:
                ok = False
                break
        if not ok:
            continue
        for lam in np.linspace(0.0, 1.5, 31):
            err = base_error + lam * pred
            loss = float(np.mean(err * err))
            if best is None or loss < best["loss"]:
                best = {"alpha": alpha, "poly2": poly2, "lambda": float(lam), "loss": loss}
    if best is None:
        raise RuntimeError("No valid ridge candidate")
    return best


def add_error_columns(df: pd.DataFrame, prefix: str, pred_y: np.ndarray) -> None:
    pred_v = np.exp(pred_y)
    df[f"{prefix}_pred_y"] = pred_y
    df[f"{prefix}_pred_V"] = pred_v
    df[f"{prefix}_error_y"] = pred_y - df["holdout_y_obs"].to_numpy(float)
    df[f"{prefix}_sq_error_y"] = df[f"{prefix}_error_y"] ** 2
    df[f"{prefix}_abs_error_y"] = np.abs(df[f"{prefix}_error_y"])
    df[f"{prefix}_error_V"] = pred_v - df["holdout_V_obs"].to_numpy(float)
    df[f"{prefix}_abs_error_V"] = np.abs(df[f"{prefix}_error_V"])
    df[f"{prefix}_sq_error_V"] = df[f"{prefix}_error_V"] ** 2


def summarize(df: pd.DataFrame, prefix: str, display: str) -> dict:
    return {
        "method": display,
        "n": len(df),
        "rmse_log": rmse(df[f"{prefix}_error_y"].to_numpy(float)),
        "mae_log": mae(df[f"{prefix}_error_y"].to_numpy(float)),
        "rmse_volume": rmse(df[f"{prefix}_error_V"].to_numpy(float)),
        "mae_volume": mae(df[f"{prefix}_error_V"].to_numpy(float)),
    }


def paired_stats(df: pd.DataFrame, proposed: str, baseline: str, metric: str) -> float:
    x = df[f"{proposed}_{metric}"].to_numpy(float)
    y = df[f"{baseline}_{metric}"].to_numpy(float)
    if np.allclose(x - y, 0.0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = PriorConfig()
    raw = pd.read_csv(SRC)
    rows = []
    for idx, row in raw.iterrows():
        if idx % 500 == 0:
            print(f"MAP fitting {idx}/{len(raw)}")
        rows.append(fit_map_row(row, prior))
    df = pd.DataFrame(rows)
    add_error_columns(df, "gompertz_map", df["gompertz_map_pred_y"].to_numpy(float))

    df["train_log_growth"] = df["logV1"] - df["logV0"]
    df["train_abs_log_growth"] = np.abs(df["train_log_growth"])
    df["train_volume_ratio"] = df["V1"] / df["V0"]
    df["baseline_volume"] = df["V0"]
    df["followup_volume"] = df["V1"]
    df["mean_train_volume"] = 0.5 * (df["V0"] + df["V1"])
    df["target_residual_y"] = df["holdout_y_obs"] - df["gompertz_map_pred_y"]

    features = [
        "logV0",
        "logV1",
        "train_log_growth",
        "train_abs_log_growth",
        "train_volume_ratio",
        "baseline_volume",
        "followup_volume",
        "mean_train_volume",
        "gompertz_map_pred_y",
        "gompertz_map_pred_V",
    ]
    X = df[features].to_numpy(float)
    y_resid = df["target_residual_y"].to_numpy(float)
    base_error = df["gompertz_map_error_y"].to_numpy(float)
    folds = make_outer_folds(df["subject_id"].to_numpy(str), 5)
    correction = np.zeros(len(df), dtype=float)
    fold_rows = []
    for f in np.unique(folds):
        train = folds != f
        test = folds == f
        best = tune_ridge_and_lambda(X[train], y_resid[train], base_error[train])
        pred = fit_predict_ridge(X[train], y_resid[train], X[test], best["alpha"], best["poly2"])
        correction[test] = best["lambda"] * pred
        fold_rows.append({"fold": int(f), "n_train": int(train.sum()), "n_test": int(test.sum()), **best})
        print(
            f"Fold {f}: alpha={best['alpha']}, poly2={best['poly2']}, "
            f"lambda={best['lambda']:.2f}, n_test={int(test.sum())}"
        )

    add_error_columns(df, "map_correction", df["gompertz_map_pred_y"].to_numpy(float) + correction)
    df["map_correction_delta_y"] = correction

    summary = pd.DataFrame(
        [
            summarize(df, "gompertz_map", "Bayesian Gompertz MAP"),
            summarize(df, "map_correction", "MAP-based mechanistic correction"),
        ]
    )
    p_log = paired_stats(df, "map_correction", "gompertz_map", "sq_error_y")
    p_vol = paired_stats(df, "map_correction", "gompertz_map", "abs_error_V")
    stats = pd.DataFrame(
        [
            {
                "comparison": "MAP-based mechanistic correction vs Bayesian Gompertz MAP",
                "p_log_sq_error": p_log,
                "p_volume_abs_error": p_vol,
            }
        ]
    )
    subgroup = []
    for name, sub in [("All", df), *[(k, v) for k, v in df.groupby("growth_class")]]:
        for prefix, display in [
            ("gompertz_map", "Bayesian Gompertz MAP"),
            ("map_correction", "MAP-based correction"),
        ]:
            row = summarize(sub, prefix, display)
            row["subset"] = name
            subgroup.append(row)
    subgroup = pd.DataFrame(subgroup)

    df.to_csv(OUT / "nlstt_8603_map_patient_level.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "nlstt_8603_map_fold_selection.csv", index=False)
    summary.to_csv(OUT / "nlstt_8603_map_summary.csv", index=False)
    stats.to_csv(OUT / "nlstt_8603_map_paired_stats.csv", index=False)
    subgroup.to_csv(OUT / "nlstt_8603_map_subgroup_summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED STATS")
    print(stats.to_string(index=False))
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
