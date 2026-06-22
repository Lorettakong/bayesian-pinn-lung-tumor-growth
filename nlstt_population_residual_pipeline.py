#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_nlstt_validation"
META = ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_volume_long_with_metadata.csv"
ALPHA = 0.05


BASE_FEATURES = [
    "logV0",
    "logV1",
    "train_log_growth",
    "train_abs_log_growth",
    "train_volume_ratio",
    "baseline_volume",
    "followup_volume",
    "mean_train_volume",
    "gompertz_pred_y",
    "gompertz_pred_V",
    "gompertz_pred_relative_interval_width",
    "gompertz_interval_width",
]


def rmse(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else np.nan


def mae(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if len(x) else np.nan


def interval_score(obs, lo, hi, alpha=ALPHA):
    if not np.isfinite(obs) or not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    width = hi - lo
    if obs < lo:
        return float(width + 2.0 / alpha * (lo - obs))
    if obs > hi:
        return float(width + 2.0 / alpha * (obs - hi))
    return float(width)


def conformal_quantile(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def load_frame():
    df = pd.read_csv(OUT / "holdout_results_all_patients.csv")
    meta = pd.read_csv(META)
    wide = meta.pivot(index="trajectory_id", columns="t_rel", values="V_obs_cm3").reset_index()
    wide.columns = ["patient_id", "V0", "V1", "V2"]
    cls = meta[["trajectory_id", "growth_class"]].drop_duplicates().rename(columns={"trajectory_id": "patient_id"})
    wide = wide.merge(cls, on="patient_id", how="left")
    df["patient_id"] = df["patient_id"].astype(str)
    wide["patient_id"] = wide["patient_id"].astype(str)
    df = df.merge(wide, on="patient_id", how="left")
    df["logV0"] = np.log(df["V0"])
    df["logV1"] = np.log(df["V1"])
    df["train_log_growth"] = np.log(df["V1"] / df["V0"])
    df["train_abs_log_growth"] = np.abs(df["train_log_growth"])
    df["train_volume_ratio"] = df["V1"] / df["V0"]
    df["baseline_volume"] = df["V0"]
    df["followup_volume"] = df["V1"]
    df["mean_train_volume"] = 0.5 * (df["V0"] + df["V1"])
    df["gompertz_pred_y"] = df["gompertz_bayes_pred_y"]
    df["gompertz_pred_V"] = df["gompertz_bayes_pred_V"]
    df["gompertz_pred_relative_interval_width"] = (
        df["gompertz_bayes_interval_width_V"] / np.maximum(df["gompertz_bayes_pred_V"].abs(), 1e-12)
    )
    df["gompertz_interval_width"] = df["gompertz_bayes_interval_width_V"]
    df["target_residual_y"] = df["holdout_y_obs"] - df["gompertz_bayes_pred_y"]
    df["subject_id"] = df["patient_id"].astype(str).str.split("_").str[0]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_outer_folds_from_groups(groups, k=5):
    # Deterministic subject-level folds. This prevents different nodules from
    # the same NLST subject appearing in both train and test folds.
    groups = np.asarray(groups, dtype=str)
    unique = np.array(sorted(pd.unique(groups)))
    group_to_fold = {g: i % k for i, g in enumerate(unique)}
    return np.array([group_to_fold[g] for g in groups], dtype=int)


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    Xs = (X - mu) / sd
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def poly2_expand(X):
    parts = [X]
    n = X.shape[1]
    inter = []
    for i in range(n):
        for j in range(i, n):
            inter.append((X[:, i] * X[:, j])[:, None])
    if inter:
        parts.append(np.hstack(inter))
    return np.hstack(parts)


def ridge_fit(X, y, alpha):
    X1 = np.c_[np.ones(len(X)), X]
    reg = np.eye(X1.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)


def ridge_predict(X, coef):
    X1 = np.c_[np.ones(len(X)), X]
    return X1 @ coef


def rbf_kernel(A, B, gamma):
    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]
    D = np.maximum(A2 + B2 - 2 * A @ B.T, 0.0)
    return np.exp(-float(gamma) * D)


def krr_fit(X, y, alpha, gamma):
    K = rbf_kernel(X, X, gamma)
    return np.linalg.solve(K + float(alpha) * np.eye(len(X)), y)


def krr_predict(X_train, X_test, dual, gamma):
    return rbf_kernel(X_test, X_train, gamma) @ dual


def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        d = np.sum((X_train - x) ** 2, axis=1)
        idx = np.argsort(d)[: int(k)]
        # Distance weighted average with a small floor.
        w = 1.0 / np.maximum(np.sqrt(d[idx]), 1e-6)
        preds.append(float(np.sum(w * y_train[idx]) / np.sum(w)))
    return np.asarray(preds)


def inner_folds(n, k=4):
    return np.arange(n) % k


@dataclass
class Candidate:
    name: str
    model: str
    params: dict


def candidate_grid():
    out = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        out.append(Candidate(f"ridge_a{alpha}", "ridge", {"alpha": alpha, "poly2": False}))
        out.append(Candidate(f"poly_ridge_a{alpha}", "ridge", {"alpha": alpha, "poly2": True}))
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for gamma in [0.01, 0.03, 0.1, 0.3, 1.0]:
            out.append(Candidate(f"krr_a{alpha}_g{gamma}", "krr", {"alpha": alpha, "gamma": gamma}))
    for k in [5, 10, 20, 40, 80]:
        out.append(Candidate(f"knn_k{k}", "knn", {"k": k}))
    return out


def fit_predict_candidate(cand, X_train_raw, y_train, X_test_raw):
    mu, sd = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, mu, sd)
    X_test = standardize_apply(X_test_raw, mu, sd)
    if cand.model == "ridge":
        if cand.params.get("poly2", False):
            X_train = poly2_expand(X_train)
            X_test = poly2_expand(X_test)
        coef = ridge_fit(X_train, y_train, cand.params["alpha"])
        return ridge_predict(X_test, coef), ridge_predict(X_train, coef)
    if cand.model == "krr":
        dual = krr_fit(X_train, y_train, cand.params["alpha"], cand.params["gamma"])
        return krr_predict(X_train, X_test, dual, cand.params["gamma"]), krr_predict(X_train, X_train, dual, cand.params["gamma"])
    if cand.model == "knn":
        return knn_predict(X_train, y_train, X_test, cand.params["k"]), knn_predict(X_train, y_train, X_train, cand.params["k"])
    raise ValueError(cand.model)


def tune_candidate_and_lambda(X_train, y_train, base_error_train):
    folds = inner_folds(len(X_train), 4)
    best = None
    for cand in candidate_grid():
        pred = np.zeros(len(X_train), dtype=float)
        ok = True
        for f in np.unique(folds):
            tr = folds != f
            va = folds == f
            try:
                pred[va], _ = fit_predict_candidate(cand, X_train[tr], y_train[tr], X_train[va])
            except np.linalg.LinAlgError:
                ok = False
                break
        if not ok:
            continue
        for lam in np.linspace(0.0, 1.5, 31):
            err = base_error_train + lam * pred
            loss = float(np.mean(err * err))
            if best is None or loss < best["loss"] - 1e-12:
                best = {"candidate": cand, "lambda": float(lam), "loss": loss}
    return best


def add_method_columns(df, prefix, pred_y):
    obs_y = df["holdout_y_obs"].to_numpy(float)
    obs_v = df["holdout_V_obs"].to_numpy(float)
    pred_v = np.exp(pred_y)
    df[f"{prefix}_pred_y"] = pred_y
    df[f"{prefix}_pred_V"] = pred_v
    df[f"{prefix}_error_y"] = pred_y - obs_y
    df[f"{prefix}_abs_error_y"] = np.abs(df[f"{prefix}_error_y"])
    df[f"{prefix}_sq_error_y"] = df[f"{prefix}_error_y"] ** 2
    df[f"{prefix}_error_V"] = pred_v - obs_v
    df[f"{prefix}_abs_error_V"] = np.abs(df[f"{prefix}_error_V"])
    df[f"{prefix}_sq_error_V"] = df[f"{prefix}_error_V"] ** 2


def add_conformal(df, prefix, source_prefix):
    pred = df[f"{source_prefix}_pred_V"].to_numpy(float)
    obs = df["holdout_V_obs"].to_numpy(float)
    lo0 = df["gompertz_bayes_interval_low_V"].to_numpy(float)
    hi0 = df["gompertz_bayes_interval_high_V"].to_numpy(float)
    base_radius = np.maximum(pred - lo0, hi0 - pred)
    abs_error = np.abs(obs - pred)
    excess = np.maximum(abs_error - base_radius, 0.0)
    q = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        train = np.ones(len(df), dtype=bool)
        train[i] = False
        q[i] = conformal_quantile(excess[train])
    radius = base_radius + q
    lo = np.maximum(pred - radius, 0.0)
    hi = pred + radius
    df[f"{prefix}_interval_low_V"] = lo
    df[f"{prefix}_interval_high_V"] = hi
    df[f"{prefix}_interval_width_V"] = hi - lo
    df[f"{prefix}_relative_interval_width_V"] = (hi - lo) / np.maximum(np.abs(obs), 1e-12)
    df[f"{prefix}_covered_95CI_V"] = ((obs >= lo) & (obs <= hi)).astype(int)
    df[f"{prefix}_interval_score_V"] = [interval_score(o, l, h) for o, l, h in zip(obs, lo, hi)]


def copy_gompertz_interval(df, prefix):
    for suf in ["interval_low_V", "interval_high_V", "interval_width_V", "relative_interval_width_V", "covered_95CI_V", "interval_score_V"]:
        df[f"{prefix}_{suf}"] = df[f"gompertz_bayes_{suf}"]


def summarize(df, prefix, display):
    return {
        "method": prefix,
        "method_display": display,
        "n": len(df),
        "rmse_log": rmse(df[f"{prefix}_error_y"]),
        "mae_log": mae(df[f"{prefix}_error_y"]),
        "rmse_volume": rmse(df[f"{prefix}_error_V"]),
        "mae_volume": mae(df[f"{prefix}_error_V"]),
        "coverage_95": float(pd.to_numeric(df.get(f"{prefix}_covered_95CI_V"), errors="coerce").mean()) if f"{prefix}_covered_95CI_V" in df else np.nan,
        "interval_score": float(pd.to_numeric(df.get(f"{prefix}_interval_score_V"), errors="coerce").mean()) if f"{prefix}_interval_score_V" in df else np.nan,
        "mean_relative_interval_width_V": float(pd.to_numeric(df.get(f"{prefix}_relative_interval_width_V"), errors="coerce").mean()) if f"{prefix}_relative_interval_width_V" in df else np.nan,
    }


def paired_p(df, a, b, metric):
    pair = df[[f"{a}_{metric}", f"{b}_{metric}"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2:
        return np.nan
    x = pair.iloc[:, 0].to_numpy(float)
    y = pair.iloc[:, 1].to_numpy(float)
    if np.allclose(x - y, 0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def main():
    df = load_frame().sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    X = df[BASE_FEATURES].to_numpy(float)
    y_resid = df["target_residual_y"].to_numpy(float)
    base_pred_y = df["gompertz_bayes_pred_y"].to_numpy(float)
    base_error_train_all = base_pred_y - df["holdout_y_obs"].to_numpy(float)

    folds = make_outer_folds_from_groups(df["subject_id"].to_numpy(str), 5)
    residual_pred = np.zeros(len(df), dtype=float)
    train_residual_fit = np.zeros(len(df), dtype=float)
    lambdas = []
    chosen = []
    for f in np.unique(folds):
        train = folds != f
        test = folds == f
        best = tune_candidate_and_lambda(X[train], y_resid[train], base_error_train_all[train])
        cand = best["candidate"]
        lam = best["lambda"]
        pred_test, pred_train = fit_predict_candidate(cand, X[train], y_resid[train], X[test])
        residual_pred[test] = lam * pred_test
        # Store fitted residuals for diagnostics only.
        lambdas.append(lam)
        chosen.append(cand.name)
        print(f"Outer fold {f}: {cand.name}, lambda={lam:.2f}, inner_loss={best['loss']:.5f}, n_test={int(test.sum())}")

    pop_pred_y = base_pred_y + residual_pred
    add_method_columns(df, "population_residual", pop_pred_y)
    copy_gompertz_interval(df, "population_residual")
    add_method_columns(df, "gompertz_bayes_copy", base_pred_y)
    copy_gompertz_interval(df, "gompertz_bayes_copy")
    add_conformal(df, "population_residual_conformal", "population_residual")
    # Point columns for conformal method.
    for suf in ["pred_y", "pred_V", "error_y", "abs_error_y", "sq_error_y", "error_V", "abs_error_V", "sq_error_V"]:
        df[f"population_residual_conformal_{suf}"] = df[f"population_residual_{suf}"]

    df["population_residual_delta_y"] = residual_pred
    df.to_csv(OUT / "nlstt_population_residual_patient_level.csv", index=False)

    # Load prior method summaries if available.
    rows = [
        summarize(df, "gompertz_bayes", "Bayesian Gompertz"),
        summarize(df, "population_residual", "Gompertz-guided population residual"),
        summarize(df, "population_residual_conformal", "Population residual + conformal"),
    ]
    for optional, display in [
        ("size_complexity_gate", "Size-aware complexity gate"),
        ("size_gate_conformal_add", "Size-aware gate + conformal"),
        ("residual_map", "Residual PINN MAP only"),
        ("nls", "NLS Gompertz"),
    ]:
        if f"{optional}_error_y" in df.columns:
            rows.append(summarize(df, optional, display))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "nlstt_population_residual_summary.csv", index=False)

    stats = []
    for b in ["gompertz_bayes", "size_complexity_gate", "residual_map", "nls"]:
        if f"{b}_error_y" not in df:
            continue
        for metric in ["sq_error_y", "abs_error_y", "sq_error_V", "abs_error_V"]:
            stats.append({
                "method_a": "population_residual",
                "method_b": b,
                "metric": metric,
                "wilcoxon_p": paired_p(df, "population_residual", b, metric),
                "mean_diff_a_minus_b": float(np.nanmean(df[f"population_residual_{metric}"] - df[f"{b}_{metric}"])),
            })
    pd.DataFrame(stats).to_csv(OUT / "nlstt_population_residual_paired_stats.csv", index=False)

    sub_rows = []
    for subset, sub in [("All", df), *[(f"Growth class: {k}", v) for k, v in df.groupby("growth_class")]]:
        for method, display in [
            ("gompertz_bayes", "Bayesian Gompertz"),
            ("population_residual", "Population residual"),
            ("size_complexity_gate", "Size-aware gate"),
            ("residual_map", "Residual MAP"),
        ]:
            if f"{method}_error_y" not in sub:
                continue
            row = summarize(sub, method, display)
            row["subset"] = subset
            sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(OUT / "nlstt_population_residual_subgroup_summary.csv", index=False)

    with open(OUT / "table_nlstt_population_residual_for_paper.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Population residual correction results on the NLSTt-300 cohort.}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Method & RMSE (log) & MAE (vol.) & Coverage & Interval score & Rel. width \\\\\n\\midrule\n")
        for _, r in summary.iterrows():
            vals = [
                r["method_display"],
                f"{r['rmse_log']:.4f}",
                f"{r['mae_volume']:.4f}",
                f"{r['coverage_95']:.3f}" if np.isfinite(r["coverage_95"]) else "N/A",
                f"{r['interval_score']:.4f}" if np.isfinite(r["interval_score"]) else "N/A",
                f"{r['mean_relative_interval_width_V']:.3f}" if np.isfinite(r["mean_relative_interval_width_V"]) else "N/A",
            ]
            if r["method"] == "population_residual_conformal":
                vals = [f"\\textbf{{{v}}}" for v in vals]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("\nSUMMARY")
    print(summary[["method_display", "rmse_log", "mae_volume", "coverage_95", "interval_score", "mean_relative_interval_width_V"]].to_string(index=False))
    print("\nChosen outer-fold models:", chosen)
    print("Outer-fold lambdas:", lambdas)
    print("\nPAIRED STATS")
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()
