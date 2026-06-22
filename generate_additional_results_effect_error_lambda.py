#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs_nlstt_validation"
OUT = SRC / "paper_outputs_population_residual"

sys.path.insert(0, str(ROOT))
from nlstt_population_residual_pipeline import (  # noqa: E402
    BASE_FEATURES,
    load_frame,
    make_outer_folds_from_groups,
    tune_candidate_and_lambda,
    fit_predict_candidate,
    rmse,
    mae,
)


def paired_rank_biserial(baseline_error: np.ndarray, proposed_error: np.ndarray) -> float:
    """Rank-biserial effect size for paired signed-rank differences.

    Positive values indicate lower error for the proposed method.
    """
    diff = np.asarray(baseline_error, dtype=float) - np.asarray(proposed_error, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return np.nan
    ranks = pd.Series(np.abs(diff)).rank(method="average").to_numpy(float)
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom > 0 else np.nan


def paired_cohens_dz(baseline_error: np.ndarray, proposed_error: np.ndarray) -> float:
    diff = np.asarray(baseline_error, dtype=float) - np.asarray(proposed_error, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.std(diff, ddof=1) < 1e-12:
        return np.nan
    return float(np.mean(diff) / np.std(diff, ddof=1))


def p_value(baseline_error: np.ndarray, proposed_error: np.ndarray) -> float:
    pair = pd.DataFrame({"b": baseline_error, "p": proposed_error}).dropna()
    if len(pair) < 2:
        return np.nan
    d = pair["b"].to_numpy(float) - pair["p"].to_numpy(float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(pair["b"], pair["p"], zero_method="wilcox").pvalue)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "N/A"
    if p < 1e-3:
        return f"{p:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-").replace("e+0", r"\times10^{") + "}"
    return f"{p:.4f}"


def write_effect_size_table(df: pd.DataFrame):
    proposed = {
        "sq_log": df["population_residual_sq_error_y"].to_numpy(float),
        "abs_vol": df["population_residual_abs_error_V"].to_numpy(float),
    }
    methods = [
        ("Bayesian Gompertz", "gompertz_bayes"),
        ("Residual PINN MAP only", "residual_map"),
        ("NLS Gompertz", "nls"),
    ]
    rows = []
    for display, prefix in methods:
        b_sq = df[f"{prefix}_sq_error_y"].to_numpy(float)
        b_abs = df[f"{prefix}_abs_error_V"].to_numpy(float)
        rows.append(
            {
                "baseline": display,
                "p_log": p_value(b_sq, proposed["sq_log"]),
                "p_vol": p_value(b_abs, proposed["abs_vol"]),
                "rank_biserial_log": paired_rank_biserial(b_sq, proposed["sq_log"]),
                "rank_biserial_vol": paired_rank_biserial(b_abs, proposed["abs_vol"]),
                "cohens_dz_log": paired_cohens_dz(b_sq, proposed["sq_log"]),
                "cohens_dz_vol": paired_cohens_dz(b_abs, proposed["abs_vol"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_effect_size_paired_tests.csv", index=False)

    with open(OUT / "table_effect_size_paired_tests.tex", "w") as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n")
        f.write("\\caption{Paired statistical comparison between the proposed residual framework and baseline methods.}\n")
        f.write("\\label{tab:paired_effect_size_tests}\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Baseline method & $p_{\\mathrm{log}}$ & $p_{\\mathrm{vol}}$ & $r_{\\mathrm{rb,log}}$ & $r_{\\mathrm{rb,vol}}$ \\\\\n\\midrule\n")
        for _, r in out.iterrows():
            f.write(
                f"{r['baseline']} & ${fmt_p(r['p_log'])}$ & ${fmt_p(r['p_vol'])}$ & "
                f"{r['rank_biserial_log']:.3f} & {r['rank_biserial_vol']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\n")
        f.write("\\vspace{1mm}\n")
        f.write("\\footnotesize{$p_{\\mathrm{log}}$ is based on paired squared log-volume error and $p_{\\mathrm{vol}}$ on paired absolute volume error. Rank-biserial effect sizes ($r_{\\mathrm{rb}}$) are computed from paired signed-rank differences; positive values indicate lower held-out error for the proposed framework.}\n")
        f.write("\\end{table}\n")
    return out


def plot_error_distribution(df: pd.DataFrame):
    data = [
        df["gompertz_bayes_sq_error_y"].to_numpy(float),
        df["population_residual_sq_error_y"].to_numpy(float),
        df["gompertz_bayes_abs_error_V"].to_numpy(float),
        df["population_residual_abs_error_V"].to_numpy(float),
    ]
    labels = ["Bayesian\nGompertz", "Proposed\nresidual", "Bayesian\nGompertz", "Proposed\nresidual"]
    colors = ["#647985", "#1f9d72", "#647985", "#1f9d72"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    rng = np.random.default_rng(7)
    for ax, vals, lab, col, title, ylabel in [
        (axes[0], data[:2], labels[:2], colors[:2], "Squared log-volume error", "Squared log-volume error"),
        (axes[1], data[2:], labels[2:], colors[2:], "Absolute volume error", "Absolute volume error (cm$^3$)"),
    ]:
        bp = ax.boxplot(vals, patch_artist=True, widths=0.48, showfliers=False)
        for patch, c in zip(bp["boxes"], col):
            patch.set_facecolor(c)
            patch.set_alpha(0.28)
            patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("#1f2328")
            med.set_linewidth(1.6)
        for i, y in enumerate(vals, start=1):
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            x = rng.normal(i, 0.045, size=len(y))
            ax.scatter(x, y, s=9, color=col[i - 1], alpha=0.35, linewidth=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(lab)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Held-out prediction error distributions", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_prediction_error_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def recompute_unshrunk_residual_predictions():
    df = load_frame().sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    X = df[BASE_FEATURES].to_numpy(float)
    y_resid = df["target_residual_y"].to_numpy(float)
    base_pred_y = df["gompertz_bayes_pred_y"].to_numpy(float)
    obs_y = df["holdout_y_obs"].to_numpy(float)
    base_error_train_all = base_pred_y - obs_y
    folds = make_outer_folds_from_groups(df["subject_id"].to_numpy(str), 5)

    unshrunk = np.zeros(len(df), dtype=float)
    selected_rows = []
    for fold in np.unique(folds):
        train = folds != fold
        test = folds == fold
        best = tune_candidate_and_lambda(X[train], y_resid[train], base_error_train_all[train])
        cand = best["candidate"]
        pred_test, _ = fit_predict_candidate(cand, X[train], y_resid[train], X[test])
        unshrunk[test] = pred_test
        selected_rows.append(
            {
                "fold": int(fold),
                "selected_model": cand.name,
                "inner_cv_selected_lambda": best["lambda"],
                "inner_loss": best["loss"],
                "n_test": int(test.sum()),
            }
        )
    pd.DataFrame(selected_rows).to_csv(OUT / "lambda_sensitivity_selected_models.csv", index=False)
    return df, unshrunk


def run_lambda_sensitivity():
    # Use the same patient-level outputs as the main paper tables. This makes
    # lambda=0 exactly match the Bayesian Gompertz backbone and lambda=1 exactly
    # match the proposed residual framework reported in the main results.
    df = pd.read_csv(SRC / "nlstt_population_residual_patient_level.csv")
    rhat = df["population_residual_delta_y"].to_numpy(float)
    base_y = df["gompertz_bayes_pred_y"].to_numpy(float)
    obs_y = df["holdout_y_obs"].to_numpy(float)
    obs_v = df["holdout_V_obs"].to_numpy(float)
    lambdas = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50], dtype=float)
    rows = []
    for lam in lambdas:
        pred_y = base_y + lam * rhat
        pred_v = np.exp(pred_y)
        err_y = pred_y - obs_y
        err_v = pred_v - obs_v
        rows.append(
            {
                "lambda": lam,
                "rmse_log": rmse(err_y),
                "mae_log": mae(err_y),
                "rmse_volume": rmse(err_v),
                "mae_volume": mae(err_v),
            }
        )
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "table_lambda_sensitivity.csv", index=False)
    with open(OUT / "table_lambda_sensitivity.tex", "w") as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n")
        f.write("\\caption{Sensitivity of prediction accuracy to the residual shrinkage coefficient $\\lambda$.}\n")
        f.write("\\label{tab:lambda_sensitivity}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("$\\lambda$ & RMSE (log) & MAE (volume) \\\\\n\\midrule\n")
        selected_i = int(np.where(np.isclose(sens["lambda"].to_numpy(float), 1.0))[0][0])
        for i, r in sens.iterrows():
            vals = [f"{r['lambda']:.2f}", f"{r['rmse_log']:.4f}", f"{r['mae_volume']:.4f}"]
            if i == selected_i:
                vals = [f"\\textbf{{{v}}}" for v in vals]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")
        f.write("\\vspace{1mm}\n")
        f.write("\\footnotesize{$\\lambda=0$ corresponds to the Bayesian Gompertz backbone without residual correction, and $\\lambda=1$ corresponds to the proposed cross-validated residual correction used in the main experiments. RMSE is reported in the log-volume domain and MAE is reported in the original volume domain for consistency with the main cohort-level evaluation.}\n")
        f.write("\\end{table}\n")

    fig, ax1 = plt.subplots(figsize=(5.8, 3.6))
    ax1.plot(sens["lambda"], sens["rmse_log"], marker="o", color="#2d72b8", linewidth=1.8, label="RMSE (log)")
    ax1.set_xlabel("Residual shrinkage coefficient $\\lambda$")
    ax1.set_ylabel("RMSE (log-volume)", color="#2d72b8")
    ax1.tick_params(axis="y", labelcolor="#2d72b8")
    ax1.grid(alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(sens["lambda"], sens["mae_volume"], marker="s", color="#1f9d72", linewidth=1.8, label="MAE (volume)")
    ax2.set_ylabel("MAE (volume, cm$^3$)", color="#1f9d72")
    ax2.tick_params(axis="y", labelcolor="#1f9d72")
    ax1.axvline(1.0, color="#555555", linestyle=":", linewidth=1.1)
    fig.suptitle("Sensitivity to residual shrinkage", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_lambda_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return sens


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC / "nlstt_population_residual_patient_level.csv")
    eff = write_effect_size_table(df)
    plot_error_distribution(df)
    sens = run_lambda_sensitivity()
    print("Saved effect-size table, error distribution figure, and lambda sensitivity outputs to:")
    print(OUT)
    print("\nEffect sizes:")
    print(eff.to_string(index=False))
    print("\nLambda sensitivity:")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
