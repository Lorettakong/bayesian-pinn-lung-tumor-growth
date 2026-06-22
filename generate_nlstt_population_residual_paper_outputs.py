#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs_nlstt_validation"
OUT = SRC / "paper_outputs_population_residual"


METHOD_LABELS = {
    "gompertz_bayes": "Bayesian Gompertz",
    "population_residual": "Gompertz-guided\npopulation residual",
    "population_residual_conformal": "Population residual\n+ conformal",
    "residual_map": "Residual PINN\nMAP only",
    "nls": "NLS Gompertz",
}


def _fmt(x: float, digits: int = 4) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.{digits}f}"


def _savefig(name: str):
    for ext in ["png", "pdf"]:
        plt.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def _interval_score(obs, lo, hi, alpha=0.05):
    width = hi - lo
    if obs < lo:
        return width + 2.0 / alpha * (lo - obs)
    if obs > hi:
        return width + 2.0 / alpha * (obs - hi)
    return width


def write_main_table(summary: pd.DataFrame):
    preferred = ["gompertz_bayes", "population_residual", "population_residual_conformal", "residual_map", "nls"]
    rows = summary.set_index("method").loc[[m for m in preferred if m in set(summary["method"])]].reset_index()
    rows.to_csv(OUT / "table_main_population_residual.csv", index=False)
    with open(OUT / "table_main_population_residual.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Performance comparison on the NLSTt-300 cohort using the held-out third CT scan.}\n")
        f.write("\\label{tab:nlstt_population_residual}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Method & RMSE (log) & MAE (volume) & Coverage & Interval score & Rel. width \\\\\n\\midrule\n")
        for _, r in rows.iterrows():
            vals = [
                str(r["method_display"]),
                _fmt(r["rmse_log"]),
                _fmt(r["mae_volume"]),
                _fmt(r["coverage_95"], 3),
                _fmt(r["interval_score"]),
                _fmt(r["mean_relative_interval_width_V"], 3),
            ]
            if r["method"] == "population_residual_conformal":
                vals = [f"\\textbf{{{v}}}" for v in vals]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_ablation_table(audit: pd.DataFrame):
    ab = audit[audit["audit"] == "ablation"].copy()
    ab = ab[["feature_set", "n_features", "rmse_log", "mae_volume", "corr_pred_residual_target"]]
    name_map = {
        "all_no_leakage_features": "All non-leaking features",
        "early_ct_only": "Early CT features only",
        "gompertz_prediction_only": "Gompertz prediction only",
        "gompertz_uncertainty_only_no_leak": "Gompertz uncertainty only",
        "gompertz_prediction_uncertainty_no_leak": "Gompertz prediction + uncertainty",
    }
    ab["Feature set"] = ab["feature_set"].map(name_map).fillna(ab["feature_set"])
    ab.to_csv(OUT / "table_feature_ablation_no_leakage.csv", index=False)
    with open(OUT / "table_feature_ablation_no_leakage.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Feature ablation for the leakage-free population residual learner.}\n")
        f.write("\\label{tab:nlstt_population_residual_ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Feature set & No. features & RMSE (log) & MAE (volume) & Corr. \\\\\n\\midrule\n")
        for _, r in ab.iterrows():
            f.write(
                f"{r['Feature set']} & {int(r['n_features'])} & {_fmt(r['rmse_log'])} & "
                f"{_fmt(r['mae_volume'])} & {_fmt(r['corr_pred_residual_target'], 3)} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_stats_table(stats: pd.DataFrame):
    keep = stats[stats["method_b"] == "gompertz_bayes"].copy()
    metric_map = {
        "sq_error_y": "Squared log error",
        "abs_error_y": "Absolute log error",
        "sq_error_V": "Squared volume error",
        "abs_error_V": "Absolute volume error",
    }
    keep["Metric"] = keep["metric"].map(metric_map)
    keep.to_csv(OUT / "table_paired_tests_vs_bayesian_gompertz.csv", index=False)
    with open(OUT / "table_paired_tests_vs_bayesian_gompertz.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Paired Wilcoxon tests comparing the proposed population residual method against Bayesian Gompertz.}\n")
        f.write("\\label{tab:nlstt_paired_tests}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Metric & Mean difference & $p$-value \\\\\n\\midrule\n")
        for _, r in keep.iterrows():
            p = r["wilcoxon_p"]
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            f.write(f"{r['Metric']} & {r['mean_diff_a_minus_b']:.4f} & {p_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def plot_method_comparison(summary: pd.DataFrame):
    rows = summary[summary["method"].isin(["gompertz_bayes", "population_residual", "population_residual_conformal", "residual_map", "nls"])].copy()
    labels = [METHOD_LABELS[m].replace("\n", " ") for m in rows["method"]]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    colors = ["#536d79", "#2d72b8", "#1f9d72", "#a65f2b", "#7a6b8f"]
    axes[0].barh(y, rows["rmse_log"], color=colors[: len(rows)], height=0.68)
    axes[0].set_xlabel("RMSE (log-volume)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, rows["mae_volume"], color=colors[: len(rows)], height=0.68)
    axes[1].set_xlabel("MAE (volume, cm$^3$)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Held-out third-scan prediction performance", y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.4)
    _savefig("fig_method_comparison")


def plot_calibration(summary: pd.DataFrame):
    rows = summary[summary["method"].isin(["gompertz_bayes", "population_residual_conformal"])].copy()
    labels = [METHOD_LABELS[m] for m in rows["method"]]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].bar(x, rows["coverage_95"], color=["#536d79", "#1f9d72"], width=0.65)
    axes[0].axhline(0.95, color="#b83232", linestyle="--", linewidth=1.2, label="Nominal 95%")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, rows["interval_score"], color=["#536d79", "#1f9d72"], width=0.65)
    axes[1].set_ylabel("Mean interval score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Uncertainty calibration after conformal adjustment", y=1.02, fontsize=11)
    _savefig("fig_calibration_interval_score")


def plot_paired_improvement(df: pd.DataFrame):
    df = df.copy()
    df["improvement_abs_V"] = df["gompertz_bayes_abs_error_V"] - df["population_residual_abs_error_V"]
    df["improvement_sq_log"] = df["gompertz_bayes_sq_error_y"] - df["population_residual_sq_error_y"]
    df = df.sort_values("improvement_abs_V").reset_index(drop=True)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharex=True)
    colors = np.where(df["improvement_abs_V"] >= 0, "#2d72b8", "#c75146")
    axes[0].bar(x, df["improvement_abs_V"], color=colors, width=0.85)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Reduction in absolute volume error")
    axes[0].set_xlabel("Patients sorted by improvement")
    axes[0].grid(axis="y", alpha=0.25)
    colors2 = np.where(df["improvement_sq_log"] >= 0, "#2d72b8", "#c75146")
    axes[1].bar(x, df["improvement_sq_log"], color=colors2, width=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Reduction in squared log error")
    axes[1].set_xlabel("Patients sorted by volume-error improvement")
    axes[1].grid(axis="y", alpha=0.25)
    _savefig("fig_paired_improvement")
    df[["patient_id", "growth_class", "improvement_abs_V", "improvement_sq_log"]].to_csv(
        OUT / "paired_improvement_patient_points.csv", index=False
    )


def plot_subgroup(subgroup: pd.DataFrame):
    sub = subgroup[subgroup["subset"].str.startswith("Growth class:")].copy()
    sub["growth_class"] = sub["subset"].str.replace("Growth class: ", "", regex=False)
    sub = sub[sub["method"].isin(["gompertz_bayes", "population_residual"])]
    order = ["decreasing", "stable", "slow_growth", "rapid_growth"]
    sub["growth_class"] = pd.Categorical(sub["growth_class"], categories=order, ordered=True)
    pivot = sub.pivot(index="growth_class", columns="method", values="rmse_log").reindex(order)
    x = np.arange(len(pivot))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - width / 2, pivot["gompertz_bayes"], width, label="Bayesian Gompertz", color="#536d79")
    ax.bar(x + width / 2, pivot["population_residual"], width, label="Population residual", color="#2d72b8")
    ax.set_ylabel("RMSE (log-volume)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in order])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Prediction error by trajectory class", fontsize=11)
    _savefig("fig_subgroup_growth_class_rmse")


def plot_cases(df: pd.DataFrame):
    d = df.copy()
    d["improvement_abs_V"] = d["gompertz_bayes_abs_error_V"] - d["population_residual_abs_error_V"]
    best = d.nlargest(2, "improvement_abs_V")
    worst = d.nsmallest(1, "improvement_abs_V")
    neutral = d.iloc[(d["improvement_abs_V"].abs()).argsort()[:1]]
    cases = pd.concat([best, neutral, worst]).drop_duplicates("patient_id").head(4)
    cases.to_csv(OUT / "case_examples_selected_patients.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for case_no, (ax, panel, (_, r)) in enumerate(zip(axes.ravel(), panel_labels, cases.iterrows()), start=1):
        t = np.array([0, 1, 2], dtype=float)
        v = np.array([r["V0"], r["V1"], r["holdout_V_obs"]], dtype=float)
        x_bg = 1.96
        x_prop = 2.04
        ax.plot(t[:2], v[:2], "o-", color="#1f2328", linewidth=1.6, markersize=4.8, label="Observed input")
        ax.scatter([2], [v[2]], color="#d55e00", s=34, zorder=5, label="Held-out observation")
        ax.errorbar(
            [x_bg],
            [r["gompertz_bayes_pred_V"]],
            yerr=[
                [max(r["gompertz_bayes_pred_V"] - r["gompertz_bayes_interval_low_V"], 0)],
                [max(r["gompertz_bayes_interval_high_V"] - r["gompertz_bayes_pred_V"], 0)],
            ],
            fmt="s",
            color="#536d79",
            ecolor="#8ca0ac",
            elinewidth=2.0,
            capsize=4,
            markersize=5.8,
            label="Bayesian Gompertz (95% PI)",
        )
        ax.errorbar(
            [x_prop],
            [r["population_residual_pred_V"]],
            yerr=[
                [max(r["population_residual_pred_V"] - r["population_residual_conformal_interval_low_V"], 0)],
                [max(r["population_residual_conformal_interval_high_V"] - r["population_residual_pred_V"], 0)],
            ],
            fmt="D",
            color="#1f9d72",
            ecolor="#75c3aa",
            elinewidth=2.0,
            capsize=4,
            markersize=5.8,
            label="Proposed (95% PI)",
        )
        ax.text(0.02, 0.95, panel, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.set_title(f"Case {case_no}: {str(r['growth_class']).replace('_', ' ')}", fontsize=8.5, pad=5)
        ax.set_xlabel("Time since baseline", fontsize=8)
        ax.set_ylabel("Volume (cm$^3$)", fontsize=8)
        ax.set_xlim(-0.1, 2.16)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "1", "2"])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.20, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.5, w_pad=1.6)
    _savefig("fig_case_examples")


def plot_permutation_audit(audit: pd.DataFrame):
    rows = audit.copy()
    real = rows[(rows["audit"] == "ablation") & (rows["feature_set"] == "all_no_leakage_features")]
    perm = rows[rows["audit"] == "permutation"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.scatter(np.zeros(len(perm)), perm["rmse_log"], color="#9a9a9a", label="Permuted residual target")
    ax.scatter([0.35], real["rmse_log"], color="#2d72b8", s=70, label="Correct residual target")
    ax.axhline(0.566233, color="#536d79", linestyle="--", linewidth=1.1, label="Bayesian Gompertz")
    ax.set_xlim(-0.25, 0.65)
    ax.set_xticks([0, 0.35])
    ax.set_xticklabels(["Permutation\nruns", "Proposed"])
    ax.set_ylabel("RMSE (log-volume)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Leakage sanity check", fontsize=11)
    _savefig("fig_permutation_sanity_check")


def write_readme():
    text = """NLSTt population residual paper outputs

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
"""
    (OUT / "README_outputs.txt").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SRC / "nlstt_population_residual_summary.csv")
    patient = pd.read_csv(SRC / "nlstt_population_residual_patient_level.csv")
    stats = pd.read_csv(SRC / "nlstt_population_residual_paired_stats.csv")
    subgroup = pd.read_csv(SRC / "nlstt_population_residual_subgroup_summary.csv")
    audit = pd.read_csv(SRC / "nlstt_population_residual_audit_no_leakage.csv")

    write_main_table(summary)
    write_ablation_table(audit)
    write_stats_table(stats)
    plot_method_comparison(summary)
    plot_calibration(summary)
    plot_paired_improvement(patient)
    plot_subgroup(subgroup)
    plot_cases(patient)
    plot_permutation_audit(audit)
    write_readme()

    manifest = pd.DataFrame(
        {"file": sorted(p.name for p in OUT.iterdir() if p.is_file())}
    )
    manifest.to_csv(OUT / "manifest.csv", index=False)
    print(f"Saved paper outputs to: {OUT}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
