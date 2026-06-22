#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs_nlstt_validation" / "nlstt_population_residual_patient_level.csv"
OUT = ROOT / "outputs_nlstt_validation" / "paper_outputs_population_residual"

CLASS_COLORS = {
    "decreasing": "#4c78a8",
    "stable": "#59a14f",
    "slow_growth": "#f28e2b",
    "rapid_growth": "#e15759",
}


def kernel_smooth(x: np.ndarray, y: np.ndarray, grid: np.ndarray, bandwidth: float):
    means = np.zeros_like(grid, dtype=float)
    ses = np.zeros_like(grid, dtype=float)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((x - g) / bandwidth) ** 2)
        if np.sum(w) <= 1e-12:
            means[i] = np.nan
            ses[i] = np.nan
            continue
        means[i] = np.sum(w * y) / np.sum(w)
        var = np.sum(w * (y - means[i]) ** 2) / np.sum(w)
        neff = (np.sum(w) ** 2) / np.sum(w * w)
        ses[i] = np.sqrt(var / max(neff, 1.0))
    return means, ses


def plot_panel(ax, d, xcol, xlabel, label):
    x = pd.to_numeric(d[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d["target_residual_y"], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    classes = d.loc[m, "growth_class"].astype(str).to_numpy()

    for cls in ["decreasing", "stable", "slow_growth", "rapid_growth"]:
        idx = classes == cls
        if idx.sum() == 0:
            continue
        ax.scatter(
            x[idx],
            y[idx],
            s=18,
            alpha=0.72,
            linewidth=0.25,
            edgecolor="white",
            color=CLASS_COLORS[cls],
            label=cls.replace("_", " "),
        )

    lo, hi = np.quantile(x, [0.02, 0.98])
    grid = np.linspace(lo, hi, 140)
    bw = max(0.25 * np.std(x), 1e-6)
    smooth, se = kernel_smooth(x, y, grid, bw)
    ax.plot(grid, smooth, color="#1f2328", linewidth=1.8)
    ax.fill_between(grid, smooth - 1.96 * se, smooth + 1.96 * se, color="#1f2328", alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#b83232", linestyle="--", linewidth=1.0)

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    ax.text(
        0.04,
        0.96,
        f"{label}\nPearson $r={pr:.2f}$\n$p={pp:.1e}$\nSpearman $\\rho={sr:.2f}$\n$p={sp:.1e}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#c8c8c8", alpha=0.92),
    )
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return {
        "panel": label,
        "x_column": xcol,
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "target_residual_y",
            "gompertz_bayes_pred_y",
            "train_log_growth",
            "gompertz_pred_relative_interval_width",
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    stats_rows = []
    stats_rows.append(
        plot_panel(
            axes[0],
            d,
            "gompertz_bayes_pred_y",
            "Bayesian Gompertz predicted log-volume",
            "(a)",
        )
    )
    stats_rows.append(
        plot_panel(
            axes[1],
            d,
            "train_log_growth",
            "Early log-growth from first two CT scans",
            "(b)",
        )
    )
    stats_rows.append(
        plot_panel(
            axes[2],
            d,
            "gompertz_pred_relative_interval_width",
            "Gompertz relative predictive interval width",
            "(c)",
        )
    )
    axes[0].set_ylabel("Held-out log-volume residual", fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    label_map = {
        "decreasing": "Decreasing",
        "stable": "Stable",
        "slow growth": "Slow growth",
        "rapid growth": "Rapid growth",
    }
    labels = [label_map.get(x, x) for x in labels]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Diagnostic residual structure of the Bayesian Gompertz backbone", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.12, 1, 0.98], w_pad=1.8)

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_gompertz_residual_diagnostics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(OUT / "fig_gompertz_residual_diagnostics_stats.csv", index=False)
    print(f"Saved: {OUT / 'fig_gompertz_residual_diagnostics.pdf'}")
    print(f"Saved: {OUT / 'fig_gompertz_residual_diagnostics.png'}")
    print(pd.DataFrame(stats_rows).to_string(index=False))


if __name__ == "__main__":
    main()
