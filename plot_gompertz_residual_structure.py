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


CLASS_LABELS = {
    "decreasing": "Decreasing",
    "stable": "Stable",
    "slow_growth": "Slow growth",
    "rapid_growth": "Rapid growth",
}

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
        # Effective sample size for a weighted mean.
        neff = (np.sum(w) ** 2) / np.sum(w * w)
        ses[i] = np.sqrt(var / max(neff, 1.0))
    return means, ses


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    cols = ["patient_id", "growth_class", "gompertz_bayes_pred_y", "holdout_y_obs", "target_residual_y"]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["gompertz_bayes_pred_y", "target_residual_y"])
    d["growth_class"] = d["growth_class"].fillna("unknown")

    x = d["gompertz_bayes_pred_y"].to_numpy(float)
    y = d["target_residual_y"].to_numpy(float)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    slope, intercept, linear_r, linear_p, slope_se = stats.linregress(x, y)

    lo, hi = np.quantile(x, [0.02, 0.98])
    grid = np.linspace(lo, hi, 160)
    bandwidth = 0.35 * np.std(x)
    smooth, se = kernel_smooth(x, y, grid, bandwidth=bandwidth)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for cls, group in d.groupby("growth_class", sort=False):
        ax.scatter(
            group["gompertz_bayes_pred_y"],
            group["target_residual_y"],
            s=28,
            alpha=0.78,
            linewidth=0.35,
            edgecolor="white",
            color=CLASS_COLORS.get(cls, "#8c8c8c"),
            label=CLASS_LABELS.get(cls, str(cls)),
        )

    ax.axhline(0.0, color="#b83232", linestyle="--", linewidth=1.2, label="Zero residual")
    ax.plot(grid, smooth, color="#1f2328", linewidth=2.0, label="Smoothed residual trend")
    ax.fill_between(grid, smooth - 1.96 * se, smooth + 1.96 * se, color="#1f2328", alpha=0.13, linewidth=0)

    # Add a light linear trend as a transparent diagnostic reference.
    ax.plot(grid, intercept + slope * grid, color="#6f4e7c", linewidth=1.2, alpha=0.65, linestyle=":", label="Linear trend")

    text = (
        f"$n={len(d)}$\n"
        f"Pearson $r={pearson_r:.2f}$, $p={pearson_p:.1e}$\n"
        f"Spearman $\\rho={spearman_r:.2f}$, $p={spearman_p:.1e}$\n"
        f"Slope $={slope:.2f}$, $p={linear_p:.1e}$"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c8c8c8", alpha=0.92),
    )

    ax.set_xlabel("Bayesian Gompertz predicted log-volume")
    ax.set_ylabel("Held-out log-volume residual")
    ax.set_title("Residual structure of the Bayesian Gompertz backbone", fontsize=11, pad=8)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=7.8, ncol=1)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_gompertz_residual_structure.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    diag = pd.DataFrame(
        [
            {
                "n": len(d),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "linear_slope": slope,
                "linear_slope_se": slope_se,
                "linear_slope_p": linear_p,
                "residual_mean": float(np.mean(y)),
                "residual_sd": float(np.std(y, ddof=1)),
            }
        ]
    )
    diag.to_csv(OUT / "fig_gompertz_residual_structure_stats.csv", index=False)
    d.to_csv(OUT / "fig_gompertz_residual_structure_points.csv", index=False)

    print(f"Saved: {OUT / 'fig_gompertz_residual_structure.pdf'}")
    print(f"Saved: {OUT / 'fig_gompertz_residual_structure.png'}")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
