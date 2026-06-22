#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs_nlstt_validation"
OUT = SRC / "paper_outputs_population_residual"


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)
    y = np.arange(1, len(values) + 1, dtype=float) / len(values)
    return values, y


def main():
    df = pd.read_csv(SRC / "nlstt_population_residual_patient_level.csv")

    series = {
        "Squared log-volume error": (
            df["gompertz_bayes_sq_error_y"].to_numpy(float),
            df["population_residual_sq_error_y"].to_numpy(float),
            "Squared log-volume error",
        ),
        "Absolute volume error": (
            df["gompertz_bayes_abs_error_V"].to_numpy(float),
            df["population_residual_abs_error_V"].to_numpy(float),
            r"Absolute volume error (cm$^3$)",
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), sharey=True)
    colors = {"Bayesian Gompertz": "#6a7f8d", "Proposed framework": "#1f9d72"}

    for ax, (title, (base, prop, xlabel)) in zip(axes, series.items()):
        xb, yb = ecdf(base)
        xp, yp = ecdf(prop)

        ax.step(xb, yb, where="post", color=colors["Bayesian Gompertz"], linewidth=2.0, label="Bayesian Gompertz")
        ax.step(xp, yp, where="post", color=colors["Proposed framework"], linewidth=2.0, label="Proposed framework")

        median_base = float(np.nanmedian(base))
        median_prop = float(np.nanmedian(prop))
        ax.axvline(median_base, color=colors["Bayesian Gompertz"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(median_prop, color=colors["Proposed framework"], linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of trajectories")
    axes[1].legend(loc="lower right", frameon=True, fontsize=8)
    fig.suptitle("Cumulative distribution of held-out prediction error", fontsize=11, y=1.03)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_prediction_error_ecdf.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = []
    for name, (base, prop, _) in series.items():
        summary.append(
            {
                "metric": name,
                "median_bayesian_gompertz": float(np.nanmedian(base)),
                "median_proposed": float(np.nanmedian(prop)),
                "mean_bayesian_gompertz": float(np.nanmean(base)),
                "mean_proposed": float(np.nanmean(prop)),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "fig_prediction_error_ecdf_summary.csv", index=False)


if __name__ == "__main__":
    main()
