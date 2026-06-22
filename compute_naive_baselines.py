from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_volume_long_with_metadata.csv"
OUT = ROOT / "outputs_naive_baselines"


def summarize_method(name: str, y_true: np.ndarray, v_true: np.ndarray, y_pred: np.ndarray, v_pred: np.ndarray) -> dict:
    return {
        "method": name,
        "n": len(y_true),
        "rmse_log": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae_log": float(np.mean(np.abs(y_true - y_pred))),
        "rmse_volume": float(np.sqrt(np.mean((v_true - v_pred) ** 2))),
        "mae_volume": float(np.mean(np.abs(v_true - v_pred))),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    long = pd.read_csv(DATA)
    wide = (
        long.pivot_table(
            index=["patient_id", "trajectory_id", "growth_class"],
            columns="study_year",
            values=["t_rel", "V_obs_cm3"],
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "_".join(str(x) for x in col if str(x) != "")
        for col in wide.columns.to_flat_index()
    ]
    wide = wide.rename(
        columns={
            "t_rel_0": "t0",
            "t_rel_1": "t1",
            "t_rel_2": "t2",
            "V_obs_cm3_0": "V0",
            "V_obs_cm3_1": "V1",
            "V_obs_cm3_2": "V2",
        }
    )
    wide = wide.dropna(subset=["t0", "t1", "t2", "V0", "V1", "V2"]).copy()
    for col in ["V0", "V1", "V2"]:
        wide[col] = wide[col].astype(float)
    for col in ["t0", "t1", "t2"]:
        wide[col] = wide[col].astype(float)

    eps = 1e-8
    y0 = np.log(np.maximum(wide["V0"].to_numpy(float), eps))
    y1 = np.log(np.maximum(wide["V1"].to_numpy(float), eps))
    y2 = np.log(np.maximum(wide["V2"].to_numpy(float), eps))
    v0 = wide["V0"].to_numpy(float)
    v1 = wide["V1"].to_numpy(float)
    v2 = wide["V2"].to_numpy(float)
    t0 = wide["t0"].to_numpy(float)
    t1 = wide["t1"].to_numpy(float)
    t2 = wide["t2"].to_numpy(float)
    dt01 = np.maximum(t1 - t0, eps)
    dt12 = t2 - t1

    # Persistence: carry forward the second CT-derived volume.
    v_persist = v1.copy()
    y_persist = np.log(np.maximum(v_persist, eps))

    # Linear extrapolation in volume space.
    v_linear_volume = np.maximum(v1 + (v1 - v0) / dt01 * dt12, eps)
    y_linear_volume = np.log(v_linear_volume)

    # Log-linear extrapolation / constant volume-doubling-rate baseline.
    y_log_linear = y1 + (y1 - y0) / dt01 * dt12
    v_log_linear = np.exp(y_log_linear)

    # Fixed doubling over one follow-up interval from the second scan.
    v_fixed_double = 2.0 * v1
    y_fixed_double = np.log(np.maximum(v_fixed_double, eps))

    rows = [
        summarize_method("Persistence last observation", y2, v2, y_persist, v_persist),
        summarize_method("Linear extrapolation in volume", y2, v2, y_linear_volume, v_linear_volume),
        summarize_method("Log-linear / VDT extrapolation", y2, v2, y_log_linear, v_log_linear),
        summarize_method("Fixed one-interval volume doubling", y2, v2, y_fixed_double, v_fixed_double),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "naive_baseline_summary_300.csv", index=False)

    pred = wide[["patient_id", "trajectory_id", "growth_class", "V0", "V1", "V2"]].copy()
    pred["persistence_pred"] = v_persist
    pred["linear_volume_pred"] = v_linear_volume
    pred["log_linear_vdt_pred"] = v_log_linear
    pred["fixed_double_pred"] = v_fixed_double
    pred.to_csv(OUT / "naive_baseline_predictions_300.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nOutputs written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
