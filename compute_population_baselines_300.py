from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_volume_long_with_metadata.csv"
OUT = ROOT / "outputs_population_baselines_300"
EPS = 1e-8
RNG = np.random.default_rng(20260618)


def build_wide() -> pd.DataFrame:
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
    wide.columns = ["_".join(str(x) for x in col if str(x) != "") for col in wide.columns.to_flat_index()]
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
    for col in ["t0", "t1", "t2", "V0", "V1", "V2"]:
        wide[col] = wide[col].astype(float)
    wide["y0"] = np.log(np.maximum(wide["V0"], EPS))
    wide["y1"] = np.log(np.maximum(wide["V1"], EPS))
    wide["y2"] = np.log(np.maximum(wide["V2"], EPS))
    wide["dt01"] = np.maximum(wide["t1"] - wide["t0"], EPS)
    wide["dt12"] = wide["t2"] - wide["t1"]
    wide["early_log_growth"] = (wide["y1"] - wide["y0"]) / wide["dt01"]
    wide["abs_early_log_growth"] = np.abs(wide["early_log_growth"])
    wide["volume_ratio"] = wide["V1"] / np.maximum(wide["V0"], EPS)
    wide["log_volume_ratio"] = wide["y1"] - wide["y0"]
    wide["mean_train_volume"] = 0.5 * (wide["V0"] + wide["V1"])
    wide["time_span"] = wide["t2"] - wide["t0"]
    return wide.reset_index(drop=True)


def group_folds(groups: np.ndarray, n_splits: int = 5) -> list[np.ndarray]:
    unique = np.array(sorted(pd.unique(groups)))
    RNG.shuffle(unique)
    fold_groups = np.array_split(unique, n_splits)
    return [np.isin(groups, fg) for fg in fold_groups]


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def poly2(x: np.ndarray) -> np.ndarray:
    cols = [x]
    cols.append(x ** 2)
    interactions = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            interactions.append((x[:, i] * x[:, j]).reshape(-1, 1))
    if interactions:
        cols.append(np.hstack(interactions))
    return np.hstack(cols)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float, use_poly2: bool = False) -> np.ndarray:
    if use_poly2:
        x_train = poly2(x_train)
        x_test = poly2(x_test)
    xt = add_intercept(x_train)
    xs = add_intercept(x_test)
    penalty = np.eye(xt.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xt.T @ xt + alpha * penalty, xt.T @ y_train)
    return xs @ beta


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length: float) -> np.ndarray:
    x1s = np.sum(x1 * x1, axis=1, keepdims=True)
    x2s = np.sum(x2 * x2, axis=1, keepdims=True).T
    dist2 = np.maximum(x1s + x2s - 2.0 * (x1 @ x2.T), 0.0)
    return np.exp(-0.5 * dist2 / max(length, 1e-6) ** 2)


def krr_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, length: float, alpha: float) -> np.ndarray:
    k = rbf_kernel(x_train, x_train, length)
    ks = rbf_kernel(x_test, x_train, length)
    coef = np.linalg.solve(k + alpha * np.eye(len(k)), y_train)
    return ks @ coef


def gp_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, length: float, noise: float) -> np.ndarray:
    y_mean = y_train.mean()
    yc = y_train - y_mean
    k = rbf_kernel(x_train, x_train, length)
    ks = rbf_kernel(x_test, x_train, length)
    coef = np.linalg.solve(k + noise**2 * np.eye(len(k)), yc)
    return y_mean + ks @ coef


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int) -> np.ndarray:
    pred = []
    for row in x_test:
        d = np.sqrt(np.sum((x_train - row) ** 2, axis=1))
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        pred.append(float(np.sum(w * y_train[idx]) / np.sum(w)))
    return np.asarray(pred)


def inner_select(x: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str) -> dict:
    folds = group_folds(groups, n_splits=4)
    candidates: list[dict] = []
    if method == "ridge":
        candidates = [{"alpha": a, "poly": False} for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]
    elif method == "poly_ridge":
        candidates = [{"alpha": a, "poly": True} for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]
    elif method == "krr":
        candidates = [{"length": l, "alpha": a} for l in [0.5, 1.0, 2.0, 5.0] for a in [0.01, 0.1, 1.0, 10.0]]
    elif method == "gp":
        candidates = [{"length": l, "noise": n} for l in [0.5, 1.0, 2.0, 5.0] for n in [0.05, 0.1, 0.2, 0.5]]
    elif method == "knn":
        candidates = [{"k": k} for k in [3, 5, 10, 20, 40]]

    best = {"loss": np.inf}
    for cand in candidates:
        losses = []
        for test_mask in folds:
            train_mask = ~test_mask
            x_tr, x_te = standardize(x[train_mask], x[test_mask])
            y_tr, y_te = y[train_mask], y[test_mask]
            if method in {"ridge", "poly_ridge"}:
                pred = ridge_predict(x_tr, y_tr, x_te, cand["alpha"], cand["poly"])
            elif method == "krr":
                pred = krr_predict(x_tr, y_tr, x_te, cand["length"], cand["alpha"])
            elif method == "gp":
                pred = gp_predict(x_tr, y_tr, x_te, cand["length"], cand["noise"])
            elif method == "knn":
                pred = knn_predict(x_tr, y_tr, x_te, cand["k"])
            losses.append(float(np.mean((y_te - pred) ** 2)))
        loss = float(np.mean(losses))
        if loss < best["loss"]:
            best = dict(cand)
            best["loss"] = loss
    return best


def summarize(name: str, frame: pd.DataFrame) -> dict:
    y_true = frame["y2"].to_numpy(float)
    y_pred = frame[f"{name}_y_pred"].to_numpy(float)
    v_true = frame["V2"].to_numpy(float)
    v_pred = np.exp(y_pred)
    return {
        "method": name,
        "n": len(frame),
        "rmse_log": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae_log": float(np.mean(np.abs(y_true - y_pred))),
        "rmse_volume": float(np.sqrt(np.mean((v_true - v_pred) ** 2))),
        "mae_volume": float(np.mean(np.abs(v_true - v_pred))),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = build_wide()
    feature_cols = [
        "y0",
        "y1",
        "early_log_growth",
        "abs_early_log_growth",
        "volume_ratio",
        "log_volume_ratio",
        "V0",
        "V1",
        "mean_train_volume",
        "dt12",
        "time_span",
    ]
    x = df[feature_cols].to_numpy(float)
    y = df["y2"].to_numpy(float)
    groups = df["patient_id"].astype(str).to_numpy()
    outer_folds = group_folds(groups, n_splits=5)

    methods = ["ridge", "poly_ridge", "krr", "gp", "knn"]
    for method in methods:
        df[f"{method}_y_pred"] = np.nan

    fold_rows = []
    for fold_id, test_mask in enumerate(outer_folds):
        train_mask = ~test_mask
        for method in methods:
            best = inner_select(x[train_mask], y[train_mask], groups[train_mask], method)
            x_tr, x_te = standardize(x[train_mask], x[test_mask])
            if method in {"ridge", "poly_ridge"}:
                pred = ridge_predict(x_tr, y[train_mask], x_te, best["alpha"], best["poly"])
            elif method == "krr":
                pred = krr_predict(x_tr, y[train_mask], x_te, best["length"], best["alpha"])
            elif method == "gp":
                pred = gp_predict(x_tr, y[train_mask], x_te, best["length"], best["noise"])
            elif method == "knn":
                pred = knn_predict(x_tr, y[train_mask], x_te, best["k"])
            df.loc[test_mask, f"{method}_y_pred"] = pred
            fold_rows.append({"fold": fold_id, "method": method, "n_test": int(test_mask.sum()), **best})
            print(f"fold={fold_id} method={method} best={best}")

    summaries = pd.DataFrame([summarize(method, df) for method in methods])
    display_names = {
        "ridge": "Population ridge regression",
        "poly_ridge": "Polynomial ridge regression",
        "krr": "RBF kernel ridge regression",
        "gp": "Gaussian process regression",
        "knn": "Distance-weighted kNN regression",
    }
    summaries["method_display"] = summaries["method"].map(display_names)
    summaries = summaries[["method_display", "n", "rmse_log", "mae_log", "rmse_volume", "mae_volume", "method"]]
    summaries.to_csv(OUT / "population_baseline_summary_300.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "population_baseline_fold_selection_300.csv", index=False)
    df.to_csv(OUT / "population_baseline_predictions_300.csv", index=False)
    print("\nSUMMARY")
    print(summaries.drop(columns=["method"]).to_string(index=False))
    print(f"\nOutputs written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
