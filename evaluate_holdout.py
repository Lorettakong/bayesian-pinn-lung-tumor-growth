import os
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from Datasets import load_patient_timeseries, get_available_patient_ids

@dataclass
class HoldoutEvalConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    map_dir: str = 'outputs'
    uq_dir: str = 'outputs_uq'
    bpinn_dir: str = 'outputs_bpinn'
    pinn_det_dir: str = 'outputs_pinn_det'
    pinn_bayes_dir: str = 'outputs_pinn_bayes'
    gompertz_bayes_uq_dir: str = 'outputs_uq_gompertz_bayes'
    gp_dir: str = 'outputs_gp'
    bgp_dir: str = 'outputs_bgp'
    nls_dir: str = 'outputs_nls'
    holdout_rule: str = 'last_timepoint'
    time_tol: float = 1e-08
    interval_alpha: float = 0.05
    out_dir: str = 'outputs_validation'

def get_time_column_name(df: pd.DataFrame) -> str:
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in df.columns:
            return col
    raise ValueError('No recognizable time column found.')

def safe_read_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def safe_mean(x) -> float:
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    return float(x.mean())

def safe_rmse_from_errors(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return np.nan
    return float(np.sqrt(np.mean(errors ** 2)))

def safe_mae_from_errors(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return np.nan
    return float(np.mean(np.abs(errors)))

def paired_ttest_from_columns(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, float]:
    out = {'n_pairs': 0, 'mean_diff_a_minus_b': np.nan, 't_statistic': np.nan, 'p_value': np.nan}
    if col_a not in df.columns or col_b not in df.columns:
        return out
    pair_df = df[[col_a, col_b]].copy().apply(pd.to_numeric, errors='coerce').dropna()
    if len(pair_df) < 2:
        return out
    a = pair_df[col_a].to_numpy(dtype=float)
    b = pair_df[col_b].to_numpy(dtype=float)
    test = ttest_rel(a, b, nan_policy='omit')
    out['n_pairs'] = int(len(pair_df))
    out['mean_diff_a_minus_b'] = float(np.mean(a - b))
    out['t_statistic'] = float(test.statistic) if np.isfinite(test.statistic) else np.nan
    out['p_value'] = float(test.pvalue) if np.isfinite(test.pvalue) else np.nan
    return out

def paired_wilcoxon_from_columns(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, float]:
    out = {'n_pairs': 0, 'median_diff_a_minus_b': np.nan, 'wilcoxon_statistic': np.nan, 'p_value': np.nan}
    if col_a not in df.columns or col_b not in df.columns:
        return out
    pair_df = df[[col_a, col_b]].copy().apply(pd.to_numeric, errors='coerce').dropna()
    if len(pair_df) < 2:
        return out
    a = pair_df[col_a].to_numpy(dtype=float)
    b = pair_df[col_b].to_numpy(dtype=float)
    diffs = a - b
    if np.allclose(diffs, 0.0):
        out['n_pairs'] = int(len(pair_df))
        out['median_diff_a_minus_b'] = 0.0
        out['wilcoxon_statistic'] = 0.0
        out['p_value'] = 1.0
        return out
    try:
        test = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided', correction=False)
        out['n_pairs'] = int(len(pair_df))
        out['median_diff_a_minus_b'] = float(np.median(diffs))
        out['wilcoxon_statistic'] = float(test.statistic) if np.isfinite(test.statistic) else np.nan
        out['p_value'] = float(test.pvalue) if np.isfinite(test.pvalue) else np.nan
    except ValueError:
        pass
    return out

def paired_effect_size_from_columns(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, float]:
    out = {'n_pairs': 0, 'mean_diff_a_minus_b': np.nan, 'effect_size_dz': np.nan}
    if col_a not in df.columns or col_b not in df.columns:
        return out
    pair_df = df[[col_a, col_b]].copy().apply(pd.to_numeric, errors='coerce').dropna()
    if len(pair_df) < 2:
        return out
    diffs = pair_df[col_a].to_numpy(dtype=float) - pair_df[col_b].to_numpy(dtype=float)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan
    out['n_pairs'] = int(len(pair_df))
    out['mean_diff_a_minus_b'] = float(np.mean(diffs))
    out['effect_size_dz'] = float(np.mean(diffs) / sd) if np.isfinite(sd) and sd > 0 else np.nan
    return out

def build_paired_ttest_table(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [('bpinn', 'bayes'), ('bpinn', 'map'), ('bpinn', 'nls'), ('bgp', 'bayes'), ('bgp', 'map'), ('bgp', 'nls'), ('bgp', 'gp'), ('gp', 'bayes'), ('gp', 'map'), ('gp', 'nls'), ('pinn_det', 'pinn_bayes'), ('pinn_det', 'bayes'), ('pinn_bayes', 'bayes'), ('pinn_bayes', 'bgp'), ('pinn_bayes', 'map'), ('pinn_bayes', 'nls'), ('bayes', 'map'), ('bayes', 'gompertz_bayes'), ('bayes', 'bgp'), ('bayes', 'gp'), ('bayes', 'nls'), ('gompertz_bayes', 'map'), ('gompertz_bayes', 'nls'), ('map', 'nls')]
    metric_defs = [('sq_error_y', 'Squared log-error'), ('abs_error_y', 'Absolute log-error'), ('sq_error_V', 'Squared volume-error'), ('abs_error_V', 'Absolute volume-error')]
    rows = []
    for method_a, method_b in comparisons:
        for metric_key, metric_label in metric_defs:
            stats = paired_ttest_from_columns(df, col_a=f'{method_a}_{metric_key}', col_b=f'{method_b}_{metric_key}')
            rows.append({'method_a': method_a, 'method_b': method_b, 'metric': metric_key, 'metric_display': metric_label, **stats})
    return pd.DataFrame(rows)

def build_paired_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    method_pairs = [('pinn_det', 'pinn_bayes'), ('pinn_bayes', 'bayes'), ('gompertz_bayes', 'bayes'), ('gp', 'bayes'), ('bgp', 'bayes'), ('pinn_det', 'bayes'), ('nls', 'bayes')]
    metrics = [('sq_error_y', 'RMSE (log)'), ('abs_error_V', 'MAE (volume)')]
    rows = []
    for method_a, method_b in method_pairs:
        for metric_key, metric_label in metrics:
            col_a = f'{method_a}_{metric_key}'
            col_b = f'{method_b}_{metric_key}'
            t_stats = paired_ttest_from_columns(df, col_a, col_b)
            w_stats = paired_wilcoxon_from_columns(df, col_a, col_b)
            e_stats = paired_effect_size_from_columns(df, col_a, col_b)
            rows.append({'method_a': method_a, 'method_b': method_b, 'metric': metric_key, 'metric_display': metric_label, 'n_pairs': t_stats['n_pairs'], 'mean_diff_a_minus_b': t_stats['mean_diff_a_minus_b'], 't_statistic': t_stats['t_statistic'], 'p_value_ttest': t_stats['p_value'], 'median_diff_a_minus_b': w_stats['median_diff_a_minus_b'], 'wilcoxon_statistic': w_stats['wilcoxon_statistic'], 'p_value_wilcoxon': w_stats['p_value'], 'effect_size_dz': e_stats['effect_size_dz']})
    return pd.DataFrame(rows)

def safe_find_row_by_time(df: pd.DataFrame, time_col: str, target_t: float, tol: float) -> pd.Series | None:
    if time_col not in df.columns:
        return None
    diff = np.abs(df[time_col].astype(float) - float(target_t))
    matches = df.loc[diff <= tol]
    if len(matches) == 0:
        return None
    return matches.iloc[0]

def sort_df_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    time_col = get_time_column_name(df)
    df_sorted = df.copy()
    df_sorted[time_col] = df_sorted[time_col].astype(float)
    df_sorted = df_sorted.sort_values(time_col, kind='mergesort').reset_index(drop=True)
    return (df_sorted, time_col)

def find_holdout_observation(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Define holdout as the last observed time point after explicit time sorting.
    """
    df_sorted, time_col = sort_df_by_time(df_raw)
    row = df_sorted.iloc[-1]
    holdout_V_obs = float(row['V_obs']) if 'V_obs' in df_sorted.columns else np.nan
    holdout_y_obs = float(np.log(holdout_V_obs)) if pd.notna(holdout_V_obs) and holdout_V_obs > 0 else np.nan
    return {'time_col': time_col, 'holdout_t': float(row[time_col]), 'holdout_scan': row['scan'] if 'scan' in df_sorted.columns else '', 'holdout_V_obs': holdout_V_obs, 'holdout_y_obs': holdout_y_obs, 'n_scans': int(len(df_sorted)), 't_min': float(df_sorted[time_col].min()), 't_max': float(df_sorted[time_col].max())}

def compute_interval_score(obs: float, lo: float, hi: float, alpha: float) -> float:
    """
    Interval Score for a central (1-alpha) interval.
    Lower is better.
    """
    if pd.isna(obs) or pd.isna(lo) or pd.isna(hi):
        return np.nan
    obs = float(obs)
    lo = float(lo)
    hi = float(hi)
    alpha = float(alpha)
    width = hi - lo
    if obs < lo:
        return float(width + 2.0 / alpha * (lo - obs))
    if obs > hi:
        return float(width + 2.0 / alpha * (obs - hi))
    return float(width)

def compute_interval_metrics(obs: float, pred_mean: float, lo: float, hi: float, alpha: float=0.05) -> Dict[str, float]:
    out = {'error': np.nan, 'abs_error': np.nan, 'sq_error': np.nan, 'interval_width': np.nan, 'relative_interval_width': np.nan, 'covered_95CI': np.nan, 'interval_score': np.nan}
    if pd.notna(obs) and pd.notna(pred_mean):
        err = float(pred_mean - obs)
        out['error'] = err
        out['abs_error'] = abs(err)
        out['sq_error'] = err ** 2
    if pd.notna(lo) and pd.notna(hi):
        width = float(hi - lo)
        out['interval_width'] = width
        if pd.notna(obs) and abs(float(obs)) > 1e-12:
            out['relative_interval_width'] = width / abs(float(obs))
    if pd.notna(obs) and pd.notna(lo) and pd.notna(hi):
        out['covered_95CI'] = int(obs >= lo and obs <= hi)
        out['interval_score'] = compute_interval_score(obs, lo, hi, alpha)
    return out

def empty_map_result():
    return {'map_available': 0, 'map_pred_V': np.nan, 'map_pred_y': np.nan, 'map_interval_low_V': np.nan, 'map_interval_high_V': np.nan, 'map_interval_width_V': np.nan, 'map_relative_interval_width_V': np.nan, 'map_covered_95CI_V': np.nan, 'map_interval_score_V': np.nan, 'map_error_V': np.nan, 'map_abs_error_V': np.nan, 'map_sq_error_V': np.nan, 'map_error_y': np.nan, 'map_abs_error_y': np.nan, 'map_sq_error_y': np.nan}

def empty_bayes_result():
    return {'bayes_available': 0, 'bayes_pred_V': np.nan, 'bayes_pred_y': np.nan, 'bayes_interval_low_V': np.nan, 'bayes_interval_high_V': np.nan, 'bayes_interval_width_V': np.nan, 'bayes_relative_interval_width_V': np.nan, 'bayes_covered_95CI_V': np.nan, 'bayes_interval_score_V': np.nan, 'bayes_error_V': np.nan, 'bayes_abs_error_V': np.nan, 'bayes_sq_error_V': np.nan, 'bayes_error_y': np.nan, 'bayes_abs_error_y': np.nan, 'bayes_sq_error_y': np.nan}

def empty_bpinn_result():
    return {'bpinn_available': 0, 'bpinn_pred_V': np.nan, 'bpinn_pred_y': np.nan, 'bpinn_interval_low_V': np.nan, 'bpinn_interval_high_V': np.nan, 'bpinn_interval_width_V': np.nan, 'bpinn_relative_interval_width_V': np.nan, 'bpinn_covered_95CI_V': np.nan, 'bpinn_interval_score_V': np.nan, 'bpinn_error_V': np.nan, 'bpinn_abs_error_V': np.nan, 'bpinn_sq_error_V': np.nan, 'bpinn_error_y': np.nan, 'bpinn_abs_error_y': np.nan, 'bpinn_sq_error_y': np.nan}

def empty_pinn_bayes_result():
    return {'pinn_bayes_available': 0, 'pinn_bayes_pred_V': np.nan, 'pinn_bayes_pred_y': np.nan, 'pinn_bayes_interval_low_V': np.nan, 'pinn_bayes_interval_high_V': np.nan, 'pinn_bayes_interval_width_V': np.nan, 'pinn_bayes_relative_interval_width_V': np.nan, 'pinn_bayes_covered_95CI_V': np.nan, 'pinn_bayes_interval_score_V': np.nan, 'pinn_bayes_error_V': np.nan, 'pinn_bayes_abs_error_V': np.nan, 'pinn_bayes_sq_error_V': np.nan, 'pinn_bayes_error_y': np.nan, 'pinn_bayes_abs_error_y': np.nan, 'pinn_bayes_sq_error_y': np.nan}

def empty_pinn_det_result():
    return {'pinn_det_available': 0, 'pinn_det_pred_V': np.nan, 'pinn_det_pred_y': np.nan, 'pinn_det_interval_low_V': np.nan, 'pinn_det_interval_high_V': np.nan, 'pinn_det_interval_width_V': np.nan, 'pinn_det_relative_interval_width_V': np.nan, 'pinn_det_covered_95CI_V': np.nan, 'pinn_det_interval_score_V': np.nan, 'pinn_det_error_V': np.nan, 'pinn_det_abs_error_V': np.nan, 'pinn_det_sq_error_V': np.nan, 'pinn_det_error_y': np.nan, 'pinn_det_abs_error_y': np.nan, 'pinn_det_sq_error_y': np.nan}

def empty_gompertz_bayes_result():
    return {'gompertz_bayes_available': 0, 'gompertz_bayes_pred_V': np.nan, 'gompertz_bayes_pred_y': np.nan, 'gompertz_bayes_interval_low_V': np.nan, 'gompertz_bayes_interval_high_V': np.nan, 'gompertz_bayes_interval_width_V': np.nan, 'gompertz_bayes_relative_interval_width_V': np.nan, 'gompertz_bayes_covered_95CI_V': np.nan, 'gompertz_bayes_interval_score_V': np.nan, 'gompertz_bayes_error_V': np.nan, 'gompertz_bayes_abs_error_V': np.nan, 'gompertz_bayes_sq_error_V': np.nan, 'gompertz_bayes_error_y': np.nan, 'gompertz_bayes_abs_error_y': np.nan, 'gompertz_bayes_sq_error_y': np.nan}

def empty_gp_result():
    return {'gp_available': 0, 'gp_pred_V': np.nan, 'gp_pred_y': np.nan, 'gp_interval_low_V': np.nan, 'gp_interval_high_V': np.nan, 'gp_interval_width_V': np.nan, 'gp_relative_interval_width_V': np.nan, 'gp_covered_95CI_V': np.nan, 'gp_interval_score_V': np.nan, 'gp_error_V': np.nan, 'gp_abs_error_V': np.nan, 'gp_sq_error_V': np.nan, 'gp_error_y': np.nan, 'gp_abs_error_y': np.nan, 'gp_sq_error_y': np.nan}

def empty_bgp_result():
    return {'bgp_available': 0, 'bgp_pred_V': np.nan, 'bgp_pred_y': np.nan, 'bgp_interval_low_V': np.nan, 'bgp_interval_high_V': np.nan, 'bgp_interval_width_V': np.nan, 'bgp_relative_interval_width_V': np.nan, 'bgp_covered_95CI_V': np.nan, 'bgp_interval_score_V': np.nan, 'bgp_error_V': np.nan, 'bgp_abs_error_V': np.nan, 'bgp_sq_error_V': np.nan, 'bgp_error_y': np.nan, 'bgp_abs_error_y': np.nan, 'bgp_sq_error_y': np.nan}

def empty_nls_result():
    return {'nls_available': 0, 'nls_pred_V': np.nan, 'nls_pred_y': np.nan, 'nls_interval_low_V': np.nan, 'nls_interval_high_V': np.nan, 'nls_interval_width_V': np.nan, 'nls_relative_interval_width_V': np.nan, 'nls_covered_95CI_V': np.nan, 'nls_interval_score_V': np.nan, 'nls_error_V': np.nan, 'nls_abs_error_V': np.nan, 'nls_sq_error_V': np.nan, 'nls_error_y': np.nan, 'nls_abs_error_y': np.nan, 'nls_sq_error_y': np.nan}

def read_map_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    pred_path = os.path.join(cfg.map_dir, f'predictions_{patient_id}.csv')
    df = safe_read_csv(pred_path)
    if df is None:
        return empty_map_result()
    time_col = get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_map_result()
    pred_V = float(row['V_pred']) if 'V_pred' in row.index else np.nan
    pred_y = float(row['y_pred']) if 'y_pred' in row.index else np.nan
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'map_available': 1, 'map_pred_V': pred_V, 'map_pred_y': pred_y, 'map_interval_low_V': np.nan, 'map_interval_high_V': np.nan, 'map_interval_width_V': np.nan, 'map_relative_interval_width_V': np.nan, 'map_covered_95CI_V': np.nan, 'map_interval_score_V': np.nan, 'map_error_V': metrics_V['error'], 'map_abs_error_V': metrics_V['abs_error'], 'map_sq_error_V': metrics_V['sq_error'], 'map_error_y': metrics_y['error'], 'map_abs_error_y': metrics_y['abs_error'], 'map_sq_error_y': metrics_y['sq_error']}

def read_bayesian_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.uq_dir, f'uq_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_bayes_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_bayes_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    lo_V = float(row['V_pred_q025']) if 'V_pred_q025' in row.index else np.nan
    hi_V = float(row['V_pred_q975']) if 'V_pred_q975' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=lo_V, hi=hi_V, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'bayes_available': 1, 'bayes_pred_V': pred_V, 'bayes_pred_y': pred_y, 'bayes_interval_low_V': lo_V, 'bayes_interval_high_V': hi_V, 'bayes_interval_width_V': metrics_V['interval_width'], 'bayes_relative_interval_width_V': metrics_V['relative_interval_width'], 'bayes_covered_95CI_V': metrics_V['covered_95CI'], 'bayes_interval_score_V': metrics_V['interval_score'], 'bayes_error_V': metrics_V['error'], 'bayes_abs_error_V': metrics_V['abs_error'], 'bayes_sq_error_V': metrics_V['sq_error'], 'bayes_error_y': metrics_y['error'], 'bayes_abs_error_y': metrics_y['abs_error'], 'bayes_sq_error_y': metrics_y['sq_error']}

def read_bpinn_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.bpinn_dir, f'bpinn_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_bpinn_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_bpinn_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    lo_V = float(row['V_pred_q025']) if 'V_pred_q025' in row.index else np.nan
    hi_V = float(row['V_pred_q975']) if 'V_pred_q975' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=lo_V, hi=hi_V, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'bpinn_available': 1, 'bpinn_pred_V': pred_V, 'bpinn_pred_y': pred_y, 'bpinn_interval_low_V': lo_V, 'bpinn_interval_high_V': hi_V, 'bpinn_interval_width_V': metrics_V['interval_width'], 'bpinn_relative_interval_width_V': metrics_V['relative_interval_width'], 'bpinn_covered_95CI_V': metrics_V['covered_95CI'], 'bpinn_interval_score_V': metrics_V['interval_score'], 'bpinn_error_V': metrics_V['error'], 'bpinn_abs_error_V': metrics_V['abs_error'], 'bpinn_sq_error_V': metrics_V['sq_error'], 'bpinn_error_y': metrics_y['error'], 'bpinn_abs_error_y': metrics_y['abs_error'], 'bpinn_sq_error_y': metrics_y['sq_error']}

def read_pinn_bayes_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.pinn_bayes_dir, f'pinn_bayes_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_pinn_bayes_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_pinn_bayes_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    lo_V = float(row['V_pred_q025']) if 'V_pred_q025' in row.index else np.nan
    hi_V = float(row['V_pred_q975']) if 'V_pred_q975' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=lo_V, hi=hi_V, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'pinn_bayes_available': 1, 'pinn_bayes_pred_V': pred_V, 'pinn_bayes_pred_y': pred_y, 'pinn_bayes_interval_low_V': lo_V, 'pinn_bayes_interval_high_V': hi_V, 'pinn_bayes_interval_width_V': metrics_V['interval_width'], 'pinn_bayes_relative_interval_width_V': metrics_V['relative_interval_width'], 'pinn_bayes_covered_95CI_V': metrics_V['covered_95CI'], 'pinn_bayes_interval_score_V': metrics_V['interval_score'], 'pinn_bayes_error_V': metrics_V['error'], 'pinn_bayes_abs_error_V': metrics_V['abs_error'], 'pinn_bayes_sq_error_V': metrics_V['sq_error'], 'pinn_bayes_error_y': metrics_y['error'], 'pinn_bayes_abs_error_y': metrics_y['abs_error'], 'pinn_bayes_sq_error_y': metrics_y['sq_error']}

def read_pinn_det_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.pinn_det_dir, f'pinn_det_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_pinn_det_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_pinn_det_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'pinn_det_available': 1, 'pinn_det_pred_V': pred_V, 'pinn_det_pred_y': pred_y, 'pinn_det_interval_low_V': np.nan, 'pinn_det_interval_high_V': np.nan, 'pinn_det_interval_width_V': np.nan, 'pinn_det_relative_interval_width_V': np.nan, 'pinn_det_covered_95CI_V': np.nan, 'pinn_det_interval_score_V': np.nan, 'pinn_det_error_V': metrics_V['error'], 'pinn_det_abs_error_V': metrics_V['abs_error'], 'pinn_det_sq_error_V': metrics_V['sq_error'], 'pinn_det_error_y': metrics_y['error'], 'pinn_det_abs_error_y': metrics_y['abs_error'], 'pinn_det_sq_error_y': metrics_y['sq_error']}

def read_gompertz_bayes_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.gompertz_bayes_uq_dir, f'uq_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_gompertz_bayes_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_gompertz_bayes_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    lo_V = float(row['V_pred_q025']) if 'V_pred_q025' in row.index else np.nan
    hi_V = float(row['V_pred_q975']) if 'V_pred_q975' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=lo_V, hi=hi_V, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'gompertz_bayes_available': 1, 'gompertz_bayes_pred_V': pred_V, 'gompertz_bayes_pred_y': pred_y, 'gompertz_bayes_interval_low_V': lo_V, 'gompertz_bayes_interval_high_V': hi_V, 'gompertz_bayes_interval_width_V': metrics_V['interval_width'], 'gompertz_bayes_relative_interval_width_V': metrics_V['relative_interval_width'], 'gompertz_bayes_covered_95CI_V': metrics_V['covered_95CI'], 'gompertz_bayes_interval_score_V': metrics_V['interval_score'], 'gompertz_bayes_error_V': metrics_V['error'], 'gompertz_bayes_abs_error_V': metrics_V['abs_error'], 'gompertz_bayes_sq_error_V': metrics_V['sq_error'], 'gompertz_bayes_error_y': metrics_y['error'], 'gompertz_bayes_abs_error_y': metrics_y['abs_error'], 'gompertz_bayes_sq_error_y': metrics_y['sq_error']}

def read_gp_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    pred_path = os.path.join(cfg.gp_dir, f'gp_predictions_{patient_id}.csv')
    df = safe_read_csv(pred_path)
    if df is None:
        return empty_gp_result()
    time_col = 't_rel_used' if 't_rel_used' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_gp_result()
    pred_V = float(row['V_pred']) if 'V_pred' in row.index else np.nan
    pred_y = float(row['y_pred']) if 'y_pred' in row.index else np.nan
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'gp_available': 1, 'gp_pred_V': pred_V, 'gp_pred_y': pred_y, 'gp_interval_low_V': np.nan, 'gp_interval_high_V': np.nan, 'gp_interval_width_V': np.nan, 'gp_relative_interval_width_V': np.nan, 'gp_covered_95CI_V': np.nan, 'gp_interval_score_V': np.nan, 'gp_error_V': metrics_V['error'], 'gp_abs_error_V': metrics_V['abs_error'], 'gp_sq_error_V': metrics_V['sq_error'], 'gp_error_y': metrics_y['error'], 'gp_abs_error_y': metrics_y['abs_error'], 'gp_sq_error_y': metrics_y['sq_error']}

def read_bgp_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    obs_pred_path = os.path.join(cfg.bgp_dir, f'bgp_observed_points_{patient_id}.csv')
    df = safe_read_csv(obs_pred_path)
    if df is None:
        return empty_bgp_result()
    time_col = 't_obs' if 't_obs' in df.columns else get_time_column_name(df)
    row = safe_find_row_by_time(df, time_col, holdout_t, cfg.time_tol)
    if row is None:
        return empty_bgp_result()
    obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
    pred_V = float(row['V_pred_mean']) if 'V_pred_mean' in row.index else np.nan
    pred_y = float(row['y_pred_mean']) if 'y_pred_mean' in row.index else np.nan
    lo_V = float(row['V_pred_q025']) if 'V_pred_q025' in row.index else np.nan
    hi_V = float(row['V_pred_q975']) if 'V_pred_q975' in row.index else np.nan
    metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=lo_V, hi=hi_V, alpha=cfg.interval_alpha)
    metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
    return {'bgp_available': 1, 'bgp_pred_V': pred_V, 'bgp_pred_y': pred_y, 'bgp_interval_low_V': lo_V, 'bgp_interval_high_V': hi_V, 'bgp_interval_width_V': metrics_V['interval_width'], 'bgp_relative_interval_width_V': metrics_V['relative_interval_width'], 'bgp_covered_95CI_V': metrics_V['covered_95CI'], 'bgp_interval_score_V': metrics_V['interval_score'], 'bgp_error_V': metrics_V['error'], 'bgp_abs_error_V': metrics_V['abs_error'], 'bgp_sq_error_V': metrics_V['sq_error'], 'bgp_error_y': metrics_y['error'], 'bgp_abs_error_y': metrics_y['abs_error'], 'bgp_sq_error_y': metrics_y['sq_error']}

def read_nls_holdout(cfg: HoldoutEvalConfig, patient_id: str, holdout_t: float, holdout_y_obs: float) -> Dict[str, Any]:
    """
    Flexible reader for baseline NLS.
    Priority:
      1) outputs_nls/nls_predictions_{pid}.csv
      2) outputs_nls/nls_summary_{pid}.csv
      3) outputs_nls/nls_summary_all_patients.csv
    """
    pred_path = os.path.join(cfg.nls_dir, f'nls_predictions_{patient_id}.csv')
    pred_df = safe_read_csv(pred_path)
    if pred_df is not None:
        time_col = get_time_column_name(pred_df)
        row = safe_find_row_by_time(pred_df, time_col, holdout_t, cfg.time_tol)
        if row is not None:
            obs_V = float(row['V_obs']) if 'V_obs' in row.index else np.nan
            pred_V = np.nan
            for c in ['V_pred', 'V_pred_mean', 'V_fit', 'V_pred_nls']:
                if c in row.index:
                    pred_V = float(row[c])
                    break
            pred_y = np.nan
            for c in ['y_pred', 'y_pred_mean', 'y_fit', 'y_pred_nls']:
                if c in row.index:
                    pred_y = float(row[c])
                    break
            metrics_V = compute_interval_metrics(obs=obs_V, pred_mean=pred_V, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
            metrics_y = compute_interval_metrics(obs=holdout_y_obs, pred_mean=pred_y, lo=np.nan, hi=np.nan, alpha=cfg.interval_alpha)
            return {'nls_available': 1, 'nls_pred_V': pred_V, 'nls_pred_y': pred_y, 'nls_interval_low_V': np.nan, 'nls_interval_high_V': np.nan, 'nls_interval_width_V': np.nan, 'nls_relative_interval_width_V': np.nan, 'nls_covered_95CI_V': np.nan, 'nls_interval_score_V': np.nan, 'nls_error_V': metrics_V['error'], 'nls_abs_error_V': metrics_V['abs_error'], 'nls_sq_error_V': metrics_V['sq_error'], 'nls_error_y': metrics_y['error'], 'nls_abs_error_y': metrics_y['abs_error'], 'nls_sq_error_y': metrics_y['sq_error']}
    summary_path = os.path.join(cfg.nls_dir, f'nls_summary_{patient_id}.csv')
    summary_df = safe_read_csv(summary_path)
    if summary_df is not None and len(summary_df) > 0:
        row = summary_df.iloc[0]
        return {'nls_available': 1, 'nls_pred_V': float(row['holdout_V_pred']) if 'holdout_V_pred' in row.index else np.nan, 'nls_pred_y': float(row['holdout_y_pred']) if 'holdout_y_pred' in row.index else np.nan, 'nls_interval_low_V': np.nan, 'nls_interval_high_V': np.nan, 'nls_interval_width_V': np.nan, 'nls_relative_interval_width_V': np.nan, 'nls_covered_95CI_V': np.nan, 'nls_interval_score_V': np.nan, 'nls_error_V': float(row['holdout_error_V']) if 'holdout_error_V' in row.index else np.nan, 'nls_abs_error_V': float(row['holdout_abs_error_V']) if 'holdout_abs_error_V' in row.index else np.nan, 'nls_sq_error_V': float(row['holdout_sq_error_V']) if 'holdout_sq_error_V' in row.index else np.nan, 'nls_error_y': float(row['holdout_y_error']) if 'holdout_y_error' in row.index else np.nan, 'nls_abs_error_y': float(row['holdout_abs_error_y']) if 'holdout_abs_error_y' in row.index else np.nan, 'nls_sq_error_y': float(row['holdout_sq_error_y']) if 'holdout_sq_error_y' in row.index else np.nan}
    all_path = os.path.join(cfg.nls_dir, 'nls_summary_all_patients.csv')
    all_df = safe_read_csv(all_path)
    if all_df is not None and len(all_df) > 0 and ('patient_id' in all_df.columns):
        all_df['patient_id'] = all_df['patient_id'].astype(str)
        match = all_df.loc[all_df['patient_id'] == str(patient_id)]
        if len(match) > 0:
            row = match.iloc[0]
            return {'nls_available': 1, 'nls_pred_V': float(row['holdout_V_pred']) if 'holdout_V_pred' in row.index else np.nan, 'nls_pred_y': float(row['holdout_y_pred']) if 'holdout_y_pred' in row.index else np.nan, 'nls_interval_low_V': np.nan, 'nls_interval_high_V': np.nan, 'nls_interval_width_V': np.nan, 'nls_relative_interval_width_V': np.nan, 'nls_covered_95CI_V': np.nan, 'nls_interval_score_V': np.nan, 'nls_error_V': float(row['holdout_error_V']) if 'holdout_error_V' in row.index else np.nan, 'nls_abs_error_V': float(row['holdout_abs_error_V']) if 'holdout_abs_error_V' in row.index else np.nan, 'nls_sq_error_V': float(row['holdout_sq_error_V']) if 'holdout_sq_error_V' in row.index else np.nan, 'nls_error_y': float(row['holdout_y_error']) if 'holdout_y_error' in row.index else np.nan, 'nls_abs_error_y': float(row['holdout_abs_error_y']) if 'holdout_abs_error_y' in row.index else np.nan, 'nls_sq_error_y': float(row['holdout_sq_error_y']) if 'holdout_sq_error_y' in row.index else np.nan}
    return empty_nls_result()

def evaluate_single_patient(cfg: HoldoutEvalConfig, patient_id: str) -> Dict[str, Any]:
    patient_id = str(patient_id)
    patient_id_loaded, t_data, y_obs, df_raw = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    patient_id_loaded = str(patient_id_loaded)
    holdout_info = find_holdout_observation(df_raw)
    holdout_t = holdout_info['holdout_t']
    holdout_y_obs = holdout_info['holdout_y_obs']
    map_res = read_map_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    bayes_res = read_bayesian_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    bpinn_res = read_bpinn_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    pinn_det_res = read_pinn_det_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    pinn_bayes_res = read_pinn_bayes_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    gompertz_bayes_res = read_gompertz_bayes_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    gp_res = read_gp_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    bgp_res = read_bgp_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    nls_res = read_nls_holdout(cfg, patient_id_loaded, holdout_t, holdout_y_obs)
    row = {'patient_id': patient_id_loaded, 'time_col': holdout_info['time_col'], 'holdout_rule': cfg.holdout_rule, 'holdout_scan': holdout_info['holdout_scan'], 'holdout_t': holdout_t, 'holdout_V_obs': holdout_info['holdout_V_obs'], 'holdout_y_obs': holdout_info['holdout_y_obs'], 'n_scans': holdout_info['n_scans'], 't_min': holdout_info['t_min'], 't_max': holdout_info['t_max'], **map_res, **bayes_res, **bpinn_res, **pinn_det_res, **pinn_bayes_res, **gompertz_bayes_res, **gp_res, **bgp_res, **nls_res}
    return row

def summarize_method(df: pd.DataFrame, method_prefix: str) -> Dict[str, Any]:
    avail_col = f'{method_prefix}_available'
    err_v_col = f'{method_prefix}_error_V'
    err_y_col = f'{method_prefix}_error_y'
    width_col = f'{method_prefix}_interval_width_V'
    rel_width_col = f'{method_prefix}_relative_interval_width_V'
    cover_col = f'{method_prefix}_covered_95CI_V'
    is_col = f'{method_prefix}_interval_score_V'
    df_use = df.copy()
    if avail_col in df_use.columns:
        df_use = df_use.loc[df_use[avail_col] == 1].copy()
    errors_v = df_use[err_v_col].to_numpy(dtype=float) if err_v_col in df_use.columns else np.array([])
    errors_y = df_use[err_y_col].to_numpy(dtype=float) if err_y_col in df_use.columns else np.array([])
    rmse_v = safe_rmse_from_errors(errors_v)
    mae_v = safe_mae_from_errors(errors_v)
    rmse_y = safe_rmse_from_errors(errors_y)
    mae_y = safe_mae_from_errors(errors_y)
    coverage = safe_mean(df_use[cover_col]) if cover_col in df_use.columns else np.nan
    mean_is = safe_mean(df_use[is_col]) if is_col in df_use.columns else np.nan
    return {'method': method_prefix, 'n_patients_available': int(len(df_use)), 'rmse_V_holdout': rmse_v, 'mae_V_holdout': mae_v, 'rmse_y_holdout': rmse_y, 'mae_y_holdout': mae_y, 'mean_interval_width_V': safe_mean(df_use[width_col]) if width_col in df_use.columns else np.nan, 'mean_relative_interval_width_V': safe_mean(df_use[rel_width_col]) if rel_width_col in df_use.columns else np.nan, 'coverage_95CI_V': coverage, 'coverage_minus_target_95CI_V': coverage - 0.95 if pd.notna(coverage) else np.nan, 'mean_interval_score_V': mean_is}

def evaluate_holdout(cfg: HoldoutEvalConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients to evaluate on holdout:')
    for pid in patient_ids:
        print(' -', pid)
    rows = []
    failed = []
    for pid in patient_ids:
        try:
            row = evaluate_single_patient(cfg, pid)
            rows.append(row)
        except Exception as e:
            print(f'\nERROR while evaluating patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if len(rows) == 0:
        print('No successful holdout evaluations.')
        return
    results_df = pd.DataFrame(rows).sort_values('patient_id').reset_index(drop=True)
    results_path = os.path.join(cfg.out_dir, 'holdout_results_all_patients.csv')
    results_df.to_csv(results_path, index=False)
    print(f'\nSaved patient-level holdout results to: {results_path}')
    method_rows = [summarize_method(results_df, 'bayes'), summarize_method(results_df, 'map'), summarize_method(results_df, 'nls')]
    method_df = pd.DataFrame(method_rows)
    pairwise_ttests = build_paired_ttest_table(results_df)
    pairwise_path = os.path.join(cfg.out_dir, 'holdout_paired_ttests.csv')
    pairwise_ttests.to_csv(pairwise_path, index=False)
    print(f'Saved paired t-tests to: {pairwise_path}')
    paired_stats = build_paired_stats_table(results_df)
    paired_stats_path = os.path.join(cfg.out_dir, 'holdout_paired_stats_extended.csv')
    paired_stats.to_csv(paired_stats_path, index=False)
    print(f'Saved extended paired stats to: {paired_stats_path}')
    p_sq_log_vs_bayes = {}
    p_abs_vol_vs_bayes = {}
    for baseline_method in ['map', 'nls']:
        sub_sq = pairwise_ttests.loc[(pairwise_ttests['method_a'] == 'bayes') & (pairwise_ttests['method_b'] == baseline_method) & (pairwise_ttests['metric'] == 'sq_error_y')]
        sub_abs = pairwise_ttests.loc[(pairwise_ttests['method_a'] == 'bayes') & (pairwise_ttests['method_b'] == baseline_method) & (pairwise_ttests['metric'] == 'abs_error_V')]
        p_sq_log_vs_bayes[baseline_method] = float(sub_sq.iloc[0]['p_value']) if len(sub_sq) > 0 else np.nan
        p_abs_vol_vs_bayes[baseline_method] = float(sub_abs.iloc[0]['p_value']) if len(sub_abs) > 0 else np.nan
    method_name_map = {'bayes': 'Bayesian PINN', 'map': 'Deterministic PINN (MAP)', 'nls': 'NLS Gompertz'}
    method_df['method_display'] = method_df['method'].map(method_name_map)
    method_df['p_value_sq_log_error_vs_bayes'] = method_df['method'].map(p_sq_log_vs_bayes)
    method_df['p_value_abs_volume_error_vs_bayes'] = method_df['method'].map(p_abs_vol_vs_bayes)
    summary_cols = ['method_display', 'n_patients_available', 'rmse_V_holdout', 'mae_V_holdout', 'rmse_y_holdout', 'mae_y_holdout', 'mean_relative_interval_width_V', 'mean_interval_score_V', 'coverage_95CI_V', 'coverage_minus_target_95CI_V', 'p_value_sq_log_error_vs_bayes', 'p_value_abs_volume_error_vs_bayes']
    summary_cols = [c for c in summary_cols if c in method_df.columns]
    method_df = method_df[summary_cols]
    summary_path = os.path.join(cfg.out_dir, 'holdout_method_summary.csv')
    method_df.to_csv(summary_path, index=False)
    print(f'Saved method-level summary to: {summary_path}')
    ablation_rows = [summarize_method(results_df, 'pinn_bayes'), summarize_method(results_df, 'gompertz_bayes'), summarize_method(results_df, 'bayes'), summarize_method(results_df, 'map'), summarize_method(results_df, 'nls')]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_name_map = {'pinn_bayes': 'PINN + Bayesian', 'gompertz_bayes': 'Gompertz + Bayesian', 'bayes': 'Proposed Bayesian PINN', 'map': 'Deterministic PINN (MAP)', 'nls': 'NLS Gompertz'}
    ablation_df['method_display'] = ablation_df['method'].map(ablation_name_map)
    ablation_cols = ['method_display', 'n_patients_available', 'rmse_V_holdout', 'mae_V_holdout', 'rmse_y_holdout', 'mae_y_holdout', 'mean_relative_interval_width_V', 'mean_interval_score_V', 'coverage_95CI_V', 'coverage_minus_target_95CI_V']
    ablation_df = ablation_df[[c for c in ablation_cols if c in ablation_df.columns]]
    ablation_path = os.path.join(cfg.out_dir, 'holdout_method_summary_ablations.csv')
    ablation_df.to_csv(ablation_path, index=False)
    print(f'Saved ablation summary to: {ablation_path}')
    extended_rows = [summarize_method(results_df, 'bayes'), summarize_method(results_df, 'pinn_bayes'), summarize_method(results_df, 'pinn_det'), summarize_method(results_df, 'gompertz_bayes'), summarize_method(results_df, 'bgp'), summarize_method(results_df, 'gp'), summarize_method(results_df, 'nls')]
    extended_df = pd.DataFrame(extended_rows)
    extended_name_map = {'bayes': 'Gompertz + PINN + Bayesian', 'pinn_bayes': 'PINN + Bayesian', 'pinn_det': 'Pure PINN', 'gompertz_bayes': 'Gompertz + Bayesian', 'bgp': 'Bayesian GP', 'gp': 'Pure GP', 'nls': 'Pure Gompertz'}
    extended_df['method_display'] = extended_df['method'].map(extended_name_map)
    extended_cols = ['method_display', 'n_patients_available', 'rmse_V_holdout', 'mae_V_holdout', 'rmse_y_holdout', 'mae_y_holdout', 'mean_relative_interval_width_V', 'mean_interval_score_V', 'coverage_95CI_V', 'coverage_minus_target_95CI_V']
    extended_df = extended_df[[c for c in extended_cols if c in extended_df.columns]]
    extended_path = os.path.join(cfg.out_dir, 'holdout_method_summary_extended.csv')
    extended_df.to_csv(extended_path, index=False)
    print(f'Saved extended method summary to: {extended_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'holdout_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved failed cases to: {failed_path}')
    print('\nHoldout evaluation finished.')
    print(f'Successful patients: {len(results_df)}')
    print(f'Failed patients    : {len(failed)}')
if __name__ == '__main__':
    config = HoldoutEvalConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, map_dir='outputs', uq_dir='outputs_uq', nls_dir='outputs_nls', out_dir='outputs_validation')
    evaluate_holdout(config)
