import os
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
TWO_COL_WIDTH_IN = 7.16
PANEL_HEIGHT_IN = 4.45
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

@dataclass
class CohortSummaryConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    map_dir: str = 'outputs'
    hmc_dir: str = 'outputs_hmc'
    uq_dir: str = 'outputs_uq'
    nls_dir: str = 'outputs_nls'
    validation_dir: str = 'outputs_validation'
    out_dir: str = 'outputs_validation'

def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1))

def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))

def _safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.median(x))

def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))

def _safe_cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    m = np.mean(x)
    if abs(m) < 1e-12:
        return np.nan
    return float(np.std(x, ddof=1) / abs(m))

def _safe_ci_width(x: np.ndarray, q_low: float=0.025, q_high: float=0.975) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q_high) - np.quantile(x, q_low))

def _safe_rel_ci_width(x: np.ndarray, q_low: float=0.025, q_high: float=0.975) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    if abs(med) < 1e-12:
        return np.nan
    width = np.quantile(x, q_high) - np.quantile(x, q_low)
    return float(width / abs(med))

def _fmt_ci(center: float, low: float, high: float, digits: int=3) -> str:
    if pd.isna(center) or pd.isna(low) or pd.isna(high):
        return ''
    return f'{center:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]'

def _fmt_pm(center: float, spread: float, digits: int=3) -> str:
    if pd.isna(center) or pd.isna(spread):
        return ''
    return f'{center:.{digits}f} ± {spread:.{digits}f}'

def _fmt_num_or_na(x: float, digits: int=3) -> str:
    if pd.isna(x):
        return 'N/A'
    return f'{float(x):.{digits}f}'

def _fmt_p_or_na(x: float) -> str:
    if pd.isna(x):
        return 'N/A'
    x = float(x)
    if x < 0.001:
        return f'{x:.2e}'
    return f'{x:.4f}'

def _rename_with_prefix(df: pd.DataFrame, prefix: str, exclude: List[str] | None=None) -> pd.DataFrame:
    exclude = exclude or []
    rename_map = {c: f'{prefix}{c}' for c in df.columns if c not in exclude}
    return df.rename(columns=rename_map)

def export_table3_for_paper(method_df: pd.DataFrame, out_dir: str) -> str:
    required_cols = ['method', 'rmse_y_holdout', 'mae_V_holdout', 'mean_relative_interval_width_V', 'mean_interval_score_V', 'coverage_minus_target_95CI_V', 'p_value_sq_log_error_vs_bayes', 'p_value_abs_volume_error_vs_bayes']
    missing = [c for c in required_cols if c not in method_df.columns]
    if missing:
        raise ValueError(f'Missing required columns for Table III export: {missing}')
    order_map = {'Bayesian PINN': 0, 'Deterministic PINN (MAP)': 1, 'NLS Gompertz': 2}
    df = method_df.copy()
    df['sort_order'] = df['method'].map(order_map).fillna(99)
    df = df.sort_values('sort_order').reset_index(drop=True)
    export_df = pd.DataFrame({'Method': df['method'], 'RMSE (log)': df['rmse_y_holdout'].map(lambda x: _fmt_num_or_na(x, digits=3)), 'p-value vs Bayesian PINN (RMSE log)': df['p_value_sq_log_error_vs_bayes'].map(_fmt_p_or_na), 'MAE (volume)': df['mae_V_holdout'].map(lambda x: _fmt_num_or_na(x, digits=3)), 'p-value vs Bayesian PINN (MAE volume)': df['p_value_abs_volume_error_vs_bayes'].map(_fmt_p_or_na), 'Mean Rel. CI Width': df['mean_relative_interval_width_V'].map(lambda x: _fmt_num_or_na(x, digits=3)), 'Interval Score': df['mean_interval_score_V'].map(lambda x: _fmt_num_or_na(x, digits=3)), 'Coverage Deviation': df['coverage_minus_target_95CI_V'].map(lambda x: _fmt_num_or_na(x, digits=2))})
    out_path = os.path.join(out_dir, 'table3_for_paper.csv')
    export_df.to_csv(out_path, index=False)
    print(f'Saved Table III export: {out_path}')
    return out_path

def build_cohort_summary(cfg: CohortSummaryConfig) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)
    map_summary = _safe_read_csv(os.path.join(cfg.map_dir, 'map_summary.csv'))
    uq_summary = _safe_read_csv(os.path.join(cfg.uq_dir, 'uq_summary_all_patients.csv'))
    nls_summary = _safe_read_csv(os.path.join(cfg.nls_dir, 'nls_summary_all_patients.csv'))
    holdout_results = _safe_read_csv(os.path.join(cfg.validation_dir, 'holdout_results_all_patients.csv'))
    holdout_method_summary = _safe_read_csv(os.path.join(cfg.validation_dir, 'holdout_method_summary.csv'))
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f'Raw tumor volume table not found: {cfg.csv_path}')
    raw_df = pd.read_csv(cfg.csv_path)
    if 'patient_id' not in raw_df.columns:
        raise ValueError('tumor_volumes.csv must contain patient_id column')
    raw_df['patient_id'] = raw_df['patient_id'].astype(str)
    if 'status' in raw_df.columns:
        raw_df = raw_df[raw_df['status'].astype(str).str.lower().str.strip() == 'ok'].copy()
    patient_ids = sorted(raw_df['patient_id'].unique().tolist())
    scan_counts = raw_df.groupby('patient_id').size().rename('n_scans').reset_index()
    rows: List[dict] = []
    for pid in patient_ids:
        npz_path = os.path.join(cfg.hmc_dir, f'hmc_samples_{pid}.npz')
        if not os.path.exists(npz_path):
            rows.append({'patient_id': str(pid), 'n_hmc_samples': 0, 'accept_rate': np.nan, 'alpha_median': np.nan, 'alpha_mean': np.nan, 'alpha_std': np.nan, 'alpha_q025': np.nan, 'alpha_q25': np.nan, 'alpha_q75': np.nan, 'alpha_q975': np.nan, 'alpha_ci_width': np.nan, 'alpha_rel_ci_width': np.nan, 'alpha_cv': np.nan, 'beta_median': np.nan, 'beta_mean': np.nan, 'beta_std': np.nan, 'beta_q025': np.nan, 'beta_q25': np.nan, 'beta_q75': np.nan, 'beta_q975': np.nan, 'beta_ci_width': np.nan, 'beta_rel_ci_width': np.nan, 'beta_cv': np.nan, 'y0_median': np.nan, 'y0_mean': np.nan, 'y0_std': np.nan, 'y0_q025': np.nan, 'y0_q25': np.nan, 'y0_q75': np.nan, 'y0_q975': np.nan, 'y0_ci_width': np.nan, 'y0_rel_ci_width': np.nan, 'y0_cv': np.nan, 'corr_alpha_beta': np.nan})
            continue
        data = np.load(npz_path)
        alpha = data['alpha'].astype(float)
        beta = data['beta'].astype(float)
        y0 = data['y0'].astype(float)
        accept_rate = float(data['accept_rate']) if 'accept_rate' in data else np.nan
        if 'alpha_beta_corr' in data:
            corr_alpha_beta = float(data['alpha_beta_corr'])
        elif 'corr_alpha_beta' in data:
            corr_alpha_beta = float(data['corr_alpha_beta'])
        else:
            corr_alpha_beta = _safe_corr(alpha, beta)
        row = {'patient_id': str(pid), 'n_hmc_samples': int(len(alpha)), 'accept_rate': accept_rate, 'alpha_median': _safe_median(alpha), 'alpha_mean': _safe_mean(alpha), 'alpha_std': _safe_std(alpha), 'alpha_q025': _safe_quantile(alpha, 0.025), 'alpha_q25': _safe_quantile(alpha, 0.25), 'alpha_q75': _safe_quantile(alpha, 0.75), 'alpha_q975': _safe_quantile(alpha, 0.975), 'alpha_ci_width': _safe_ci_width(alpha), 'alpha_rel_ci_width': _safe_rel_ci_width(alpha), 'alpha_cv': _safe_cv(alpha), 'beta_median': _safe_median(beta), 'beta_mean': _safe_mean(beta), 'beta_std': _safe_std(beta), 'beta_q025': _safe_quantile(beta, 0.025), 'beta_q25': _safe_quantile(beta, 0.25), 'beta_q75': _safe_quantile(beta, 0.75), 'beta_q975': _safe_quantile(beta, 0.975), 'beta_ci_width': _safe_ci_width(beta), 'beta_rel_ci_width': _safe_rel_ci_width(beta), 'beta_cv': _safe_cv(beta), 'y0_median': _safe_median(y0), 'y0_mean': _safe_mean(y0), 'y0_std': _safe_std(y0), 'y0_q025': _safe_quantile(y0, 0.025), 'y0_q25': _safe_quantile(y0, 0.25), 'y0_q75': _safe_quantile(y0, 0.75), 'y0_q975': _safe_quantile(y0, 0.975), 'y0_ci_width': _safe_ci_width(y0), 'y0_rel_ci_width': _safe_rel_ci_width(y0), 'y0_cv': _safe_cv(y0), 'corr_alpha_beta': corr_alpha_beta}
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary) == 0:
        print('No patients found. Nothing to summarize.')
        return summary
    summary = summary.merge(scan_counts, on='patient_id', how='left')
    if map_summary is not None and len(map_summary) > 0:
        map_summary = map_summary.copy()
        map_summary['patient_id'] = map_summary['patient_id'].astype(str)
        keep_cols = [c for c in ['patient_id', 'n_data', 't_min', 't_max', 't0_obs', 'y0_obs', 'alpha_map', 'beta_map', 'best_epoch', 'best_U_total', 'best_U_data', 'best_U_phys', 'best_U_prior', 'best_U_nn_prior', 'best_U_y0', 'final_U_total', 'final_U_data', 'final_U_phys', 'final_U_prior', 'final_U_nn_prior', 'final_U_y0', 'rmse_log', 'rmse_volume'] if c in map_summary.columns]
        summary = summary.merge(map_summary[keep_cols], on='patient_id', how='left')
    if uq_summary is not None and len(uq_summary) > 0:
        uq_summary = uq_summary.copy()
        uq_summary['patient_id'] = uq_summary['patient_id'].astype(str)
        keep_cols = [c for c in ['patient_id', 'n_hmc_samples', 'accept_rate', 'corr_alpha_beta', 't0', 'alpha_mean', 'alpha_std', 'alpha_q025', 'alpha_q50', 'alpha_q975', 'beta_mean', 'beta_std', 'beta_q025', 'beta_q50', 'beta_q975', 'y0_mean', 'y0_std', 'y0_q025', 'y0_q50', 'y0_q975', 'y_observed_coverage_95CI', 'V_observed_coverage_95CI', 'last_observed_V', 'final_V_mean', 'final_V_q025', 'final_V_q975', 'final_ci_width_V', 'final_ci_width_over_mean_V', 'mean_future_ci_width_V', 'max_future_ci_width_V', 'mean_future_ci_width_over_mean_V', 'max_future_ci_width_over_mean_V', 'alert_triggered', 'first_alert_t', 'last_observation_t', 'prediction_end_t'] if c in uq_summary.columns]
        uq_summary = uq_summary[keep_cols].copy()
        uq_summary = _rename_with_prefix(uq_summary, 'uq_', exclude=['patient_id'])
        summary = summary.merge(uq_summary, on='patient_id', how='left')
    if nls_summary is not None and len(nls_summary) > 0:
        nls_summary = nls_summary.copy()
        nls_summary['patient_id'] = nls_summary['patient_id'].astype(str)
        keep_cols = [c for c in ['patient_id', 'n_data', 't0', 'y0_obs', 't_min', 't_max', 'success', 'status', 'cost', 'alpha_nls', 'beta_nls', 'y0_nls', 'alpha_se', 'beta_se', 'y0_se', 'rmse_log', 'rmse_volume', 'mae_log', 'mae_volume', 'holdout_t', 'holdout_y_obs', 'holdout_y_pred', 'holdout_y_error', 'holdout_abs_error_y', 'holdout_sq_error_y', 'holdout_V_obs', 'holdout_V_pred', 'holdout_error_V', 'holdout_abs_error_V', 'holdout_sq_error_V'] if c in nls_summary.columns]
        nls_summary = nls_summary[keep_cols].copy()
        nls_summary = _rename_with_prefix(nls_summary, 'nlsfit_', exclude=['patient_id'])
        summary = summary.merge(nls_summary, on='patient_id', how='left')
    if holdout_results is not None and len(holdout_results) > 0:
        holdout_results = holdout_results.copy()
        holdout_results['patient_id'] = holdout_results['patient_id'].astype(str)
        keep_cols = [c for c in ['patient_id', 'holdout_rule', 'holdout_scan', 'holdout_t', 'holdout_V_obs', 'holdout_y_obs', 'bayes_pred_V', 'bayes_pred_y', 'bayes_interval_low_V', 'bayes_interval_high_V', 'bayes_interval_width_V', 'bayes_relative_interval_width_V', 'bayes_covered_95CI_V', 'bayes_interval_score_V', 'bayes_abs_error_V', 'bayes_abs_error_y', 'map_pred_V', 'map_pred_y', 'map_abs_error_V', 'map_abs_error_y', 'nls_pred_V', 'nls_pred_y', 'nls_abs_error_V', 'nls_abs_error_y'] if c in holdout_results.columns]
        summary = summary.merge(holdout_results[keep_cols], on='patient_id', how='left')
    summary['alpha_posterior'] = summary.apply(lambda r: _fmt_ci(r['alpha_median'], r['alpha_q025'], r['alpha_q975']), axis=1)
    summary['beta_posterior'] = summary.apply(lambda r: _fmt_ci(r['beta_median'], r['beta_q025'], r['beta_q975']), axis=1)
    summary['y0_posterior'] = summary.apply(lambda r: _fmt_ci(r['y0_median'], r['y0_q025'], r['y0_q975']), axis=1)
    summary['alpha_mean_pm_std'] = summary.apply(lambda r: _fmt_pm(r['alpha_mean'], r['alpha_std']), axis=1)
    summary['beta_mean_pm_std'] = summary.apply(lambda r: _fmt_pm(r['beta_mean'], r['beta_std']), axis=1)
    summary['y0_mean_pm_std'] = summary.apply(lambda r: _fmt_pm(r['y0_mean'], r['y0_std']), axis=1)
    summary['nls_alpha_pm_se'] = summary.apply(lambda r: _fmt_pm(r['nlsfit_alpha_nls'], r['nlsfit_alpha_se']) if 'nlsfit_alpha_nls' in summary.columns and 'nlsfit_alpha_se' in summary.columns else '', axis=1)
    summary['nls_beta_pm_se'] = summary.apply(lambda r: _fmt_pm(r['nlsfit_beta_nls'], r['nlsfit_beta_se']) if 'nlsfit_beta_nls' in summary.columns and 'nlsfit_beta_se' in summary.columns else '', axis=1)
    summary['large_uncertainty_flag'] = (summary['alpha_rel_ci_width'] > 1.5) | (summary['beta_rel_ci_width'] > 1.5)
    summary['low_accept_rate_flag'] = summary['accept_rate'] < 0.6 if 'accept_rate' in summary.columns else False
    concise_cols = ['patient_id', 'alpha_posterior', 'beta_posterior', 'y0_posterior', 'n_scans', 'rmse_log', 'rmse_volume', 'corr_alpha_beta', 'accept_rate', 'alpha_rel_ci_width', 'beta_rel_ci_width', 'large_uncertainty_flag', 'low_accept_rate_flag', 'holdout_V_obs', 'bayes_pred_V', 'bayes_interval_width_V', 'bayes_relative_interval_width_V', 'bayes_covered_95CI_V', 'bayes_abs_error_V', 'map_pred_V', 'map_abs_error_V', 'nls_pred_V', 'nls_abs_error_V', 'holdout_y_obs', 'bayes_pred_y', 'bayes_abs_error_y', 'map_pred_y', 'map_abs_error_y', 'nls_pred_y', 'nls_abs_error_y']
    concise_cols = [c for c in concise_cols if c in summary.columns]
    full_order = concise_cols + [c for c in summary.columns if c not in concise_cols]
    summary = summary[full_order].sort_values('patient_id').reset_index(drop=True)
    out_csv = os.path.join(cfg.out_dir, 'cohort_summary_compact.csv')
    summary.to_csv(out_csv, index=False)
    paper_cols = [c for c in ['patient_id', 'alpha_posterior', 'beta_posterior', 'n_scans', 'rmse_log', 'rmse_volume'] if c in summary.columns]
    paper_table = summary[paper_cols].copy()
    paper_csv = os.path.join(cfg.out_dir, 'cohort_summary_for_paper.csv')
    paper_table.to_csv(paper_csv, index=False)
    uncertainty_cols = [c for c in ['patient_id', 'alpha_median', 'alpha_q025', 'alpha_q975', 'alpha_ci_width', 'alpha_rel_ci_width', 'alpha_cv', 'beta_median', 'beta_q025', 'beta_q975', 'beta_ci_width', 'beta_rel_ci_width', 'beta_cv', 'y0_median', 'y0_q025', 'y0_q975', 'y0_ci_width', 'y0_rel_ci_width', 'y0_cv', 'accept_rate', 'corr_alpha_beta', 'large_uncertainty_flag', 'low_accept_rate_flag'] if c in summary.columns]
    uncertainty_table = summary[uncertainty_cols].copy()
    uncertainty_csv = os.path.join(cfg.out_dir, 'cohort_uncertainty_table.csv')
    uncertainty_table.to_csv(uncertainty_csv, index=False)
    holdout_cols = [c for c in ['patient_id', 'holdout_scan', 'holdout_t', 'holdout_V_obs', 'bayes_pred_V', 'bayes_interval_low_V', 'bayes_interval_high_V', 'bayes_interval_width_V', 'bayes_relative_interval_width_V', 'bayes_covered_95CI_V', 'bayes_abs_error_V', 'map_pred_V', 'map_abs_error_V', 'nls_pred_V', 'nls_abs_error_V', 'holdout_y_obs', 'bayes_pred_y', 'bayes_abs_error_y', 'map_pred_y', 'map_abs_error_y', 'nls_pred_y', 'nls_abs_error_y'] if c in summary.columns]
    if len(holdout_cols) > 0:
        holdout_patient_table = summary[holdout_cols].copy()
        holdout_patient_csv = os.path.join(cfg.out_dir, 'cohort_holdout_patient_table.csv')
        holdout_patient_table.to_csv(holdout_patient_csv, index=False)
        print(f'Saved holdout patient table: {holdout_patient_csv}')
    comparison_cols = [c for c in ['patient_id', 'alpha_map', 'beta_map', 'nlsfit_alpha_nls', 'nlsfit_beta_nls', 'alpha_median', 'beta_median', 'rmse_log', 'rmse_volume', 'nlsfit_rmse_log', 'nlsfit_rmse_volume', 'bayes_abs_error_V', 'map_abs_error_V', 'nls_abs_error_V', 'bayes_abs_error_y', 'map_abs_error_y', 'nls_abs_error_y'] if c in summary.columns]
    if len(comparison_cols) > 0:
        comparison_table = summary[comparison_cols].copy()
        comparison_csv = os.path.join(cfg.out_dir, 'cohort_methods_comparison_table.csv')
        comparison_table.to_csv(comparison_csv, index=False)
        print(f'Saved methods comparison table: {comparison_csv}')
    if holdout_method_summary is not None and len(holdout_method_summary) > 0:
        method_df = holdout_method_summary.copy()
        rename_map = {'method_display': 'method'}
        method_df = method_df.rename(columns=rename_map)
        preferred_cols = [c for c in ['method', 'n_patients_available', 'rmse_V_holdout', 'mae_V_holdout', 'rmse_y_holdout', 'mae_y_holdout', 'mean_relative_interval_width_V', 'mean_interval_score_V', 'coverage_95CI_V', 'coverage_minus_target_95CI_V', 'p_value_sq_log_error_vs_bayes', 'p_value_abs_volume_error_vs_bayes'] if c in method_df.columns]
        if len(preferred_cols) > 0:
            method_df = method_df[preferred_cols]
        method_summary_csv = os.path.join(cfg.out_dir, 'cohort_holdout_method_summary.csv')
        method_df.to_csv(method_summary_csv, index=False)
        print(f'Saved holdout method summary copy: {method_summary_csv}')
        export_table3_for_paper(method_df, cfg.out_dir)
    overall_rows = [{'metric': 'n_patients', 'value': len(summary)}, {'metric': 'mean_alpha_mean', 'value': _safe_mean(summary['alpha_mean'].to_numpy(dtype=float)) if 'alpha_mean' in summary.columns else np.nan}, {'metric': 'mean_beta_mean', 'value': _safe_mean(summary['beta_mean'].to_numpy(dtype=float)) if 'beta_mean' in summary.columns else np.nan}, {'metric': 'median_alpha_median', 'value': _safe_median(summary['alpha_median'].to_numpy(dtype=float)) if 'alpha_median' in summary.columns else np.nan}, {'metric': 'median_beta_median', 'value': _safe_median(summary['beta_median'].to_numpy(dtype=float)) if 'beta_median' in summary.columns else np.nan}, {'metric': 'mean_accept_rate', 'value': _safe_mean(summary['accept_rate'].to_numpy(dtype=float)) if 'accept_rate' in summary.columns else np.nan}, {'metric': 'mean_rmse_log_map', 'value': _safe_mean(summary['rmse_log'].to_numpy(dtype=float)) if 'rmse_log' in summary.columns else np.nan}, {'metric': 'mean_rmse_volume_map', 'value': _safe_mean(summary['rmse_volume'].to_numpy(dtype=float)) if 'rmse_volume' in summary.columns else np.nan}, {'metric': 'mean_rmse_log_nls', 'value': _safe_mean(summary['nlsfit_rmse_log'].to_numpy(dtype=float)) if 'nlsfit_rmse_log' in summary.columns else np.nan}, {'metric': 'mean_rmse_volume_nls', 'value': _safe_mean(summary['nlsfit_rmse_volume'].to_numpy(dtype=float)) if 'nlsfit_rmse_volume' in summary.columns else np.nan}, {'metric': 'mean_bayes_holdout_abs_error_V', 'value': _safe_mean(summary['bayes_abs_error_V'].to_numpy(dtype=float)) if 'bayes_abs_error_V' in summary.columns else np.nan}, {'metric': 'mean_map_holdout_abs_error_V', 'value': _safe_mean(summary['map_abs_error_V'].to_numpy(dtype=float)) if 'map_abs_error_V' in summary.columns else np.nan}, {'metric': 'mean_nls_holdout_abs_error_V', 'value': _safe_mean(summary['nls_abs_error_V'].to_numpy(dtype=float)) if 'nls_abs_error_V' in summary.columns else np.nan}, {'metric': 'mean_bayes_holdout_abs_error_y', 'value': _safe_mean(summary['bayes_abs_error_y'].to_numpy(dtype=float)) if 'bayes_abs_error_y' in summary.columns else np.nan}, {'metric': 'mean_map_holdout_abs_error_y', 'value': _safe_mean(summary['map_abs_error_y'].to_numpy(dtype=float)) if 'map_abs_error_y' in summary.columns else np.nan}, {'metric': 'mean_nls_holdout_abs_error_y', 'value': _safe_mean(summary['nls_abs_error_y'].to_numpy(dtype=float)) if 'nls_abs_error_y' in summary.columns else np.nan}, {'metric': 'mean_bayes_coverage_95CI', 'value': _safe_mean(summary['bayes_covered_95CI_V'].to_numpy(dtype=float)) if 'bayes_covered_95CI_V' in summary.columns else np.nan}, {'metric': 'alert_rate', 'value': _safe_mean(summary['uq_alert_triggered'].to_numpy(dtype=float)) if 'uq_alert_triggered' in summary.columns else np.nan}, {'metric': 'large_uncertainty_rate', 'value': _safe_mean(summary['large_uncertainty_flag'].to_numpy(dtype=float)) if 'large_uncertainty_flag' in summary.columns else np.nan}, {'metric': 'low_accept_rate_rate', 'value': _safe_mean(summary['low_accept_rate_flag'].to_numpy(dtype=float)) if 'low_accept_rate_flag' in summary.columns else np.nan}]
    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(cfg.out_dir, 'cohort_overall_stats.csv')
    overall_df.to_csv(overall_csv, index=False)
    print(f'Saved compact summary: {out_csv}')
    print(f'Saved paper table: {paper_csv}')
    print(f'Saved uncertainty table: {uncertainty_csv}')
    print(f'Saved overall stats: {overall_csv}')
    return summary

def _select_highlight_points_by_patient_id(summary: pd.DataFrame, highlight_patient_ids: List[str] | None=None) -> List[int]:
    """
    Return row indices for specified patient IDs.
    Only these points will be highlighted and annotated.
    """
    if highlight_patient_ids is None:
        highlight_patient_ids = ['100067', '100082', '100153', '100166']
    s = summary.copy().reset_index(drop=True)
    s['patient_id'] = s['patient_id'].astype(str)
    selected_idx = []
    for pid in highlight_patient_ids:
        match = s.index[s['patient_id'] == str(pid)].tolist()
        if len(match) > 0:
            selected_idx.append(match[0])
    return selected_idx

def _style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def _scan_marker(n_scans: float) -> str:
    if not np.isfinite(n_scans):
        return '^'
    n_scans = int(round(float(n_scans)))
    if n_scans <= 3:
        return 'o'
    if n_scans == 4:
        return 's'
    if n_scans == 5:
        return 'D'
    return 'P'

def plot_alpha_beta_population(summary: pd.DataFrame, out_dir: str, highlight_patient_ids: List[str] | None=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    required_cols = ['alpha_median', 'beta_median', 'alpha_q025', 'alpha_q975', 'beta_q025', 'beta_q975', 'patient_id']
    missing = [c for c in required_cols if c not in summary.columns]
    if missing:
        raise ValueError(f'Missing required columns for plotting: {missing}')
    if highlight_patient_ids is None:
        highlight_patient_ids = ['100067', '100082', '100153', '100166']
    df = summary.copy().reset_index(drop=True)
    df['patient_id'] = df['patient_id'].astype(str)
    x = df['alpha_median'].to_numpy(dtype=float)
    y = df['beta_median'].to_numpy(dtype=float)
    x_low = df['alpha_q025'].to_numpy(dtype=float)
    x_high = df['alpha_q975'].to_numpy(dtype=float)
    y_low = df['beta_q025'].to_numpy(dtype=float)
    y_high = df['beta_q975'].to_numpy(dtype=float)
    xerr_lower = np.maximum(0.0, x - x_low)
    xerr_upper = np.maximum(0.0, x_high - x)
    yerr_lower = np.maximum(0.0, y - y_low)
    yerr_upper = np.maximum(0.0, y_high - y)
    xerr = np.vstack([xerr_lower, xerr_upper])
    yerr = np.vstack([yerr_lower, yerr_upper])
    labels = df['patient_id'].tolist()
    highlight_idx = _select_highlight_points_by_patient_id(df, highlight_patient_ids=highlight_patient_ids)
    all_idx = np.arange(len(df))
    background_idx = np.array([i for i in all_idx if i not in highlight_idx], dtype=int)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, PANEL_HEIGHT_IN))
    _style_axes(ax)
    if len(background_idx) > 0:
        ax.errorbar(x[background_idx], y[background_idx], xerr=xerr[:, background_idx], yerr=yerr[:, background_idx], fmt='none', ecolor='#CBD5E1', elinewidth=0.8, capsize=2, alpha=0.42, zorder=1)
    if len(background_idx) > 0:
        ax.scatter(x[background_idx], y[background_idx], color='#94A3B8', marker='o', s=28, alpha=0.72, edgecolors='white', linewidths=0.45, label='Other patients', zorder=3)
    if len(highlight_idx) > 0:
        color_map = {'100067': '#2563EB', '100082': '#0F766E', '100153': '#D97706', '100166': '#DC2626'}
        marker_map = {'100067': 'o', '100082': 's', '100153': 'D', '100166': '*'}
        for i in highlight_idx:
            pid = labels[i]
            color = color_map.get(pid, '#F97316')
            marker = marker_map.get(pid, 'o')
            ax.errorbar([x[i]], [y[i]], xerr=xerr[:, [i]], yerr=yerr[:, [i]], fmt='none', ecolor=color, elinewidth=1.55, capsize=2.8, alpha=0.96, zorder=4)
            ax.scatter([x[i]], [y[i]], color=color, marker=marker, s=80 if marker != '*' else 135, edgecolors='white', linewidths=0.8, label=f'Patient {pid}', zorder=5)
    corr = _safe_corr(x, y)
    if np.isfinite(corr):
        ax.text(0.98, 0.04, f'Cohort corr$(\\alpha,\\beta)$ = {corr:.2f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.95})
    ax.set_xlabel('Posterior median of $\\alpha$ (a.u.$^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Posterior median of $\\beta$ (a.u.$^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    ax.set_title('Population-level posterior distribution of $\\alpha$ and $\\beta$', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='upper right')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'population_alpha_beta_scatter.pdf')
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved population scatter: {out_path}')
    print('Representative patient IDs:', [labels[i] for i in highlight_idx])
    return out_path
if __name__ == '__main__':
    cfg = CohortSummaryConfig()
    summary_df = build_cohort_summary(cfg)
    if len(summary_df) > 0:
        plot_alpha_beta_population(summary_df, cfg.out_dir, highlight_patient_ids=['100067', '100082', '100153', '100166'])
