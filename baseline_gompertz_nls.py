import os
import math
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from Datasets import load_patient_timeseries, get_available_patient_ids, split_patient_train_holdout

@dataclass
class NLSConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    init_alpha: float = 0.2
    init_beta: float = 0.05
    use_obs_init_for_y0: bool = True
    init_y0: float = 0.0
    lower_alpha: float = 1e-08
    lower_beta: float = 1e-08
    upper_alpha: float = 10.0
    upper_beta: float = 10.0
    lower_y0: float = -20.0
    upper_y0: float = 20.0
    n_grid: int = 200
    out_dir: str = 'outputs_nls'

def get_time_column_name(df: pd.DataFrame) -> str:
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in df.columns:
            return col
    return 't'

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))

def sort_patient_data_by_time(t_np: np.ndarray, y_np: np.ndarray, df: pd.DataFrame, time_col: str):
    """
    Force one consistent ordering across:
    - t_np
    - y_np
    - df

    Primary sort: numerical t_np
    Secondary stable fallback: original order preserved for ties
    """
    t_np = np.asarray(t_np, dtype=float).reshape(-1)
    y_np = np.asarray(y_np, dtype=float).reshape(-1)
    if len(t_np) != len(y_np):
        raise ValueError(f'Length mismatch: len(t_np)={len(t_np)}, len(y_np)={len(y_np)}')
    if len(df) != len(t_np):
        raise ValueError(f'Length mismatch: len(df)={len(df)}, len(t_np)={len(t_np)}')
    order = np.argsort(t_np, kind='mergesort')
    t_sorted = t_np[order]
    y_sorted = y_np[order]
    df_sorted = df.iloc[order].reset_index(drop=True)
    return (t_sorted, y_sorted, df_sorted, order)

def gompertz_log_closed_form(t: np.ndarray, alpha: float, beta: float, y0: float, t0: float) -> np.ndarray:
    """
    Closed-form solution of:
        dy/dt = alpha - beta * y
    with:
        y(t0) = y0
    """
    t = np.asarray(t, dtype=float)
    alpha = float(alpha)
    beta = max(float(beta), 1e-12)
    y0 = float(y0)
    t0 = float(t0)
    y_inf = alpha / beta
    return y_inf + (y0 - y_inf) * np.exp(-beta * (t - t0))

def residuals_nls(params: np.ndarray, t_data: np.ndarray, y_obs: np.ndarray, t0: float) -> np.ndarray:
    alpha, beta, y0 = params
    y_hat = gompertz_log_closed_form(t_data, alpha=alpha, beta=beta, y0=y0, t0=t0)
    return y_hat - y_obs

def fit_nls_single_patient(cfg: NLSConfig, patient_id: str) -> Dict[str, Any]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    patient_id = str(patient_id)
    print('\n' + '=' * 70)
    print(f'Running NLS Gompertz fit for patient {patient_id}')
    split = split_patient_train_holdout(cfg.csv_path, patient_id=patient_id, keep_only_ok=True, holdout_rule='last_timepoint')
    patient_id_loaded = str(split['patient_id'])
    df = split['full_df'].copy()
    train_df = split['train_df'].copy()
    time_col = get_time_column_name(df)
    _, t_data_full, y_obs_full, _ = load_patient_timeseries(cfg.csv_path, patient_id=patient_id_loaded, keep_only_ok=True)
    t_full_np = t_data_full.detach().cpu().numpy().reshape(-1).astype(float)
    y_full_np = y_obs_full.detach().cpu().numpy().reshape(-1).astype(float)
    t_train_np = split['t_train'].detach().cpu().numpy().reshape(-1).astype(float)
    y_train_np = split['y_train'].detach().cpu().numpy().reshape(-1).astype(float)
    t_np, y_np, train_df, _ = sort_patient_data_by_time(t_train_np, y_train_np, train_df, time_col)
    t_full_np, y_full_np, df, _ = sort_patient_data_by_time(t_full_np, y_full_np, df, time_col)
    v_obs_np = np.exp(y_full_np)
    t0 = float(t_np[0])
    y0_obs = float(y_np[0])
    t_min = float(np.min(t_full_np))
    t_max = float(np.max(t_full_np))
    print(f'Using leave-last-out training with n_train = {len(t_np)}, n_full = {len(t_full_np)}')
    show_cols = [c for c in ['patient_id', 'scan', time_col, 'V_obs'] if c in df.columns]
    if len(show_cols) > 0:
        print(df[show_cols])
    if 'scan' in df.columns:
        holdout_scan = str(df.iloc[-1]['scan'])
    else:
        holdout_scan = ''
    init_y0 = y0_obs if cfg.use_obs_init_for_y0 else float(cfg.init_y0)
    x0 = np.array([cfg.init_alpha, cfg.init_beta, init_y0], dtype=float)
    lower = np.array([cfg.lower_alpha, cfg.lower_beta, cfg.lower_y0], dtype=float)
    upper = np.array([cfg.upper_alpha, cfg.upper_beta, cfg.upper_y0], dtype=float)
    result = least_squares(residuals_nls, x0=x0, bounds=(lower, upper), args=(t_np, y_np, t0), method='trf')
    alpha_hat, beta_hat, y0_hat = [float(x) for x in result.x]
    y_pred_obs = gompertz_log_closed_form(t_full_np, alpha_hat, beta_hat, y0_hat, t0)
    v_pred_obs = np.exp(y_pred_obs)
    rmse_log = rmse(y_full_np, y_pred_obs)
    rmse_volume = rmse(v_obs_np, v_pred_obs)
    mae_log = mae(y_full_np, y_pred_obs)
    mae_volume = mae(v_obs_np, v_pred_obs)
    holdout_idx = int(len(t_full_np) - 1)
    holdout_t = float(t_full_np[holdout_idx])
    holdout_y_obs = float(y_full_np[holdout_idx])
    holdout_y_pred = float(y_pred_obs[holdout_idx])
    holdout_y_error = float(holdout_y_pred - holdout_y_obs)
    holdout_abs_error_y = float(abs(holdout_y_error))
    holdout_sq_error_y = float(holdout_y_error ** 2)
    holdout_V_obs = float(v_obs_np[holdout_idx])
    holdout_V_pred = float(v_pred_obs[holdout_idx])
    holdout_error_V = float(holdout_V_pred - holdout_V_obs)
    holdout_abs_error_V = float(abs(holdout_error_V))
    holdout_sq_error_V = float(holdout_error_V ** 2)
    t_grid = np.linspace(t_min, t_max, cfg.n_grid)
    y_grid = gompertz_log_closed_form(t_grid, alpha_hat, beta_hat, y0_hat, t0)
    v_grid = np.exp(y_grid)
    alpha_se = np.nan
    beta_se = np.nan
    y0_se = np.nan
    if result.jac is not None:
        try:
            J = np.asarray(result.jac, dtype=float)
            n = len(y_train_np)
            p = J.shape[1]
            if n > p:
                rss = float(np.sum(result.fun ** 2))
                sigma2 = rss / max(n - p, 1)
                cov = sigma2 * np.linalg.pinv(J.T @ J)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                alpha_se, beta_se, y0_se = [float(s) for s in se]
        except Exception:
            pass
    summary_row = {'patient_id': patient_id_loaded, 'n_data': int(len(t_np)), 'n_scans': int(len(df)), 'time_col': time_col, 't0': t0, 'y0_obs': y0_obs, 't_min': t_min, 't_max': t_max, 'success': int(bool(result.success)), 'status': int(result.status), 'message': str(result.message), 'cost': float(result.cost), 'nfev': int(getattr(result, 'nfev', -1)), 'njev': int(getattr(result, 'njev', -1) if getattr(result, 'njev', None) is not None else -1), 'init_alpha': float(cfg.init_alpha), 'init_beta': float(cfg.init_beta), 'init_y0': float(init_y0), 'alpha_nls': alpha_hat, 'beta_nls': beta_hat, 'y0_nls': y0_hat, 'alpha_se': alpha_se, 'beta_se': beta_se, 'y0_se': y0_se, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume, 'mae_log': mae_log, 'mae_volume': mae_volume, 'holdout_scan': holdout_scan, 'holdout_t': holdout_t, 'holdout_y_obs': holdout_y_obs, 'holdout_y_pred': holdout_y_pred, 'holdout_y_error': holdout_y_error, 'holdout_abs_error_y': holdout_abs_error_y, 'holdout_sq_error_y': holdout_sq_error_y, 'holdout_V_obs': holdout_V_obs, 'holdout_V_pred': holdout_V_pred, 'holdout_error_V': holdout_error_V, 'holdout_abs_error_V': holdout_abs_error_V, 'holdout_sq_error_V': holdout_sq_error_V}
    summary_path = os.path.join(cfg.out_dir, f'nls_summary_{patient_id_loaded}.csv')
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    pred_df = df.copy()
    pred_df['y_obs'] = y_full_np
    pred_df['y_pred'] = y_pred_obs
    pred_df['y_pred_nls'] = y_pred_obs
    pred_df['V_obs'] = v_obs_np
    pred_df['V_pred'] = v_pred_obs
    pred_df['V_pred_nls'] = v_pred_obs
    pred_path = os.path.join(cfg.out_dir, f'nls_predictions_{patient_id_loaded}.csv')
    pred_df.to_csv(pred_path, index=False)
    grid_df = pd.DataFrame({'patient_id': patient_id_loaded, 't': t_grid, 'y_pred': y_grid, 'y_pred_nls': y_grid, 'V_pred': v_grid, 'V_pred_nls': v_grid})
    grid_path = os.path.join(cfg.out_dir, f'nls_grid_{patient_id_loaded}.csv')
    grid_df.to_csv(grid_path, index=False)
    plt.figure()
    plt.scatter(t_full_np, y_full_np, label='Observed')
    plt.plot(t_grid, y_grid, label='NLS fit')
    plt.xlabel(time_col)
    plt.ylabel('y = log(V)')
    plt.title(f'NLS Gompertz fit (log-space) | patient {patient_id_loaded}')
    plt.legend()
    plt.tight_layout()
    log_plot_path = os.path.join(cfg.out_dir, f'nls_log_fit_{patient_id_loaded}.png')
    plt.savefig(log_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    plt.figure()
    plt.scatter(t_full_np, v_obs_np, label='Observed')
    plt.plot(t_grid, v_grid, label='NLS fit')
    plt.xlabel(time_col)
    plt.ylabel('V (cm^3)')
    plt.title(f'NLS Gompertz fit (volume-space) | patient {patient_id_loaded}')
    plt.legend()
    plt.tight_layout()
    volume_plot_path = os.path.join(cfg.out_dir, f'nls_volume_fit_{patient_id_loaded}.png')
    plt.savefig(volume_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved summary to: {summary_path}')
    print(f'Saved predictions to: {pred_path}')
    print(f'Saved grid to: {grid_path}')
    print('NLS done.')
    print(f't0 used   : {t0:.6f}')
    print(f'y0_obs    : {y0_obs:.6f}')
    print(f"alpha_hat : {alpha_hat:.6f} ± {(alpha_se if not math.isnan(alpha_se) else float('nan')):.6f}")
    print(f"beta_hat  : {beta_hat:.6f} ± {(beta_se if not math.isnan(beta_se) else float('nan')):.6f}")
    print(f"y0_hat    : {y0_hat:.6f} ± {(y0_se if not math.isnan(y0_se) else float('nan')):.6f}")
    print(f'RMSE log   : {rmse_log:.6f}')
    print(f'RMSE volume: {rmse_volume:.6f}')
    print(f'MAE log    : {mae_log:.6f}')
    print(f'MAE volume : {mae_volume:.6f}')
    return {**summary_row, 'summary_csv': summary_path, 'predictions_csv': pred_path, 'grid_csv': grid_path, 'log_plot_png': log_plot_path, 'volume_plot_png': volume_plot_path}

def run_nls(cfg: NLSConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients to run NLS on:')
    for pid in patient_ids:
        print(' -', pid)
    results = []
    failed = []
    for pid in patient_ids:
        try:
            out = fit_nls_single_patient(cfg, pid)
            results.append(out)
        except Exception as e:
            print(f'\nERROR while running NLS for patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if len(results) > 0:
        summary_all_path = os.path.join(cfg.out_dir, 'nls_summary_all_patients.csv')
        pd.DataFrame(results).to_csv(summary_all_path, index=False)
        print(f'\nSaved NLS overall summary to: {summary_all_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'nls_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved NLS failed cases to: {failed_path}')
    print('\nAll NLS runs finished.')
    print(f'Successful patients: {len(results)}')
    print(f'Failed patients    : {len(failed)}')
if __name__ == '__main__':
    config = NLSConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, out_dir='outputs_nls')
    run_nls(config)
