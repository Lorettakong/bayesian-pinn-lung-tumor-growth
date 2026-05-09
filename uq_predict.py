import os
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Datasets import load_patient_timeseries, get_available_patient_ids
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
ONE_COL_WIDTH_IN = 3.5
PREDICTIVE_HEIGHT_IN = 2.65
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

@dataclass
class UQConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    hmc_dir: str = 'outputs_hmc'
    out_dir: str = 'outputs_uq'
    n_grid: int = 200
    t_extend_factor: float = 1.5
    clamp_y: float | None = None
    ci_low: float = 2.5
    ci_high: float = 97.5
    alert_future_only: bool = True
    alert_frac_threshold: float = 0.5
    alert_min_mean: float = 1e-06
    use_cuda_if_available: bool = True

def get_hmc_npz_path(cfg: UQConfig, patient_id: str) -> str:
    return os.path.join(cfg.hmc_dir, f'hmc_samples_{patient_id}.npz')

def get_time_column_name(df: pd.DataFrame) -> str:
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in df.columns:
            return col
    return 't'

def summarize_ci(arr: np.ndarray, low=2.5, high=97.5):
    mean = np.mean(arr, axis=0)
    lo = np.percentile(arr, low, axis=0)
    hi = np.percentile(arr, high, axis=0)
    std = np.std(arr, axis=0)
    median = np.median(arr, axis=0)
    return (mean, median, lo, hi, std)

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))

def safe_max(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.max(x))

def infer_time_axis_label(time_col: str) -> str:
    key = str(time_col).lower().strip()
    if key in {'time_days', 'days_from_baseline'}:
        return 'Time since baseline (days)'
    if key == 'time_months':
        return 'Time since baseline (months)'
    if key in {'t_rel', 't_rel_used'}:
        return 'Time since baseline'
    return f'{time_col} (a.u.)'

def style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def draw_prediction_panel(x_grid: np.ndarray, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray, last_obs_t: float, x_label: str, y_label: str, title: str, out_path: str, obs_label: str, coverage: float | None=None):
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
    style_axes(ax)
    future_mask = x_grid > last_obs_t
    if np.any(future_mask):
        ax.axvspan(last_obs_t, float(x_grid[-1]), color='#E2E8F0', alpha=0.45, zorder=0, label='Extrapolation window')
    ax.fill_between(x_grid, lo, hi, color='#93C5FD', alpha=0.42, label='95% CrI', zorder=1)
    ax.plot(x_grid, mean, color='#1D4ED8', linewidth=1.5, label='Posterior mean', zorder=3)
    ax.scatter(x_obs, y_obs, s=18, color='#F97316', edgecolor='white', linewidth=0.5, label=obs_label, zorder=4)
    ax.axvline(last_obs_t, linestyle='--', linewidth=1.1, color='#0F766E', label='Last observation', zorder=2)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=6)
    if coverage is not None and np.isfinite(coverage):
        ax.text(0.98, 0.02, f'Observed-point 95% coverage = {coverage:.2f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, color='#334155', bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.95})
    ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='best', handlelength=2.0, borderpad=0.35, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)

def load_hmc_samples(cfg: UQConfig, patient_id: str):
    npz_path = get_hmc_npz_path(cfg, patient_id)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'HMC samples file not found: {npz_path}\nPlease make sure HMC.py generated hmc_samples_{patient_id}.npz')
    data = np.load(npz_path)
    required_keys = ['alpha', 'beta', 'y0']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"HMC npz for patient {patient_id} is missing key '{key}'. Expected keys: {required_keys}")
    alpha = data['alpha'].astype(np.float64)
    beta = data['beta'].astype(np.float64)
    y0 = data['y0'].astype(np.float64)
    accept_rate = float(data['accept_rate']) if 'accept_rate' in data else np.nan
    if 'alpha_beta_corr' in data:
        corr_alpha_beta = float(data['alpha_beta_corr'])
    elif 'corr_alpha_beta' in data:
        corr_alpha_beta = float(data['corr_alpha_beta'])
    else:
        corr_alpha_beta = safe_corr(alpha, beta)
    return (alpha, beta, y0, accept_rate, corr_alpha_beta)

def gompertz_log_closed_form(t_grid: np.ndarray, alpha: float, beta: float, y0: float, t0: float, clamp_y: float | None=None) -> np.ndarray:
    """
    Closed-form solution of:
        dy/dt = alpha - beta * y

    with initial condition:
        y(t0) = y0
    """
    alpha = float(alpha)
    beta = max(float(beta), 1e-08)
    y0 = float(y0)
    t0 = float(t0)
    y_inf = alpha / beta
    y = y_inf + (y0 - y_inf) * np.exp(-beta * (t_grid - t0))
    if clamp_y is not None:
        y = np.clip(y, -clamp_y, clamp_y)
    return y

def run_uq_single_patient(cfg: UQConfig, patient_id: str) -> Dict[str, Any]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda' if cfg.use_cuda_if_available and torch.cuda.is_available() else 'cpu')
    patient_id = str(patient_id)
    print('\n' + '=' * 70)
    print(f'Running UQ for patient {patient_id}')
    print('Using device:', device)
    patient_id_loaded, t_data, y_obs, df = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    time_col = get_time_column_name(df)
    t_obs_np = t_data.detach().cpu().numpy().reshape(-1).astype(np.float64)
    y_obs_np = y_obs.detach().cpu().numpy().reshape(-1).astype(np.float64)
    V_obs_np = np.exp(y_obs_np)
    t_min = float(np.min(t_obs_np))
    t_max = float(np.max(t_obs_np))
    span = t_max - t_min if t_max > t_min else 1.0
    t_pred_max = t_max + cfg.t_extend_factor * span
    t_grid_np = np.linspace(t_min, t_pred_max, cfg.n_grid).astype(np.float64)
    alpha_samps, beta_samps, y0_samps, accept_rate, corr_alpha_beta = load_hmc_samples(cfg, patient_id)
    S = len(alpha_samps)
    if S == 0:
        raise ValueError(f'No HMC samples found for patient {patient_id}.')
    if not (len(beta_samps) == S and len(y0_samps) == S):
        raise ValueError(f'Sample size mismatch for patient {patient_id}: len(alpha)={len(alpha_samps)}, len(beta)={len(beta_samps)}, len(y0)={len(y0_samps)}')
    print(f'Loaded HMC samples: {S}')
    print(f'alpha mean/std: {alpha_samps.mean():.6f} / {alpha_samps.std():.6f}')
    print(f'beta  mean/std: {beta_samps.mean():.6f} / {beta_samps.std():.6f}')
    print(f'y0    mean/std: {y0_samps.mean():.6f} / {y0_samps.std():.6f}')
    print(f'accept_rate   : {accept_rate:.6f}')
    if not np.isnan(corr_alpha_beta):
        print(f'corr(alpha, beta): {corr_alpha_beta:.6f}')
    t0 = float(t_obs_np[0])
    y_samples = np.zeros((S, cfg.n_grid), dtype=np.float64)
    for s in range(S):
        y_samples[s, :] = gompertz_log_closed_form(t_grid=t_grid_np, alpha=float(alpha_samps[s]), beta=float(beta_samps[s]), y0=float(y0_samps[s]), t0=t0, clamp_y=cfg.clamp_y)
    V_samples = np.exp(y_samples)
    y_mean, y_median, y_lo, y_hi, y_std = summarize_ci(y_samples, cfg.ci_low, cfg.ci_high)
    V_mean, V_median, V_lo, V_hi, V_std = summarize_ci(V_samples, cfg.ci_low, cfg.ci_high)
    y_obs_pred_samples = np.zeros((S, len(t_obs_np)), dtype=np.float64)
    for s in range(S):
        y_obs_pred_samples[s, :] = gompertz_log_closed_form(t_grid=t_obs_np, alpha=float(alpha_samps[s]), beta=float(beta_samps[s]), y0=float(y0_samps[s]), t0=t0, clamp_y=cfg.clamp_y)
    V_obs_pred_samples = np.exp(y_obs_pred_samples)
    y_obs_pred_mean, y_obs_pred_median, y_obs_pred_lo, y_obs_pred_hi, y_obs_pred_std = summarize_ci(y_obs_pred_samples, cfg.ci_low, cfg.ci_high)
    V_obs_pred_mean, V_obs_pred_median, V_obs_pred_lo, V_obs_pred_hi, V_obs_pred_std = summarize_ci(V_obs_pred_samples, cfg.ci_low, cfg.ci_high)
    y_in_ci = ((y_obs_np >= y_obs_pred_lo) & (y_obs_np <= y_obs_pred_hi)).astype(int)
    V_in_ci = ((V_obs_np >= V_obs_pred_lo) & (V_obs_np <= V_obs_pred_hi)).astype(int)
    y_coverage_rate = float(np.mean(y_in_ci))
    V_coverage_rate = float(np.mean(V_in_ci))
    y_resid_mean = y_obs_np - y_obs_pred_mean
    V_resid_mean = V_obs_np - V_obs_pred_mean
    if cfg.alert_future_only:
        future_mask = t_grid_np > t_max
    else:
        future_mask = np.ones_like(t_grid_np, dtype=bool)
    width = V_hi - V_lo
    denom = np.maximum(V_mean, cfg.alert_min_mean)
    ratio = width / denom
    alert_points = np.where(future_mask & (ratio > cfg.alert_frac_threshold))[0]
    if len(alert_points) > 0:
        first_idx = int(alert_points[0])
        alert_triggered = True
        first_alert_t = float(t_grid_np[first_idx])
        print('\nALERT:')
        print(f'95% CI width / mean > {cfg.alert_frac_threshold:.2f}')
        print(f'First trigger at t = {first_alert_t:.6f}')
    else:
        alert_triggered = False
        first_alert_t = np.nan
        print('\nNo alert triggered under current rule.')
    future_ratio = ratio[future_mask]
    future_width = width[future_mask]
    grid_df = pd.DataFrame({'patient_id': patient_id, 't': t_grid_np, 'y_mean': y_mean, 'y_median': y_median, 'y_std': y_std, 'y_q025': y_lo, 'y_q975': y_hi, 'V_mean': V_mean, 'V_median': V_median, 'V_std': V_std, 'V_q025': V_lo, 'V_q975': V_hi, 'ci_width_V': width, 'ci_width_over_mean_V': ratio, 'is_future': (t_grid_np > t_max).astype(int)})
    grid_csv_path = os.path.join(cfg.out_dir, f'uq_grid_{patient_id}.csv')
    grid_df.to_csv(grid_csv_path, index=False)
    obs_pred_df = pd.DataFrame({'patient_id': patient_id, 't_obs': t_obs_np, 'y_obs': y_obs_np, 'y_pred_mean': y_obs_pred_mean, 'y_pred_median': y_obs_pred_median, 'y_pred_std': y_obs_pred_std, 'y_pred_q025': y_obs_pred_lo, 'y_pred_q975': y_obs_pred_hi, 'y_resid_mean': y_resid_mean, 'y_in_95CI': y_in_ci, 'V_obs': V_obs_np, 'V_pred_mean': V_obs_pred_mean, 'V_pred_median': V_obs_pred_median, 'V_pred_std': V_obs_pred_std, 'V_pred_q025': V_obs_pred_lo, 'V_pred_q975': V_obs_pred_hi, 'V_resid_mean': V_resid_mean, 'V_in_95CI': V_in_ci})
    obs_pred_csv_path = os.path.join(cfg.out_dir, f'uq_observed_points_{patient_id}.csv')
    obs_pred_df.to_csv(obs_pred_csv_path, index=False)
    npz_out = os.path.join(cfg.out_dir, f'uq_predictions_{patient_id}.npz')
    np.savez(npz_out, t_grid=t_grid_np, y_mean=y_mean, y_median=y_median, y_lo=y_lo, y_hi=y_hi, y_std=y_std, V_mean=V_mean, V_median=V_median, V_lo=V_lo, V_hi=V_hi, V_std=V_std, t_obs=t_obs_np, y_obs=y_obs_np, V_obs=V_obs_np, alpha_samples=alpha_samps, beta_samples=beta_samps, y0_samples=y0_samps, accept_rate=accept_rate, corr_alpha_beta=corr_alpha_beta, t0=t0)
    y_fig_path = os.path.join(cfg.out_dir, f'uq_y_log_volume_{patient_id}.pdf')
    draw_prediction_panel(x_grid=t_grid_np, mean=y_mean, lo=y_lo, hi=y_hi, x_obs=t_obs_np, y_obs=y_obs_np, last_obs_t=t_max, x_label=infer_time_axis_label(time_col), y_label='Log tumor volume, $\\log V(t)$ (unitless)', title=f'Posterior predictive log-volume | Patient {patient_id}', out_path=y_fig_path, obs_label='Observed $\\log V_{\\mathrm{obs}}$', coverage=y_coverage_rate)
    V_fig_path = os.path.join(cfg.out_dir, f'uq_V_volume_{patient_id}.pdf')
    draw_prediction_panel(x_grid=t_grid_np, mean=V_mean, lo=V_lo, hi=V_hi, x_obs=t_obs_np, y_obs=V_obs_np, last_obs_t=t_max, x_label=infer_time_axis_label(time_col), y_label='Tumor volume, $V(t)$ (cm$^3$)', title=f'Posterior predictive tumor volume | Patient {patient_id}', out_path=V_fig_path, obs_label='Observed $V_{\\mathrm{obs}}$', coverage=V_coverage_rate)
    summary_row = {'patient_id': patient_id, 'n_hmc_samples': int(S), 'accept_rate': float(accept_rate), 'corr_alpha_beta': float(corr_alpha_beta), 't0': t0, 'alpha_mean': float(alpha_samps.mean()), 'alpha_std': float(alpha_samps.std()), 'alpha_q025': float(np.quantile(alpha_samps, 0.025)), 'alpha_q50': float(np.quantile(alpha_samps, 0.5)), 'alpha_q975': float(np.quantile(alpha_samps, 0.975)), 'beta_mean': float(beta_samps.mean()), 'beta_std': float(beta_samps.std()), 'beta_q025': float(np.quantile(beta_samps, 0.025)), 'beta_q50': float(np.quantile(beta_samps, 0.5)), 'beta_q975': float(np.quantile(beta_samps, 0.975)), 'y0_mean': float(y0_samps.mean()), 'y0_std': float(y0_samps.std()), 'y0_q025': float(np.quantile(y0_samps, 0.025)), 'y0_q50': float(np.quantile(y0_samps, 0.5)), 'y0_q975': float(np.quantile(y0_samps, 0.975)), 'last_observation_t': float(t_max), 'prediction_end_t': float(t_pred_max), 'last_observed_V': float(V_obs_np[-1]), 'final_V_mean': float(V_mean[-1]), 'final_V_q025': float(V_lo[-1]), 'final_V_q975': float(V_hi[-1]), 'final_ci_width_V': float(width[-1]), 'final_ci_width_over_mean_V': float(ratio[-1]), 'mean_future_ci_width_V': safe_mean(future_width), 'max_future_ci_width_V': safe_max(future_width), 'mean_future_ci_width_over_mean_V': safe_mean(future_ratio), 'max_future_ci_width_over_mean_V': safe_max(future_ratio), 'y_observed_coverage_95CI': y_coverage_rate, 'V_observed_coverage_95CI': V_coverage_rate, 'alert_triggered': int(alert_triggered), 'first_alert_t': first_alert_t, 'grid_csv': grid_csv_path, 'observed_points_csv': obs_pred_csv_path, 'npz_path': npz_out, 'log_plot_pdf': y_fig_path, 'volume_plot_pdf': V_fig_path}
    summary_csv_path = os.path.join(cfg.out_dir, f'uq_summary_{patient_id}.csv')
    pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)
    print('\nSaved:')
    print(' -', y_fig_path)
    print(' -', V_fig_path)
    print(' -', grid_csv_path)
    print(' -', obs_pred_csv_path)
    print(' -', npz_out)
    print(' -', summary_csv_path)
    print(f'Observed-point 95% CrI coverage (log-space): {y_coverage_rate:.3f}')
    print(f'Observed-point 95% CrI coverage (volume)   : {V_coverage_rate:.3f}')
    if not np.isnan(corr_alpha_beta):
        print(f'corr(alpha, beta): {corr_alpha_beta:.6f}')
    return summary_row

def run_uq(cfg: UQConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients to run UQ on:')
    for pid in patient_ids:
        print(' -', pid)
    results = []
    failed = []
    for pid in patient_ids:
        try:
            result = run_uq_single_patient(cfg, pid)
            results.append(result)
        except Exception as e:
            print(f'\nERROR while running UQ for patient {pid}: {e}')
            failed.append({'patient_id': pid, 'error': str(e)})
    if len(results) > 0:
        summary_all_path = os.path.join(cfg.out_dir, 'uq_summary_all_patients.csv')
        pd.DataFrame(results).to_csv(summary_all_path, index=False)
        print(f'\nSaved overall UQ summary to: {summary_all_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'uq_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved UQ failed cases to: {failed_path}')
    print('\nAll UQ runs finished.')
    print(f'Successful patients: {len(results)}')
    print(f'Failed patients    : {len(failed)}')
if __name__ == '__main__':
    config = UQConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, hmc_dir='outputs_hmc', out_dir='outputs_uq')
    run_uq(config)
