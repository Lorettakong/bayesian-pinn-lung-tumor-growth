import os
import math
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Datasets import load_patient_timeseries, get_available_patient_ids
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
ONE_COL_WIDTH_IN = 3.5
PREDICTIVE_HEIGHT_IN = 2.8
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

@dataclass
class CalibrationConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    hmc_dir: str = 'outputs_hmc'
    gompertz_bayes_hmc_dir: str = 'outputs_hmc_gompertz_bayes'
    out_dir: str = 'outputs_validation'
    patient_id: str | None = None
    run_all_patients: bool = True
    nominal_levels: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    use_volume_space: bool = True
    use_log_space: bool = True

def get_time_column_name(df: pd.DataFrame) -> str:
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in df.columns:
            return col
    return 't'

def safe_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))

def gompertz_log_closed_form(t_grid: np.ndarray, alpha: float, beta: float, y0: float, t0: float) -> np.ndarray:
    alpha = float(alpha)
    beta = max(float(beta), 1e-08)
    y0 = float(y0)
    t0 = float(t0)
    y_inf = alpha / beta
    return y_inf + (y0 - y_inf) * np.exp(-beta * (t_grid - t0))

def load_hmc_samples(hmc_dir: str, patient_id: str):
    npz_path = os.path.join(hmc_dir, f'hmc_samples_{patient_id}.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'HMC samples not found: {npz_path}')
    data = np.load(npz_path)
    for key in ['alpha', 'beta', 'y0']:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {npz_path}")
    alpha = data['alpha'].astype(np.float64)
    beta = data['beta'].astype(np.float64)
    y0 = data['y0'].astype(np.float64)
    return (alpha, beta, y0)

def sort_observed_data(t_obs: np.ndarray, y_obs: np.ndarray):
    order = np.argsort(t_obs, kind='mergesort')
    return (t_obs[order], y_obs[order], order)

def empirical_coverage_from_samples(obs: np.ndarray, pred_samples: np.ndarray, nominal_level: float) -> float:
    """
    obs: shape (n_obs,)
    pred_samples: shape (n_samples, n_obs)
    """
    alpha = 1.0 - float(nominal_level)
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)
    lo = np.percentile(pred_samples, q_low, axis=0)
    hi = np.percentile(pred_samples, q_high, axis=0)
    covered = (obs >= lo) & (obs <= hi)
    return float(np.mean(covered))

def calibration_for_single_patient(cfg: CalibrationConfig, patient_id: str) -> List[Dict[str, Any]]:
    patient_id = str(patient_id)
    patient_id_loaded, t_data, y_obs, df = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    patient_id_loaded = str(patient_id_loaded)
    t_obs = t_data.detach().cpu().numpy().reshape(-1).astype(np.float64)
    y_obs = y_obs.detach().cpu().numpy().reshape(-1).astype(np.float64)
    t_obs, y_obs, _ = sort_observed_data(t_obs, y_obs)
    v_obs = np.exp(y_obs)
    alpha_samps, beta_samps, y0_samps = load_hmc_samples(cfg.hmc_dir, patient_id_loaded)
    S = len(alpha_samps)
    if S == 0:
        raise ValueError(f'No posterior samples for patient {patient_id_loaded}')
    t0 = float(t_obs[0])
    y_pred_samples = np.zeros((S, len(t_obs)), dtype=np.float64)
    for s in range(S):
        y_pred_samples[s, :] = gompertz_log_closed_form(t_grid=t_obs, alpha=float(alpha_samps[s]), beta=float(beta_samps[s]), y0=float(y0_samps[s]), t0=t0)
    v_pred_samples = np.exp(y_pred_samples)
    rows = []
    for level in cfg.nominal_levels:
        row = {'patient_id': patient_id_loaded, 'nominal_level': float(level)}
        if cfg.use_log_space:
            row['empirical_coverage_y'] = empirical_coverage_from_samples(obs=y_obs, pred_samples=y_pred_samples, nominal_level=level)
        else:
            row['empirical_coverage_y'] = np.nan
        if cfg.use_volume_space:
            row['empirical_coverage_V'] = empirical_coverage_from_samples(obs=v_obs, pred_samples=v_pred_samples, nominal_level=level)
        else:
            row['empirical_coverage_V'] = np.nan
        rows.append(row)
    return rows

def build_calibration_table(cfg: CalibrationConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients for calibration:')
    for pid in patient_ids:
        print(' -', pid)
    all_rows = []
    failed = []
    for pid in patient_ids:
        try:
            rows = calibration_for_single_patient(cfg, pid)
            all_rows.extend(rows)
        except Exception as e:
            print(f'\nERROR while processing patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if len(all_rows) == 0:
        raise ValueError('No calibration rows were generated.')
    calib_df = pd.DataFrame(all_rows)
    summary_rows = []
    for level in cfg.nominal_levels:
        sub = calib_df.loc[calib_df['nominal_level'] == float(level)].copy()
        summary_rows.append({'nominal_level': float(level), 'mean_empirical_coverage_y': safe_mean(sub['empirical_coverage_y']) if 'empirical_coverage_y' in sub.columns else np.nan, 'mean_empirical_coverage_V': safe_mean(sub['empirical_coverage_V']) if 'empirical_coverage_V' in sub.columns else np.nan, 'n_patients': int(len(sub))})
    summary_df = pd.DataFrame(summary_rows)
    calib_csv = os.path.join(cfg.out_dir, 'calibration_patient_level.csv')
    summary_csv = os.path.join(cfg.out_dir, 'calibration_cohort_summary.csv')
    calib_df.to_csv(calib_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    if len(failed) > 0:
        failed_csv = os.path.join(cfg.out_dir, 'calibration_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        print(f'Saved failed cases: {failed_csv}')
    print(f'Saved patient-level calibration table: {calib_csv}')
    print(f'Saved cohort calibration summary: {summary_csv}')
    return (calib_df, summary_df)

def build_calibration_summary_for_hmc_dir(cfg: CalibrationConfig, hmc_dir: str, method_label: str) -> pd.DataFrame:
    tmp_cfg = CalibrationConfig(csv_path=cfg.csv_path, hmc_dir=hmc_dir, gompertz_bayes_hmc_dir=cfg.gompertz_bayes_hmc_dir, out_dir=cfg.out_dir, patient_id=cfg.patient_id, run_all_patients=cfg.run_all_patients, nominal_levels=cfg.nominal_levels, use_volume_space=cfg.use_volume_space, use_log_space=cfg.use_log_space)
    _, summary_df = build_calibration_table(tmp_cfg)
    summary_df = summary_df.copy()
    summary_df['method'] = method_label
    return summary_df

def plot_calibration_curve(summary_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}
    nominal = summary_df['nominal_level'].to_numpy(dtype=float)

    def style_axes(ax):
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')
        ax.tick_params(labelsize=TICK_SIZE)

    def plot_one(empirical: np.ndarray, title: str, filename: str, key: str):
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
        style_axes(ax)
        ax.plot([0.5, 0.95], [0.5, 0.95], linestyle='--', color='#3E7CB1', linewidth=1.3, label='Ideal calibration', zorder=1)
        ax.plot(nominal, empirical, color='#F97316', linewidth=1.5, marker='s', markersize=4.5, markerfacecolor='#F97316', markeredgecolor='white', markeredgewidth=0.4, label='Empirical coverage', zorder=3)
        ax.fill_between(nominal, nominal, empirical, color='#FDBA74', alpha=0.18, zorder=2)
        mean_abs_gap = float(np.mean(np.abs(empirical - nominal)))
        ax.text(0.97, 0.05, f'Mean abs. gap = {mean_abs_gap:.03f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.95})
        ax.set_xlabel('Nominal coverage', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Empirical coverage', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xlim(0.48, 0.97)
        ax.set_ylim(0.48, 1.0)
        ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='upper left')
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=240, bbox_inches='tight')
        plt.close(fig)
        out_paths[key] = path
        print(f'Saved calibration curve: {path}')
    if 'mean_empirical_coverage_y' in summary_df.columns and summary_df['mean_empirical_coverage_y'].notna().any():
        empirical_y = summary_df['mean_empirical_coverage_y'].to_numpy(dtype=float)
        plot_one(empirical=empirical_y, title='Predictive calibration in log-space', filename='calibration_curve_log_space.pdf', key='log_space_pdf')
    if 'mean_empirical_coverage_V' in summary_df.columns and summary_df['mean_empirical_coverage_V'].notna().any():
        empirical_v = summary_df['mean_empirical_coverage_V'].to_numpy(dtype=float)
        plot_one(empirical=empirical_v, title='Predictive calibration in volume-space', filename='calibration_curve_volume_space.pdf', key='volume_space_pdf')
    return out_paths

def plot_calibration_comparison(summary_frames: List[pd.DataFrame], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}

    def style_axes(ax):
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')
        ax.tick_params(labelsize=TICK_SIZE)
    method_styles = {'Gompertz + PINN + Bayesian': {'color': '#F97316', 'marker': 's', 'label': 'Gompertz + PINN + Bayesian'}, 'Gompertz + Bayesian': {'color': '#1D4ED8', 'marker': 'o', 'label': 'Gompertz + Bayesian'}}

    def plot_one(value_col: str, title: str, filename: str, key: str):
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
        style_axes(ax)
        ax.plot([0.5, 0.95], [0.5, 0.95], linestyle='--', color='#64748B', linewidth=1.3, label='Ideal calibration', zorder=1)
        gap_lines = []
        for frame in summary_frames:
            if value_col not in frame.columns or not frame[value_col].notna().any():
                continue
            method = str(frame['method'].iloc[0])
            style = method_styles.get(method, {'color': '#0F172A', 'marker': 'o', 'label': method})
            nominal = frame['nominal_level'].to_numpy(dtype=float)
            empirical = frame[value_col].to_numpy(dtype=float)
            ax.plot(nominal, empirical, color=style['color'], linewidth=1.5, marker=style['marker'], markersize=4.5, markerfacecolor=style['color'], markeredgecolor='white', markeredgewidth=0.4, label=style['label'], zorder=3)
            mean_abs_gap = float(np.mean(np.abs(empirical - nominal)))
            gap_lines.append(f"{style['label']}: {mean_abs_gap:.03f}")
        if gap_lines:
            ax.text(0.97, 0.05, '\n'.join(gap_lines), transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.95})
        ax.set_xlabel('Nominal coverage', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Empirical coverage', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xlim(0.48, 0.97)
        ax.set_ylim(0.48, 1.0)
        ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='upper left')
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=240, bbox_inches='tight')
        plt.close(fig)
        out_paths[key] = path
        print(f'Saved calibration comparison: {path}')
    plot_one(value_col='mean_empirical_coverage_V', title='Predictive calibration in volume-space', filename='calibration_curve_volume_space_comparison.pdf', key='volume_space_comparison_pdf')
    plot_one(value_col='mean_empirical_coverage_y', title='Predictive calibration in log-space', filename='calibration_curve_log_space_comparison.pdf', key='log_space_comparison_pdf')
    return out_paths
if __name__ == '__main__':
    cfg = CalibrationConfig()
    patient_df, summary_df = build_calibration_table(cfg)
    plot_calibration_curve(summary_df, cfg.out_dir)
    proposed_summary = build_calibration_summary_for_hmc_dir(cfg, hmc_dir=cfg.hmc_dir, method_label='Gompertz + PINN + Bayesian')
    gompertz_bayes_summary = build_calibration_summary_for_hmc_dir(cfg, hmc_dir=cfg.gompertz_bayes_hmc_dir, method_label='Gompertz + Bayesian')
    plot_calibration_comparison([proposed_summary, gompertz_bayes_summary], cfg.out_dir)
