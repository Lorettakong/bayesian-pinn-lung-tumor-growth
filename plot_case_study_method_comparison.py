import os
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
TWO_COL_WIDTH_IN = 7.16
PANEL_HEIGHT_IN = 5.25
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8
PANEL_CAPTION_SIZE = 8.5

@dataclass
class CaseStudyComparisonConfig:
    proposed_uq_dir: str = 'outputs_uq'
    gompertz_bayes_uq_dir: str = 'outputs_uq_gompertz_bayes'
    validation_dir: str = 'outputs_validation'
    out_dir: str = 'outputs_validation'
    patient_ids: tuple[str, ...] = ('100111', '100056', '100137', '100081')

def style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def add_panel_caption(ax, text: str):
    ax.text(0.5, -0.22, text, transform=ax.transAxes, ha='center', va='top', fontsize=PANEL_CAPTION_SIZE)

def _read_required_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def build_case_study_figure(cfg: CaseStudyComparisonConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    holdout_df = _read_required_csv(os.path.join(cfg.validation_dir, 'holdout_results_all_patients.csv'))
    holdout_df['patient_id'] = holdout_df['patient_id'].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_IN, PANEL_HEIGHT_IN), sharex=False, sharey=False)
    axes = axes.ravel()
    legend_handles = None
    legend_labels = None
    for idx, patient_id in enumerate(cfg.patient_ids):
        ax = axes[idx]
        style_axes(ax)
        proposed_grid = _read_required_csv(os.path.join(cfg.proposed_uq_dir, f'uq_grid_{patient_id}.csv'))
        proposed_obs = _read_required_csv(os.path.join(cfg.proposed_uq_dir, f'uq_observed_points_{patient_id}.csv'))
        gompertz_grid = _read_required_csv(os.path.join(cfg.gompertz_bayes_uq_dir, f'uq_grid_{patient_id}.csv'))
        row = holdout_df.loc[holdout_df['patient_id'] == str(patient_id)].iloc[0]
        t_obs = proposed_obs['t_obs'].to_numpy(dtype=float)
        v_obs = proposed_obs['V_obs'].to_numpy(dtype=float)
        gomp_fill = ax.fill_between(gompertz_grid['t'].to_numpy(dtype=float), gompertz_grid['V_q025'].to_numpy(dtype=float), gompertz_grid['V_q975'].to_numpy(dtype=float), facecolor='#BFDBFE', edgecolor='#1D4ED8', linewidth=0.8, alpha=0.16, zorder=0, label='Gompertz+Bayes 95% CrI')
        ax.plot(gompertz_grid['t'].to_numpy(dtype=float), gompertz_grid['V_q025'].to_numpy(dtype=float), color='#1D4ED8', linewidth=0.9, linestyle=':', zorder=1)
        ax.plot(gompertz_grid['t'].to_numpy(dtype=float), gompertz_grid['V_q975'].to_numpy(dtype=float), color='#1D4ED8', linewidth=0.9, linestyle=':', zorder=1)
        prop_fill = ax.fill_between(proposed_grid['t'].to_numpy(dtype=float), proposed_grid['V_q025'].to_numpy(dtype=float), proposed_grid['V_q975'].to_numpy(dtype=float), color='#FDBA74', alpha=0.3, zorder=2, label='Proposed 95% CrI')
        prop_line, = ax.plot(proposed_grid['t'].to_numpy(dtype=float), proposed_grid['V_mean'].to_numpy(dtype=float), color='#EA580C', linewidth=1.6, zorder=4, label='Proposed mean')
        gomp_line, = ax.plot(gompertz_grid['t'].to_numpy(dtype=float), gompertz_grid['V_mean'].to_numpy(dtype=float), color='#1D4ED8', linewidth=1.5, linestyle='--', zorder=3, label='Gompertz+Bayes mean')
        obs_scatter = ax.scatter(t_obs, v_obs, s=18, color='#111827', edgecolor='white', linewidth=0.5, zorder=4, label='Observed')
        ax.axvline(float(row['holdout_t']), linestyle=':', linewidth=1.0, color='#64748B', zorder=1)
        ax.set_title(f'Patient {patient_id}', fontsize=TITLE_SIZE, fontweight='bold', pad=5)
        ax.set_xlabel('Time since baseline', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Tumor volume ($\\mathrm{cm}^3$)', fontsize=AXIS_LABEL_SIZE)
        ax.text(0.98, 0.03, f"|e| proposed = {float(row['bayes_abs_error_V']):.3f}\n|e| Gompertz+Bayes = {float(row['gompertz_bayes_abs_error_V']):.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, color='#334155', bbox={'boxstyle': 'round,pad=0.22', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.96})
        panel_label = ['(a)', '(b)', '(c)', '(d)'][idx]
        add_panel_caption(ax, f'{panel_label} Comparison for patient {patient_id}')
        if legend_handles is None:
            legend_handles = [obs_scatter, prop_line, gomp_line, prop_fill, gomp_fill]
            legend_labels = ['Observed', 'Proposed mean', 'Gompertz+Bayes mean', 'Proposed 95% CrI', 'Gompertz+Bayes 95% CrI']
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.985, hspace=0.52, wspace=0.28)
    out_path = os.path.join(cfg.out_dir, 'case_study_proposed_vs_gompertz_bayes_best_cases.pdf')
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved case-study comparison figure: {out_path}')
    return out_path
if __name__ == '__main__':
    build_case_study_figure(CaseStudyComparisonConfig())
