import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
TWO_COL_WIDTH_IN = 7.16
PLOT_HEIGHT_IN = 6.1
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.0
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

@dataclass
class ImprovementPlotConfig:
    validation_dir: str = 'outputs_validation'
    out_dir: str = 'outputs_validation'

def style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def build_improvement_plot(cfg: ImprovementPlotConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.validation_dir, 'holdout_results_all_patients.csv')
    df = pd.read_csv(path)
    df['patient_id'] = df['patient_id'].astype(str)
    df['delta_abs_error_V'] = df['gompertz_bayes_abs_error_V'] - df['bayes_abs_error_V']
    df = df.sort_values('delta_abs_error_V', ascending=False).reset_index(drop=True)
    y = df['delta_abs_error_V'].to_numpy(dtype=float)
    patients = df['patient_id'].tolist()
    ypos = np.arange(len(df))
    colors = np.where(y >= 0.0, '#EA580C', '#1D4ED8')
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, PLOT_HEIGHT_IN), gridspec_kw={'width_ratios': [1.05, 1.15]}, sharey=True)
    ax_all, ax_zoom = axes
    for ax in axes:
        style_axes(ax)
        ax.axvspan(0.0, max(0.008, float(np.max(y) * 1.15)), color='#FFEDD5', alpha=0.55, zorder=0)
        ax.axvspan(min(-0.35, float(np.min(y) * 1.15)), 0.0, color='#DBEAFE', alpha=0.45, zorder=0)
        ax.axvline(0.0, color='#475569', linewidth=1.0, zorder=2)
        ax.hlines(ypos, 0.0, y, color=colors, linewidth=1.2, zorder=2)
        ax.scatter(y, ypos, s=34, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    ax_all.set_xscale('symlog', linthresh=0.001, linscale=1.0)
    better = int(np.sum(y > 0))
    worse = int(np.sum(y < 0))
    mean_delta = float(np.mean(y))
    median_delta = float(np.median(y))
    highlight_ids = {'100111', '100056', '100137', '100081'}
    for i, pid in enumerate(patients):
        if pid in highlight_ids:
            offset = 1.12 if y[i] >= 0 else 0.9
            text_x = y[i] * offset if abs(y[i]) > 1e-09 else 0.00012
            ax_all.text(text_x, ypos[i], pid, ha='left' if y[i] >= 0 else 'right', va='center', fontsize=7.6, color='#111827')
            if pid != '100137':
                ax_zoom.text(text_x, ypos[i], pid, ha='left' if y[i] >= 0 else 'right', va='center', fontsize=7.6, color='#111827')
    ax_all.text(0.00022, len(df) - 1.1, 'Proposed better', fontsize=ANNOTATION_SIZE, color='#9A3412', ha='left', va='center')
    ax_all.text(-0.00022, len(df) - 1.1, 'Gompertz+Bayes better', fontsize=ANNOTATION_SIZE, color='#1D4ED8', ha='right', va='center')
    ax_all.text(0.99, 0.98, f'Proposed better: {better}/30\nGompertz+Bayes better: {worse}/30\nMean delta: {mean_delta:.4f}\nMedian delta: {median_delta:.2e}', transform=ax_all.transAxes, ha='right', va='top', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.24', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.96})
    ax_all.set_xlabel('Delta absolute error (symlog)', fontsize=AXIS_LABEL_SIZE)
    ax_zoom.set_xlabel('Delta absolute error (zoomed linear scale)', fontsize=AXIS_LABEL_SIZE)
    ax_all.set_ylabel('Patients (sorted by improvement)', fontsize=AXIS_LABEL_SIZE)
    ax_all.set_title('All patients', fontsize=TITLE_SIZE, fontweight='bold')
    ax_zoom.set_title('Zoom without extreme outlier', fontsize=TITLE_SIZE, fontweight='bold')
    ax_all.set_yticks(ypos)
    show_ids = []
    for pid in patients:
        if pid in highlight_ids:
            show_ids.append(pid)
        else:
            show_ids.append('')
    ax_all.set_yticklabels(show_ids)
    ax_zoom.set_yticks(ypos)
    ax_zoom.set_yticklabels(show_ids)
    ax_zoom.set_xlim(-0.001, 0.007)
    ax_zoom.axvline(0.0, color='#475569', linewidth=1.0, zorder=2)
    ax_zoom.text(0.98, 0.04, 'Patient 100137 excluded from zoom view\nbecause its negative delta dominates the scale.', transform=ax_zoom.transAxes, ha='right', va='bottom', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.24', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.96})
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor='#EA580C', edgecolor='white', label='Proposed method lower error'), Patch(facecolor='#1D4ED8', edgecolor='white', label='Gompertz + Bayesian lower error')]
    ax_all.legend(handles=legend_handles, frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='lower left')
    fig.suptitle('Paired holdout error improvement in volume-space', fontsize=TITLE_SIZE + 0.5, fontweight='bold', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(cfg.out_dir, 'paired_improvement_plot_volume_abs_error.pdf')
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved paired improvement plot: {out_path}')
    return out_path
if __name__ == '__main__':
    build_improvement_plot(ImprovementPlotConfig())
