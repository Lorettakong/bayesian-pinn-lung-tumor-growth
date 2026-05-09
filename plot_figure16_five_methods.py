import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
TWO_COL_WIDTH_IN = 7.16
PLOT_HEIGHT_IN = 4.2
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

@dataclass
class Figure16Config:
    table_csv: str = 'outputs_validation/table3_five_methods_with_ttests_for_paper.csv'
    out_dir: str = 'outputs_validation'

def style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def _to_float(x):
    if isinstance(x, str) and x.strip().upper() == 'N/A':
        return np.nan
    return float(x)

def build_figure16(cfg: Figure16Config) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    df = pd.read_csv(cfg.table_csv)
    for c in ['RMSE (log)', 'MAE (volume)', 'Mean Rel. CI Width', 'Interval Score', 'Coverage Deviation']:
        if c in df.columns:
            df[c] = df[c].apply(_to_float)
    cost_map = {'Pure Gompertz': 1.0, 'Pure PINN': 8.0, 'Gompertz + Bayesian': 12.0, 'Gompertz + PINN + Bayesian': 18.0, 'PINN + Bayesian': 25.0}
    color_map = {'Gompertz + PINN + Bayesian': '#EA580C', 'PINN + Bayesian': '#2563EB', 'Gompertz + Bayesian': '#0F766E', 'Pure PINN': '#64748B', 'Pure Gompertz': '#111827'}
    marker_map = {'Gompertz + PINN + Bayesian': 'o', 'PINN + Bayesian': 's', 'Gompertz + Bayesian': 'D', 'Pure PINN': '^', 'Pure Gompertz': 'P'}
    df['cost_proxy'] = df['Method'].map(cost_map)
    width = df['Mean Rel. CI Width'].to_numpy(dtype=float)
    size = np.where(np.isfinite(width), 140 + 900 * width, 55)
    df['bubble_size'] = size
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, PLOT_HEIGHT_IN))
    style_axes(ax)
    for _, row in df.iterrows():
        method = row['Method']
        x = float(row['cost_proxy'])
        y = float(row['RMSE (log)'])
        s = float(row['bubble_size'])
        color = color_map.get(method, '#334155')
        marker = marker_map.get(method, 'o')
        ax.scatter(x, y, s=s, color=color, marker=marker, alpha=0.82, edgecolor='white', linewidth=0.8, zorder=3, label=method)
        dx, dy = (0.35, 0.008)
        if method == 'Pure Gompertz':
            dx, dy = (0.45, 0.002)
        elif method == 'Pure PINN':
            dx, dy = (0.45, 0.01)
        elif method == 'PINN + Bayesian':
            dx, dy = (0.45, 0.004)
        elif method == 'Gompertz + Bayesian':
            dx, dy = (0.45, -0.012)
        elif method == 'Gompertz + PINN + Bayesian':
            dx, dy = (0.45, 0.01)
        ax.text(x + dx, y + dy, method, fontsize=ANNOTATION_SIZE, color='#111827', ha='left', va='center')
    ax.annotate('Bubble size = Mean Rel. CI Width\n(Bayesian methods only)', xy=(18.0, 0.109), xytext=(10.5, 0.23), arrowprops=dict(arrowstyle='->', color='#475569', lw=0.9), fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.96})
    ax.set_xlabel('Relative computation cost (proxy)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('RMSE in log-space', fontsize=AXIS_LABEL_SIZE)
    ax.set_title('Efficiency-performance trade-off across five methods', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlim(0, 28)
    ax.set_ylim(0, max(0.3, float(np.nanmax(df['RMSE (log)']) * 1.12)))
    legend_order = ['Gompertz + PINN + Bayesian', 'PINN + Bayesian', 'Gompertz + Bayesian', 'Pure PINN', 'Pure Gompertz']
    legend_handles = []
    for method in legend_order:
        legend_handles.append(Line2D([0], [0], marker=marker_map[method], color='none', markerfacecolor=color_map[method], markeredgecolor='white', markeredgewidth=0.8, markersize=8.5, alpha=0.9, linestyle='None', label=method))
    ax.legend(handles=legend_handles, frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='upper right', borderpad=0.45, labelspacing=0.45, handletextpad=0.5)
    fig.tight_layout()
    out_path = os.path.join(cfg.out_dir, 'figure16_five_methods_tradeoff.pdf')
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    out_csv = os.path.join(cfg.out_dir, 'figure16_five_methods_tradeoff_points.csv')
    df[['Method', 'cost_proxy', 'RMSE (log)', 'Mean Rel. CI Width', 'Interval Score', 'Coverage Deviation']].to_csv(out_csv, index=False)
    print(f'Saved figure: {out_path}')
    print(f'Saved data: {out_csv}')
    return out_path
if __name__ == '__main__':
    build_figure16(Figure16Config())
