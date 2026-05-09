import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ExportTable3Config:
    in_dir: str = 'outputs_validation'
    out_csv: str = 'outputs_validation/table3_seven_methods_with_fullstats_for_paper.csv'

def fmt_num(x, digits=5):
    if pd.isna(x):
        return 'N/A'
    return f'{float(x):.{digits}f}'

def fmt_p(x):
    if pd.isna(x):
        return 'N/A'
    x = float(x)
    if x < 0.001:
        return f'{x:.2e}'
    return f'{x:.4f}'

def get_stat(stats_df: pd.DataFrame, method_a: str, metric: str, col: str):
    sub = stats_df.loc[(stats_df['method_a'] == method_a) & (stats_df['method_b'] == 'bayes') & (stats_df['metric'] == metric)]
    if len(sub) == 0:
        return np.nan
    return sub.iloc[0][col]

def export_table3(cfg: ExportTable3Config):
    summary_path = os.path.join(cfg.in_dir, 'holdout_method_summary_extended.csv')
    stats_path = os.path.join(cfg.in_dir, 'holdout_paired_stats_extended.csv')
    summary_df = pd.read_csv(summary_path)
    stats_df = pd.read_csv(stats_path)
    reverse_map = {'Gompertz + PINN + Bayesian': 'bayes', 'PINN + Bayesian': 'pinn_bayes', 'Pure PINN': 'pinn_det', 'Gompertz + Bayesian': 'gompertz_bayes', 'Bayesian GP': 'bgp', 'Pure GP': 'gp', 'Pure Gompertz': 'nls'}
    rows = []
    for _, row in summary_df.iterrows():
        method_display = row['method_display']
        key = reverse_map[method_display]
        is_bayes = key in {'bayes', 'pinn_bayes', 'gompertz_bayes', 'bgp'}
        is_ref = key == 'bayes'
        rows.append({'Method': method_display, 'RMSE (log)': fmt_num(row['rmse_y_holdout']), 't-test p (RMSE)': 'N/A' if is_ref else fmt_p(get_stat(stats_df, key, 'sq_error_y', 'p_value_ttest')), 'Wilcoxon p (RMSE)': 'N/A' if is_ref else fmt_p(get_stat(stats_df, key, 'sq_error_y', 'p_value_wilcoxon')), 'Effect size dz (RMSE)': 'N/A' if is_ref else fmt_num(get_stat(stats_df, key, 'sq_error_y', 'effect_size_dz')), 'MAE (volume)': fmt_num(row['mae_V_holdout']), 't-test p (MAE)': 'N/A' if is_ref else fmt_p(get_stat(stats_df, key, 'abs_error_V', 'p_value_ttest')), 'Wilcoxon p (MAE)': 'N/A' if is_ref else fmt_p(get_stat(stats_df, key, 'abs_error_V', 'p_value_wilcoxon')), 'Effect size dz (MAE)': 'N/A' if is_ref else fmt_num(get_stat(stats_df, key, 'abs_error_V', 'effect_size_dz')), 'Mean Rel. CI Width': fmt_num(row['mean_relative_interval_width_V']) if is_bayes else 'N/A', 'Interval Score': fmt_num(row['mean_interval_score_V']) if is_bayes else 'N/A', 'Coverage Deviation': fmt_num(row['coverage_minus_target_95CI_V']) if is_bayes else 'N/A'})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(cfg.out_csv, index=False)
    print(f'Saved paper-ready extended Table III to: {cfg.out_csv}')
if __name__ == '__main__':
    export_table3(ExportTable3Config())
