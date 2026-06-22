from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parent
main_path = ROOT / 'outputs_nlstt_validation' / 'nlstt_population_residual_patient_level.csv'
naive_path = ROOT / 'outputs_naive_baselines' / 'naive_baseline_predictions_300.csv'
pop_path = ROOT / 'outputs_population_baselines_300' / 'population_baseline_predictions_300.csv'
out_dir = ROOT / 'outputs_enhanced_baseline_stats'
out_dir.mkdir(exist_ok=True)

main = pd.read_csv(main_path)
naive = pd.read_csv(naive_path)
pop = pd.read_csv(pop_path)

df = main[['patient_id','holdout_y_obs','holdout_V_obs',
           'gompertz_bayes_pred_y','gompertz_bayes_pred_V',
           'residual_map_pred_y','residual_map_pred_V',
           'nls_pred_y','nls_pred_V',
           'population_residual_conformal_pred_y','population_residual_conformal_pred_V']].copy()

df = df.merge(naive[['trajectory_id','persistence_pred','linear_volume_pred','log_linear_vdt_pred','fixed_double_pred']],
              left_on='patient_id', right_on='trajectory_id', how='left')
df = df.merge(pop[['trajectory_id','gp_y_pred']], left_on='patient_id', right_on='trajectory_id', how='left', suffixes=('','_pop'))

# predictions in log/volume
preds = {
    'Bayesian Gompertz': ('gompertz_bayes_pred_y','gompertz_bayes_pred_V'),
    'Linear extrapolation': (None,'linear_volume_pred'),
    'Log-linear / VDT extrapolation': (None,'log_linear_vdt_pred'),
    'Gaussian process regression': ('gp_y_pred', None),
    'Residual PINN MAP only': ('residual_map_pred_y','residual_map_pred_V'),
    'NLS Gompertz': ('nls_pred_y','nls_pred_V'),
    'Proposed mechanistic correction framework': ('population_residual_conformal_pred_y','population_residual_conformal_pred_V'),
}

y_true = df['holdout_y_obs'].to_numpy(float)
v_true = df['holdout_V_obs'].to_numpy(float)
prop_y = df['population_residual_conformal_pred_y'].to_numpy(float)
prop_v = df['population_residual_conformal_pred_V'].to_numpy(float)
prop_sq_log_err = (y_true - prop_y)**2
prop_abs_vol_err = np.abs(v_true - prop_v)


def rmse(a):
    return float(np.sqrt(np.mean(a*a)))

def fmt_p(p):
    if np.isnan(p): return '--'
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10**exp)
        return f'{mant:.1f}\\times10^{{{exp}}}'
    return f'{p:.3f}'

def rbc(diff):
    # positive means baseline error larger than proposed error
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return np.nan
    ranks = rankdata(np.abs(diff))
    wpos = ranks[diff > 0].sum()
    wneg = ranks[diff < 0].sum()
    return float((wpos - wneg) / (len(diff) * (len(diff)+1) / 2))

rows=[]
for name,(ycol,vcol) in preds.items():
    if ycol is None:
        vpred = df[vcol].to_numpy(float)
        ypred = np.log(np.maximum(vpred, 1e-12))
    elif vcol is None:
        ypred = df[ycol].to_numpy(float)
        vpred = np.exp(ypred)
    else:
        ypred = df[ycol].to_numpy(float)
        vpred = df[vcol].to_numpy(float)
    sq_log_err = (y_true - ypred)**2
    abs_vol_err = np.abs(v_true - vpred)
    if name == 'Proposed mechanistic correction framework':
        p_log = p_vol = rb_log = rb_vol = np.nan
    else:
        dlog = sq_log_err - prop_sq_log_err
        dvol = abs_vol_err - prop_abs_vol_err
        # one-sided: baseline error > proposed error
        try: p_log = float(wilcoxon(dlog, alternative='greater', zero_method='wilcox').pvalue)
        except Exception: p_log = np.nan
        try: p_vol = float(wilcoxon(dvol, alternative='greater', zero_method='wilcox').pvalue)
        except Exception: p_vol = np.nan
        rb_log = rbc(dlog)
        rb_vol = rbc(dvol)
    rows.append({
        'Method': name,
        'RMSE_log': rmse(y_true-ypred),
        'MAE_volume': float(np.mean(abs_vol_err)),
        'p_log': p_log,
        'p_vol': p_vol,
        'r_rb_log': rb_log,
        'r_rb_vol': rb_vol,
        'p_log_fmt': fmt_p(p_log),
        'p_vol_fmt': fmt_p(p_vol),
    })

res = pd.DataFrame(rows)
res.to_csv(out_dir/'enhanced_baseline_stats.csv', index=False)
print(res[['Method','RMSE_log','MAE_volume','p_log_fmt','p_vol_fmt','r_rb_log','r_rb_vol']].to_string(index=False))
