import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Datasets import get_available_patient_ids, load_patient_timeseries, split_patient_train_holdout

@dataclass
class GPBaselineConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    out_dir_gp: str = 'outputs_gp'
    out_dir_bgp: str = 'outputs_bgp'
    grid_points: int = 200
    bgp_num_samples: int = 400
    bgp_burn_in: int = 120
    bgp_proposal_std: tuple[float, float, float] = (0.2, 0.2, 0.25)

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, sigma_f: float, length_scale: float) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=float).reshape(1, -1)
    sqdist = (x1 - x2) ** 2
    return sigma_f ** 2 * np.exp(-0.5 * sqdist / length_scale ** 2)

def neg_log_marginal_likelihood(log_params: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    sigma_f = float(np.exp(log_params[0]))
    length_scale = float(np.exp(log_params[1]))
    sigma_n = float(np.exp(log_params[2]))
    k = rbf_kernel(t, t, sigma_f, length_scale)
    k = k + (sigma_n ** 2 + 1e-08) * np.eye(len(t))
    try:
        l = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return 1000000000000.0
    alpha = np.linalg.solve(l.T, np.linalg.solve(l, y))
    nll = 0.5 * float(y.T @ alpha)
    nll += np.sum(np.log(np.diag(l)))
    nll += 0.5 * len(t) * np.log(2.0 * np.pi)
    return float(nll)

def log_hyperparameter_posterior(log_params: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """
    Unnormalized log-posterior over GP hyperparameters in log-space.
    Posterior = log marginal likelihood + weak Gaussian priors on log-params.
    """
    nll = neg_log_marginal_likelihood(log_params, t, y)
    if not np.isfinite(nll):
        return -np.inf
    mu = np.array([0.0, 0.0, -2.0], dtype=float)
    sd = np.array([2.0, 2.0, 2.0], dtype=float)
    log_prior = -0.5 * np.sum(((log_params - mu) / sd) ** 2) - np.sum(np.log(sd)) - 0.5 * 3 * np.log(2.0 * np.pi)
    return float(-nll + log_prior)

def fit_gp_hyperparams(t: np.ndarray, y: np.ndarray) -> dict:
    init = np.log([1.0, max(np.std(t), 1.0), 0.1])
    bounds = [(-6.0, 6.0), (-6.0, 6.0), (-10.0, 1.0)]
    result = minimize(neg_log_marginal_likelihood, x0=init, args=(t, y), method='L-BFGS-B', bounds=bounds)
    log_params = result.x if result.success else init
    return {'sigma_f': float(np.exp(log_params[0])), 'length_scale': float(np.exp(log_params[1])), 'sigma_n': float(np.exp(log_params[2])), 'log_params': np.asarray(log_params, dtype=float), 'opt_success': int(bool(result.success)), 'opt_fun': float(result.fun) if np.isfinite(result.fun) else np.nan}

def gp_posterior(t_train: np.ndarray, y_train: np.ndarray, t_test: np.ndarray, params: dict):
    sigma_f = params['sigma_f']
    length_scale = params['length_scale']
    sigma_n = params['sigma_n']
    k_tt = rbf_kernel(t_train, t_train, sigma_f, length_scale)
    k_tt = k_tt + (sigma_n ** 2 + 1e-08) * np.eye(len(t_train))
    k_ts = rbf_kernel(t_train, t_test, sigma_f, length_scale)
    k_ss = rbf_kernel(t_test, t_test, sigma_f, length_scale)
    l = np.linalg.cholesky(k_tt)
    alpha = np.linalg.solve(l.T, np.linalg.solve(l, y_train))
    mean = k_ts.T @ alpha
    v = np.linalg.solve(l, k_ts)
    cov = k_ss - v.T @ v
    var = np.maximum(np.diag(cov), 1e-12)
    std = np.sqrt(var)
    return (mean.reshape(-1), std.reshape(-1))

def sample_bgp_hyperparams(cfg: GPBaselineConfig, t: np.ndarray, y: np.ndarray, init_log_params: np.ndarray) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(123)
    current = np.asarray(init_log_params, dtype=float).copy()
    current_lp = log_hyperparameter_posterior(current, t, y)
    proposal_std = np.asarray(cfg.bgp_proposal_std, dtype=float)
    samples = []
    accepts = 0
    total_steps = int(cfg.bgp_num_samples + cfg.bgp_burn_in)
    for step in range(total_steps):
        proposal = current + rng.normal(0.0, proposal_std, size=current.shape)
        proposal_lp = log_hyperparameter_posterior(proposal, t, y)
        log_acc = proposal_lp - current_lp
        if np.log(rng.uniform()) < log_acc:
            current = proposal
            current_lp = proposal_lp
            accepts += 1
        if step >= cfg.bgp_burn_in:
            samples.append(current.copy())
    if len(samples) == 0:
        samples = [np.asarray(init_log_params, dtype=float).copy()]
    return (np.asarray(samples, dtype=float), accepts / max(total_steps, 1))

def bgp_posterior_predictive(cfg: GPBaselineConfig, t_train: np.ndarray, y_train: np.ndarray, t_test: np.ndarray, init_log_params: np.ndarray):
    hyper_samples, accept_rate = sample_bgp_hyperparams(cfg, t_train, y_train, init_log_params)
    mean_draws = []
    std_draws = []
    for lp in hyper_samples:
        params = {'sigma_f': float(np.exp(lp[0])), 'length_scale': float(np.exp(lp[1])), 'sigma_n': float(np.exp(lp[2]))}
        mean_i, std_i = gp_posterior(t_train, y_train, t_test, params)
        mean_draws.append(mean_i)
        std_draws.append(std_i)
    mean_draws = np.asarray(mean_draws, dtype=float)
    std_draws = np.asarray(std_draws, dtype=float)
    pred_mean = mean_draws.mean(axis=0)
    second_moment = (std_draws ** 2 + mean_draws ** 2).mean(axis=0)
    pred_std = np.sqrt(np.maximum(second_moment - pred_mean ** 2, 1e-12))
    q025 = np.quantile(mean_draws, 0.025, axis=0)
    q975 = np.quantile(mean_draws, 0.975, axis=0)
    return (pred_mean, pred_std, q025, q975, accept_rate)

def build_prediction_df(patient_id: str, t_obs: np.ndarray, y_obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    z = 1.96
    y_lo = mean - z * std
    y_hi = mean + z * std
    v_obs = np.exp(y_obs)
    v_mean = np.exp(mean)
    v_lo = np.exp(y_lo)
    v_hi = np.exp(y_hi)
    return pd.DataFrame({'patient_id': str(patient_id), 't_obs': t_obs.astype(float), 'y_obs': y_obs.astype(float), 'y_pred_mean': mean.astype(float), 'y_pred_std': std.astype(float), 'y_pred_q025': y_lo.astype(float), 'y_pred_q975': y_hi.astype(float), 'V_obs': v_obs.astype(float), 'V_pred_mean': v_mean.astype(float), 'V_pred_q025': v_lo.astype(float), 'V_pred_q975': v_hi.astype(float)})

def train_single_patient(cfg: GPBaselineConfig, patient_id: str) -> dict:
    split = split_patient_train_holdout(cfg.csv_path, patient_id=patient_id, keep_only_ok=True, holdout_rule='last_timepoint')
    patient_id = str(split['patient_id'])
    _, t_tensor_full, y_tensor_full, _ = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    t_train = split['t_train'].view(-1).numpy().astype(float)
    y_train = split['y_train'].view(-1).numpy().astype(float)
    t_obs = t_tensor_full.view(-1).numpy().astype(float)
    y_obs = y_tensor_full.view(-1).numpy().astype(float)
    params = fit_gp_hyperparams(t_train, y_train)
    y_mean_obs, y_std_obs = gp_posterior(t_train, y_train, t_obs, params)
    y_mean_obs_bgp, y_std_obs_bgp, y_q025_obs_bgp, y_q975_obs_bgp, accept_rate = bgp_posterior_predictive(cfg, t_train, y_train, t_obs, params['log_params'])
    obs_df = build_prediction_df(patient_id, t_obs, y_obs, y_mean_obs, y_std_obs)
    obs_bgp_df = pd.DataFrame({'patient_id': str(patient_id), 't_obs': t_obs.astype(float), 'y_obs': y_obs.astype(float), 'y_pred_mean': y_mean_obs_bgp.astype(float), 'y_pred_std': y_std_obs_bgp.astype(float), 'y_pred_q025': y_q025_obs_bgp.astype(float), 'y_pred_q975': y_q975_obs_bgp.astype(float), 'V_obs': np.exp(y_obs).astype(float), 'V_pred_mean': np.exp(y_mean_obs_bgp).astype(float), 'V_pred_q025': np.exp(y_q025_obs_bgp).astype(float), 'V_pred_q975': np.exp(y_q975_obs_bgp).astype(float)})
    t_grid = np.linspace(float(t_obs.min()), float(t_obs.max()), int(cfg.grid_points))
    y_mean_grid, y_std_grid = gp_posterior(t_train, y_train, t_grid, params)
    y_mean_grid_bgp, y_std_grid_bgp, y_q025_grid_bgp, y_q975_grid_bgp, _ = bgp_posterior_predictive(cfg, t_train, y_train, t_grid, params['log_params'])
    z = 1.96
    grid_df = pd.DataFrame({'patient_id': str(patient_id), 't_grid': t_grid, 'y_pred_mean': y_mean_grid, 'y_pred_std': y_std_grid, 'y_pred_q025': y_mean_grid - z * y_std_grid, 'y_pred_q975': y_mean_grid + z * y_std_grid, 'V_pred_mean': np.exp(y_mean_grid), 'V_pred_q025': np.exp(y_mean_grid - z * y_std_grid), 'V_pred_q975': np.exp(y_mean_grid + z * y_std_grid)})
    bgp_grid_df = pd.DataFrame({'patient_id': str(patient_id), 't_grid': t_grid, 'y_pred_mean': y_mean_grid_bgp, 'y_pred_std': y_std_grid_bgp, 'y_pred_q025': y_q025_grid_bgp, 'y_pred_q975': y_q975_grid_bgp, 'V_pred_mean': np.exp(y_mean_grid_bgp), 'V_pred_q025': np.exp(y_q025_grid_bgp), 'V_pred_q975': np.exp(y_q975_grid_bgp)})
    gp_pred_df = obs_df[['patient_id', 't_obs', 'y_obs', 'y_pred_mean', 'V_obs', 'V_pred_mean']].rename(columns={'t_obs': 't_rel_used', 'y_pred_mean': 'y_pred', 'V_pred_mean': 'V_pred'})
    holdout_idx = int(np.argmax(t_obs))
    summary = {'patient_id': str(patient_id), 'sigma_f': params['sigma_f'], 'length_scale': params['length_scale'], 'sigma_n': params['sigma_n'], 'bgp_accept_rate': float(accept_rate), 'opt_success': params['opt_success'], 'holdout_t': float(t_obs[holdout_idx]), 'holdout_y_obs': float(y_obs[holdout_idx]), 'holdout_V_obs': float(np.exp(y_obs[holdout_idx])), 'holdout_y_pred': float(y_mean_obs[holdout_idx]), 'holdout_V_pred': float(np.exp(y_mean_obs[holdout_idx])), 'holdout_y_pred_q025': float(obs_df.loc[holdout_idx, 'y_pred_q025']), 'holdout_y_pred_q975': float(obs_df.loc[holdout_idx, 'y_pred_q975']), 'holdout_V_pred_q025': float(obs_df.loc[holdout_idx, 'V_pred_q025']), 'holdout_V_pred_q975': float(obs_df.loc[holdout_idx, 'V_pred_q975']), 'holdout_error_y': float(y_mean_obs[holdout_idx] - y_obs[holdout_idx]), 'holdout_abs_error_y': float(abs(y_mean_obs[holdout_idx] - y_obs[holdout_idx])), 'holdout_sq_error_y': float((y_mean_obs[holdout_idx] - y_obs[holdout_idx]) ** 2), 'holdout_error_V': float(np.exp(y_mean_obs[holdout_idx]) - np.exp(y_obs[holdout_idx])), 'holdout_abs_error_V': float(abs(np.exp(y_mean_obs[holdout_idx]) - np.exp(y_obs[holdout_idx]))), 'holdout_sq_error_V': float((np.exp(y_mean_obs[holdout_idx]) - np.exp(y_obs[holdout_idx])) ** 2)}
    os.makedirs(cfg.out_dir_gp, exist_ok=True)
    os.makedirs(cfg.out_dir_bgp, exist_ok=True)
    gp_pred_df.to_csv(os.path.join(cfg.out_dir_gp, f'gp_predictions_{patient_id}.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(cfg.out_dir_gp, f'gp_summary_{patient_id}.csv'), index=False)
    obs_bgp_df.to_csv(os.path.join(cfg.out_dir_bgp, f'bgp_observed_points_{patient_id}.csv'), index=False)
    bgp_grid_df.to_csv(os.path.join(cfg.out_dir_bgp, f'bgp_grid_{patient_id}.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(cfg.out_dir_bgp, f'bgp_summary_{patient_id}.csv'), index=False)
    return summary

def run_gp_baselines(cfg: GPBaselineConfig):
    patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True) if cfg.run_all_patients else [str(cfg.patient_id)]
    summaries = []
    failed = []
    for pid in patient_ids:
        try:
            summaries.append(train_single_patient(cfg, str(pid)))
            print(f'Finished GP baselines for patient {pid}')
        except Exception as exc:
            failed.append({'patient_id': str(pid), 'error': str(exc)})
            print(f'ERROR in GP baselines for patient {pid}: {exc}')
    if summaries:
        summary_df = pd.DataFrame(summaries).sort_values('patient_id').reset_index(drop=True)
        os.makedirs(cfg.out_dir_gp, exist_ok=True)
        os.makedirs(cfg.out_dir_bgp, exist_ok=True)
        summary_df.to_csv(os.path.join(cfg.out_dir_gp, 'gp_summary_all_patients.csv'), index=False)
        summary_df.to_csv(os.path.join(cfg.out_dir_bgp, 'bgp_summary_all_patients.csv'), index=False)
    if failed:
        failed_df = pd.DataFrame(failed).sort_values('patient_id').reset_index(drop=True)
        failed_df.to_csv(os.path.join(cfg.out_dir_gp, 'gp_failed_cases.csv'), index=False)
        failed_df.to_csv(os.path.join(cfg.out_dir_bgp, 'bgp_failed_cases.csv'), index=False)
if __name__ == '__main__':
    cfg = GPBaselineConfig()
    run_gp_baselines(cfg)
