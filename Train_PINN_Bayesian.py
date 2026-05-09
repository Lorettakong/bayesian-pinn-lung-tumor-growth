import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Datasets import load_patient_timeseries, get_available_patient_ids, split_patient_train_holdout

@dataclass
class PINNBayesConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    use_time_normalization: bool = True
    sigma_d: float = 0.2
    y0_prior_sd: float = 0.5
    weight_prior_std: float = 1.0
    lr: float = 0.0005
    epochs: int = 2500
    print_every: int = 200
    seed: int = 42
    w_data: float = 1.0
    w_y0: float = 1.0
    kl_weight: float = 0.001
    kl_warmup_epochs: int = 500
    mc_samples_train: int = 3
    mc_samples_eval: int = 400
    map_dir: str = 'outputs'
    use_map_init: bool = True
    out_dir: str = 'outputs_pinn_bayes'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_time_column_name(df: pd.DataFrame) -> str:
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in df.columns:
            return col
    raise ValueError('No recognizable time column found in dataframe.')

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def gaussian_penalty(x: torch.Tensor, center: torch.Tensor, sd: float) -> torch.Tensor:
    sd = max(float(sd), 1e-08)
    return 0.5 * ((x - center) / sd) ** 2

def summarize_ci(arr: np.ndarray, low: float=2.5, high: float=97.5):
    mean = np.mean(arr, axis=0)
    lo = np.percentile(arr, low, axis=0)
    hi = np.percentile(arr, high, axis=0)
    std = np.std(arr, axis=0)
    median = np.median(arr, axis=0)
    return (mean, median, lo, hi, std)

def kl_gaussian(mu_q: torch.Tensor, rho_q: torch.Tensor, mu_p: float=0.0, std_p: float=1.0) -> torch.Tensor:
    std_q = F.softplus(rho_q) + 1e-08
    var_q = std_q ** 2
    var_p = float(std_p) ** 2
    mu_p = float(mu_p)
    std_p_t = torch.tensor(float(std_p), device=mu_q.device, dtype=mu_q.dtype)
    return torch.sum(torch.log(std_p_t / std_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5)

class BayesianLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, prior_std: float=1.0):
        super().__init__()
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.weight_sample = None
        self.bias_sample = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    def resample(self):
        self.weight_sample = self.weight_mu + F.softplus(self.weight_rho) * torch.randn_like(self.weight_mu)
        self.bias_sample = self.bias_mu + F.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_sample is None or self.bias_sample is None:
            self.resample()
        return F.linear(x, self.weight_sample, self.bias_sample)

    def kl_divergence(self) -> torch.Tensor:
        return kl_gaussian(self.weight_mu, self.weight_rho, std_p=self.prior_std) + kl_gaussian(self.bias_mu, self.bias_rho, std_p=self.prior_std)

class BayesianPINNAblation(nn.Module):

    def __init__(self, hidden_dim: int, num_hidden_layers: int, weight_prior_std: float=1.0, use_time_normalization: bool=True):
        super().__init__()
        self.use_time_normalization = use_time_normalization
        self.net = nn.ModuleList()
        in_dim = 1
        for _ in range(num_hidden_layers):
            self.net.append(BayesianLinear(in_dim, hidden_dim, prior_std=weight_prior_std))
            self.net.append(nn.Tanh())
            in_dim = hidden_dim
        self.net.append(BayesianLinear(in_dim, 1, prior_std=weight_prior_std))
        self.register_buffer('time_norm_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('time_norm_max', torch.tensor(1.0, dtype=torch.float32))

    def set_time_normalization_stats(self, t_min: float, t_max: float):
        self.time_norm_min.fill_(float(t_min))
        self.time_norm_max.fill_(float(t_max))

    def initialize_from_map_state(self, map_state: Dict[str, torch.Tensor]):
        bayes_linears = [m for m in self.net if isinstance(m, BayesianLinear)]
        for idx, layer in enumerate(bayes_linears):
            w_key = f'net.{2 * idx}.weight'
            b_key = f'net.{2 * idx}.bias'
            if w_key in map_state:
                layer.weight_mu.data.copy_(map_state[w_key])
            if b_key in map_state:
                layer.bias_mu.data.copy_(map_state[b_key])

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        if not self.use_time_normalization:
            return t
        t_min = self.time_norm_min.to(device=t.device, dtype=t.dtype)
        t_max = self.time_norm_max.to(device=t.device, dtype=t.dtype)
        denom = torch.clamp(t_max - t_min, min=1e-06)
        return 2.0 * (t - t_min) / denom - 1.0

    def resample(self):
        for m in self.net:
            if isinstance(m, BayesianLinear):
                m.resample()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = self._normalize_time(t.to(dtype=torch.float32))
        for layer in self.net:
            x = layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        total = None
        for m in self.net:
            if isinstance(m, BayesianLinear):
                term = m.kl_divergence()
                total = term if total is None else total + term
        if total is None:
            total = torch.tensor(0.0, dtype=torch.float32, device=self.time_norm_min.device)
        return total

def load_map_checkpoint_if_available(cfg: PINNBayesConfig, patient_id: str):
    ckpt_path = os.path.join(cfg.map_dir, f'map_model_{patient_id}.pt')
    if not cfg.use_map_init or not os.path.exists(ckpt_path):
        return None
    return torch.load(ckpt_path, map_location='cpu')

def elbo_components(model: BayesianPINNAblation, t_data: torch.Tensor, y_obs: torch.Tensor, t0_obs: torch.Tensor, y0_obs: torch.Tensor, cfg: PINNBayesConfig) -> Dict[str, torch.Tensor]:
    y_hat = model(t_data)
    y0_hat = model(t0_obs)
    resid = y_obs - y_hat
    sigma_d = max(float(cfg.sigma_d), 1e-08)
    U_data = 0.5 * torch.sum((resid / sigma_d) ** 2) + 0.5 * y_hat.shape[0] * math.log(2.0 * math.pi * sigma_d ** 2)
    U_y0 = gaussian_penalty(y0_hat, y0_obs, cfg.y0_prior_sd).sum()
    U_kl = model.kl_divergence()
    return {'U_data': U_data, 'U_y0': U_y0, 'U_kl': U_kl}

def posterior_predictive(model: BayesianPINNAblation, t_grid: torch.Tensor, num_samples: int) -> np.ndarray:
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            model.resample()
            preds.append(model(t_grid.to(device=device)).detach().cpu().numpy().reshape(-1))
    return np.asarray(preds, dtype=np.float64)

def train_single_patient(cfg: PINNBayesConfig, patient_id: str) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = split_patient_train_holdout(cfg.csv_path, patient_id=patient_id, keep_only_ok=True, holdout_rule='last_timepoint')
    patient_id_loaded = str(split['patient_id'])
    df = split['full_df'].copy()
    _, t_data_full, y_obs_full, _ = load_patient_timeseries(cfg.csv_path, patient_id=patient_id_loaded, keep_only_ok=True)
    t_data = split['t_train'].to(device=device, dtype=torch.float32)
    y_obs = split['y_train'].to(device=device, dtype=torch.float32)
    t_data_full = t_data_full.to(device=device, dtype=torch.float32)
    y_obs_full = y_obs_full.to(device=device, dtype=torch.float32)
    t_min = float(t_data.min().item())
    t_max = float(t_data.max().item())
    t_full_min = float(t_data_full.min().item())
    t_full_max = float(t_data_full.max().item())
    first_idx = torch.argmin(t_data.view(-1))
    t0_obs = t_data[first_idx].view(1, 1)
    y0_obs = y_obs[first_idx].view(1, 1)
    model = BayesianPINNAblation(hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers, weight_prior_std=cfg.weight_prior_std, use_time_normalization=cfg.use_time_normalization).to(device)
    model.set_time_normalization_stats(t_min=t_min, t_max=t_max)
    map_ckpt = load_map_checkpoint_if_available(cfg, patient_id_loaded)
    if map_ckpt is not None and 'model_state' in map_ckpt:
        model.initialize_from_map_state(map_ckpt['model_state'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        sums = {'U_data': 0.0, 'U_y0': 0.0, 'U_kl': 0.0}
        total_loss = 0.0
        warmup = min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
        kl_scale = cfg.kl_weight * warmup
        for _ in range(cfg.mc_samples_train):
            model.resample()
            comps = elbo_components(model, t_data, y_obs, t0_obs, y0_obs, cfg)
            loss = (cfg.w_data * comps['U_data'] + cfg.w_y0 * comps['U_y0'] + kl_scale * comps['U_kl']) / cfg.mc_samples_train
            loss.backward()
            total_loss += float(loss.item())
            for key in sums:
                sums[key] += float(comps[key].item()) / cfg.mc_samples_train
        optimizer.step()
        history.append({'epoch': epoch, 'loss_total': total_loss, 'U_data': sums['U_data'], 'U_y0': sums['U_y0'], 'U_kl': sums['U_kl'], 'kl_scale': kl_scale})
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % cfg.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | loss={total_loss:.4f} | U_data={sums['U_data']:.4f} | U_y0={sums['U_y0']:.4f} | U_kl={sums['U_kl']:.4f} | kl_scale={kl_scale:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    t_obs_np = t_data_full.detach().cpu().numpy().reshape(-1)
    y_obs_np = y_obs_full.detach().cpu().numpy().reshape(-1)
    v_obs_np = np.exp(y_obs_np)
    t_grid = torch.linspace(t_full_min, t_full_max, 200, device=device, dtype=torch.float32).view(-1, 1)
    y_pred_samples_grid = posterior_predictive(model, t_grid, cfg.mc_samples_eval)
    y_pred_samples_obs = posterior_predictive(model, t_data_full.view(-1, 1), cfg.mc_samples_eval)
    v_pred_samples_grid = np.exp(y_pred_samples_grid)
    v_pred_samples_obs = np.exp(y_pred_samples_obs)
    y_mean_grid, _, y_lo_grid, y_hi_grid, _ = summarize_ci(y_pred_samples_grid)
    v_mean_grid, _, v_lo_grid, v_hi_grid, _ = summarize_ci(v_pred_samples_grid)
    y_mean_obs, _, y_lo_obs, y_hi_obs, _ = summarize_ci(y_pred_samples_obs)
    v_mean_obs, _, v_lo_obs, v_hi_obs, _ = summarize_ci(v_pred_samples_obs)
    rmse_log = rmse(y_obs_np, y_mean_obs)
    rmse_volume = rmse(v_obs_np, v_mean_obs)
    pred_df = df.copy()
    pred_df['y_obs'] = y_obs_np
    pred_df['y_pred_mean'] = y_mean_obs
    pred_df['y_pred_q025'] = y_lo_obs
    pred_df['y_pred_q975'] = y_hi_obs
    pred_df['V_obs'] = v_obs_np
    pred_df['V_pred_mean'] = v_mean_obs
    pred_df['V_pred_q025'] = v_lo_obs
    pred_df['V_pred_q975'] = v_hi_obs
    pred_path = os.path.join(cfg.out_dir, f'pinn_bayes_predictions_{patient_id_loaded}.csv')
    pred_df.to_csv(pred_path, index=False)
    obs_pred_df = pd.DataFrame({'patient_id': patient_id_loaded, 't_obs': t_obs_np, 'y_obs': y_obs_np, 'y_pred_mean': y_mean_obs, 'y_pred_q025': y_lo_obs, 'y_pred_q975': y_hi_obs, 'V_obs': v_obs_np, 'V_pred_mean': v_mean_obs, 'V_pred_q025': v_lo_obs, 'V_pred_q975': v_hi_obs})
    obs_pred_path = os.path.join(cfg.out_dir, f'pinn_bayes_observed_points_{patient_id_loaded}.csv')
    obs_pred_df.to_csv(obs_pred_path, index=False)
    grid_df = pd.DataFrame({'t': t_grid.detach().cpu().numpy().reshape(-1), 'y_mean': y_mean_grid, 'y_q025': y_lo_grid, 'y_q975': y_hi_grid, 'V_mean': v_mean_grid, 'V_q025': v_lo_grid, 'V_q975': v_hi_grid})
    grid_path = os.path.join(cfg.out_dir, f'pinn_bayes_grid_{patient_id_loaded}.csv')
    grid_df.to_csv(grid_path, index=False)
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(cfg.out_dir, f'pinn_bayes_loss_history_{patient_id_loaded}.csv')
    hist_df.to_csv(hist_path, index=False)
    ckpt_path = os.path.join(cfg.out_dir, f'pinn_bayes_model_{patient_id_loaded}.pt')
    torch.save({'patient_id': patient_id_loaded, 'model_state': model.state_dict(), 'config': asdict(cfg), 'best_epoch': best_epoch, 'best_loss': best_loss, 't_min': t_min, 't_max': t_max, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume}, ckpt_path)
    return {'patient_id': patient_id_loaded, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume, 'best_epoch': best_epoch, 'best_loss': best_loss, 'predictions_csv': pred_path, 'observed_points_csv': obs_pred_path, 'grid_csv': grid_path, 'loss_history_csv': hist_path, 'ckpt_path': ckpt_path}

def train_pinn_bayesian(cfg: PINNBayesConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    rows = []
    failed = []
    for pid in patient_ids:
        print('\n' + '=' * 70)
        print(f'Training PINN + Bayesian ablation for patient {pid}')
        try:
            rows.append(train_single_patient(cfg, str(pid)))
        except Exception as e:
            print(f'ERROR while training patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if rows:
        summary_df = pd.DataFrame(rows).sort_values('patient_id').reset_index(drop=True)
        summary_path = os.path.join(cfg.out_dir, 'pinn_bayes_summary_all_patients.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'Saved summary to: {summary_path}')
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = os.path.join(cfg.out_dir, 'pinn_bayes_failed_patients.csv')
        failed_df.to_csv(failed_path, index=False)
        print(f'Saved failures to: {failed_path}')
if __name__ == '__main__':
    train_pinn_bayesian(PINNBayesConfig())
