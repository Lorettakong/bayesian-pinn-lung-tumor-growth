import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Datasets import load_patient_timeseries, get_available_patient_ids
from Losses import nll_data_gaussian, nll_phys_gaussian
from Physics import build_collocation_points, gompertz_log_residual
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})

@dataclass
class BayesPINNConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    use_time_normalization: bool = True
    n_phys: int = 200
    phys_extend_ratio: float = 0.0
    sigma_d: float = 0.2
    sigma_p: float = 0.5
    mu_log_alpha: float = math.log(0.2)
    sd_log_alpha: float = 0.5
    mu_log_beta: float = math.log(0.05)
    sd_log_beta: float = 0.5
    y0_prior_sd: float = 0.5
    weight_prior_std: float = 1.0
    bias_prior_std: float = 1.0
    lr: float = 0.0005
    epochs: int = 2500
    print_every: int = 200
    seed: int = 42
    w_data: float = 1.0
    w_phys: float = 1.0
    w_y0: float = 1.0
    kl_weight: float = 0.001
    kl_warmup_epochs: int = 500
    mc_samples_train: int = 3
    mc_samples_eval: int = 400
    map_dir: str = 'outputs'
    out_dir: str = 'outputs_bpinn'

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

def infer_time_axis_label(time_col: str) -> str:
    key = str(time_col).lower().strip()
    if key in {'time_days', 'days_from_baseline'}:
        return 'Time since baseline (days)'
    if key == 'time_months':
        return 'Time since baseline (months)'
    if key in {'t_rel', 't_rel_used'}:
        return 'Time since baseline'
    return time_col

def kl_gaussian(mu_q: torch.Tensor, rho_q: torch.Tensor, mu_p: float=0.0, std_p: float=1.0) -> torch.Tensor:
    std_q = F.softplus(rho_q) + 1e-08
    var_q = std_q ** 2
    var_p = float(std_p) ** 2
    mu_p = float(mu_p)
    return torch.sum(torch.log(torch.tensor(std_p, device=mu_q.device, dtype=mu_q.dtype) / std_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5)

class BayesianLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, prior_std: float=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        self.weight_sample = self.weight_mu + F.softplus(self.weight_rho) * eps_w
        self.bias_sample = self.bias_mu + F.softplus(self.bias_rho) * eps_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_sample is None or self.bias_sample is None:
            self.resample()
        return F.linear(x, self.weight_sample, self.bias_sample)

    def kl_divergence(self) -> torch.Tensor:
        return kl_gaussian(self.weight_mu, self.weight_rho, std_p=self.prior_std) + kl_gaussian(self.bias_mu, self.bias_rho, std_p=self.prior_std)

class BayesianGompertzPINN(nn.Module):

    def __init__(self, hidden_dim: int, num_hidden_layers: int, mu_log_alpha: float, mu_log_beta: float, sd_log_alpha: float, sd_log_beta: float, weight_prior_std: float=1.0, bias_prior_std: float=1.0, use_time_normalization: bool=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.use_time_normalization = use_time_normalization
        self.weight_prior_std = weight_prior_std
        self.bias_prior_std = bias_prior_std
        self.mu_log_alpha = float(mu_log_alpha)
        self.mu_log_beta = float(mu_log_beta)
        self.sd_log_alpha = float(sd_log_alpha)
        self.sd_log_beta = float(sd_log_beta)
        layers: List[nn.Module] = []
        in_dim = 1
        for _ in range(num_hidden_layers):
            layers.append(BayesianLinear(in_dim, hidden_dim, prior_std=weight_prior_std))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(BayesianLinear(in_dim, 1, prior_std=weight_prior_std))
        self.net = nn.ModuleList(layers)
        self.log_alpha_mu = nn.Parameter(torch.tensor(self.mu_log_alpha, dtype=torch.float32))
        self.log_alpha_rho = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))
        self.log_beta_mu = nn.Parameter(torch.tensor(self.mu_log_beta, dtype=torch.float32))
        self.log_beta_rho = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))
        self.log_alpha_sample = None
        self.log_beta_sample = None

    def initialize_from_map_state(self, map_state: Dict[str, torch.Tensor], alpha_map: float, beta_map: float):
        bayes_linears = [m for m in self.net if isinstance(m, BayesianLinear)]
        linear_idx = 0
        for key, tensor in map_state.items():
            if not key.startswith('net.'):
                continue
            parts = key.split('.')
            layer_idx = int(parts[1])
            attr = parts[2]
            module = None
            count = -1
            for m in self.net:
                if isinstance(m, BayesianLinear):
                    count += 1
                    if count == linear_idx:
                        module = m
                        break
            if module is None:
                continue
            if attr == 'weight':
                module.weight_mu.data.copy_(tensor)
            elif attr == 'bias':
                module.bias_mu.data.copy_(tensor)
                linear_idx += 1
        self.log_alpha_mu.data.fill_(math.log(max(float(alpha_map), 1e-08)))
        self.log_beta_mu.data.fill_(math.log(max(float(beta_map), 1e-08)))

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        if not self.use_time_normalization:
            return t
        t_min = torch.min(t)
        t_max = torch.max(t)
        denom = torch.clamp(t_max - t_min, min=1e-06)
        return 2.0 * (t - t_min) / denom - 1.0

    def resample(self):
        for m in self.net:
            if isinstance(m, BayesianLinear):
                m.resample()
        self.log_alpha_sample = self.log_alpha_mu + F.softplus(self.log_alpha_rho) * torch.randn_like(self.log_alpha_mu)
        self.log_beta_sample = self.log_beta_mu + F.softplus(self.log_beta_rho) * torch.randn_like(self.log_beta_mu)

    @property
    def alpha(self):
        if self.log_alpha_sample is None:
            self.resample()
        return torch.exp(self.log_alpha_sample)

    @property
    def beta(self):
        if self.log_beta_sample is None:
            self.resample()
        return torch.exp(self.log_beta_sample)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = self._normalize_time(t.to(dtype=torch.float32))
        for layer in self.net:
            x = layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_alpha_mu.device, dtype=self.log_alpha_mu.dtype)
        for m in self.net:
            if isinstance(m, BayesianLinear):
                total = total + m.kl_divergence()
        total = total + kl_gaussian(self.log_alpha_mu, self.log_alpha_rho, mu_p=self.mu_log_alpha, std_p=self.sd_log_alpha)
        total = total + kl_gaussian(self.log_beta_mu, self.log_beta_rho, mu_p=self.mu_log_beta, std_p=self.sd_log_beta)
        return total

def load_map_checkpoint_if_available(cfg: BayesPINNConfig, patient_id: str):
    ckpt_path = os.path.join(cfg.map_dir, f'map_model_{patient_id}.pt')
    if not os.path.exists(ckpt_path):
        return None
    return torch.load(ckpt_path, map_location='cpu')

def elbo_components(model: BayesianGompertzPINN, t_data: torch.Tensor, y_obs: torch.Tensor, t_phys: torch.Tensor, t0_obs: torch.Tensor, y0_obs: torch.Tensor, cfg: BayesPINNConfig) -> Dict[str, torch.Tensor]:
    y_hat = model(t_data)
    residual_f = gompertz_log_residual(model, t_phys)
    y0_hat = model(t0_obs)
    U_data = nll_data_gaussian(y_hat, y_obs, cfg.sigma_d)
    U_phys = nll_phys_gaussian(residual_f, cfg.sigma_p)
    U_y0 = gaussian_penalty(y0_hat, y0_obs, cfg.y0_prior_sd).sum()
    U_kl = model.kl_divergence()
    return {'U_data': U_data, 'U_phys': U_phys, 'U_y0': U_y0, 'U_kl': U_kl}

def posterior_predictive(model: BayesianGompertzPINN, t_grid: torch.Tensor, num_samples: int) -> np.ndarray:
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            model.resample()
            preds.append(model(t_grid.to(device=device)).detach().cpu().numpy().reshape(-1))
    return np.asarray(preds, dtype=np.float64)

def train_bpinn_single_patient(cfg: BayesPINNConfig, patient_id: str) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patient_id = str(patient_id)
    patient_id_loaded, t_data, y_obs, df = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    patient_id_loaded = str(patient_id_loaded)
    time_col = get_time_column_name(df)
    t_data = t_data.to(device=device, dtype=torch.float32)
    y_obs = y_obs.to(device=device, dtype=torch.float32)
    t_min = float(t_data.min().item())
    t_max = float(t_data.max().item())
    first_idx = torch.argmin(t_data.view(-1))
    t0_obs = t_data[first_idx].view(1, 1)
    y0_obs = y_obs[first_idx].view(1, 1)
    model = BayesianGompertzPINN(hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers, mu_log_alpha=cfg.mu_log_alpha, mu_log_beta=cfg.mu_log_beta, sd_log_alpha=cfg.sd_log_alpha, sd_log_beta=cfg.sd_log_beta, weight_prior_std=cfg.weight_prior_std, bias_prior_std=cfg.bias_prior_std, use_time_normalization=cfg.use_time_normalization).to(device)
    map_ckpt = load_map_checkpoint_if_available(cfg, patient_id_loaded)
    if map_ckpt is not None and 'model_state' in map_ckpt:
        model.initialize_from_map_state(map_ckpt['model_state'], alpha_map=float(map_ckpt.get('alpha', math.exp(cfg.mu_log_alpha))), beta_map=float(map_ckpt.get('beta', math.exp(cfg.mu_log_beta))))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        t_phys = build_collocation_points(t_obs=t_data.detach().cpu(), num_points=cfg.n_phys, extend_ratio=cfg.phys_extend_ratio).to(device)
        sums = {'U_data': 0.0, 'U_phys': 0.0, 'U_y0': 0.0, 'U_kl': 0.0}
        total_loss = 0.0
        warmup = min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
        kl_scale = cfg.kl_weight * warmup
        for _ in range(cfg.mc_samples_train):
            model.resample()
            comps = elbo_components(model, t_data, y_obs, t_phys, t0_obs, y0_obs, cfg)
            loss = (cfg.w_data * comps['U_data'] + cfg.w_phys * comps['U_phys'] + cfg.w_y0 * comps['U_y0'] + kl_scale * comps['U_kl']) / cfg.mc_samples_train
            loss.backward()
            total_loss = total_loss + float(loss.item())
            for key in sums:
                sums[key] += float(comps[key].item()) / cfg.mc_samples_train
        optimizer.step()
        history.append({'epoch': epoch, 'loss_total': total_loss, 'U_data': sums['U_data'], 'U_phys': sums['U_phys'], 'U_y0': sums['U_y0'], 'U_kl': sums['U_kl'], 'kl_scale': kl_scale})
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % cfg.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | loss={total_loss:.4f} | U_data={sums['U_data']:.4f} | U_phys={sums['U_phys']:.4f} | U_y0={sums['U_y0']:.4f} | U_kl={sums['U_kl']:.4f} | kl_scale={kl_scale:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    t_obs_np = t_data.detach().cpu().numpy().reshape(-1)
    y_obs_np = y_obs.detach().cpu().numpy().reshape(-1)
    v_obs_np = np.exp(y_obs_np)
    t_grid = torch.linspace(t_min, t_max, 200, device=device, dtype=torch.float32).view(-1, 1)
    y_pred_samples_grid = posterior_predictive(model, t_grid, cfg.mc_samples_eval)
    y_pred_samples_obs = posterior_predictive(model, t_data.view(-1, 1), cfg.mc_samples_eval)
    v_pred_samples_grid = np.exp(y_pred_samples_grid)
    v_pred_samples_obs = np.exp(y_pred_samples_obs)
    y_mean_grid, y_median_grid, y_lo_grid, y_hi_grid, y_std_grid = summarize_ci(y_pred_samples_grid)
    v_mean_grid, v_median_grid, v_lo_grid, v_hi_grid, v_std_grid = summarize_ci(v_pred_samples_grid)
    y_mean_obs, y_median_obs, y_lo_obs, y_hi_obs, y_std_obs = summarize_ci(y_pred_samples_obs)
    v_mean_obs, v_median_obs, v_lo_obs, v_hi_obs, v_std_obs = summarize_ci(v_pred_samples_obs)
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
    pred_path = os.path.join(cfg.out_dir, f'bpinn_predictions_{patient_id_loaded}.csv')
    pred_df.to_csv(pred_path, index=False)
    obs_pred_df = pd.DataFrame({'patient_id': patient_id_loaded, 't_obs': t_obs_np, 'y_obs': y_obs_np, 'y_pred_mean': y_mean_obs, 'y_pred_q025': y_lo_obs, 'y_pred_q975': y_hi_obs, 'V_obs': v_obs_np, 'V_pred_mean': v_mean_obs, 'V_pred_q025': v_lo_obs, 'V_pred_q975': v_hi_obs})
    obs_pred_path = os.path.join(cfg.out_dir, f'bpinn_observed_points_{patient_id_loaded}.csv')
    obs_pred_df.to_csv(obs_pred_path, index=False)
    grid_df = pd.DataFrame({'t': t_grid.detach().cpu().numpy().reshape(-1), 'y_mean': y_mean_grid, 'y_q025': y_lo_grid, 'y_q975': y_hi_grid, 'V_mean': v_mean_grid, 'V_q025': v_lo_grid, 'V_q975': v_hi_grid})
    grid_path = os.path.join(cfg.out_dir, f'bpinn_grid_{patient_id_loaded}.csv')
    grid_df.to_csv(grid_path, index=False)
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(cfg.out_dir, f'bpinn_loss_history_{patient_id_loaded}.csv')
    hist_df.to_csv(hist_path, index=False)
    ckpt_path = os.path.join(cfg.out_dir, f'bpinn_model_{patient_id_loaded}.pt')
    torch.save({'patient_id': patient_id_loaded, 'model_state': model.state_dict(), 'config': asdict(cfg), 'best_epoch': best_epoch, 'best_loss': best_loss, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume}, ckpt_path)
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['loss_total'], label='Total ELBO')
    plt.plot(hist_df['epoch'], hist_df['U_data'], label='U_data')
    plt.plot(hist_df['epoch'], hist_df['U_phys'], label='U_phys')
    plt.plot(hist_df['epoch'], hist_df['U_kl'], label='KL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Bayesian PINN training | patient {patient_id_loaded}')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(cfg.out_dir, f'bpinn_loss_curve_{patient_id_loaded}.png')
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    plt.figure()
    plt.scatter(t_obs_np, y_obs_np, label='Observed')
    plt.plot(t_grid.detach().cpu().numpy().reshape(-1), y_mean_grid, label='Posterior mean')
    plt.fill_between(t_grid.detach().cpu().numpy().reshape(-1), y_lo_grid, y_hi_grid, alpha=0.25, label='95% CrI')
    plt.xlabel(infer_time_axis_label(time_col))
    plt.ylabel('Log tumor volume')
    plt.title(f'Bayesian PINN fit (log-space) | patient {patient_id_loaded}')
    plt.legend()
    plt.tight_layout()
    log_plot_path = os.path.join(cfg.out_dir, f'bpinn_log_fit_{patient_id_loaded}.png')
    plt.savefig(log_plot_path, dpi=200)
    plt.close()
    plt.figure()
    plt.scatter(t_obs_np, v_obs_np, label='Observed')
    plt.plot(t_grid.detach().cpu().numpy().reshape(-1), v_mean_grid, label='Posterior mean')
    plt.fill_between(t_grid.detach().cpu().numpy().reshape(-1), v_lo_grid, v_hi_grid, alpha=0.25, label='95% CrI')
    plt.xlabel(infer_time_axis_label(time_col))
    plt.ylabel('Tumor volume')
    plt.title(f'Bayesian PINN fit (volume-space) | patient {patient_id_loaded}')
    plt.legend()
    plt.tight_layout()
    vol_plot_path = os.path.join(cfg.out_dir, f'bpinn_volume_fit_{patient_id_loaded}.png')
    plt.savefig(vol_plot_path, dpi=200)
    plt.close()
    return {'patient_id': patient_id_loaded, 'n_data': int(len(t_obs_np)), 'best_epoch': int(best_epoch), 'best_loss': float(best_loss), 'rmse_log': rmse_log, 'rmse_volume': rmse_volume, 'predictions_csv': pred_path, 'observed_points_csv': obs_pred_path, 'grid_csv': grid_path, 'loss_history_csv': hist_path, 'ckpt_path': ckpt_path, 'loss_curve_png': loss_plot_path, 'log_plot_png': log_plot_path, 'volume_plot_png': vol_plot_path}

def train_bpinn(cfg: BayesPINNConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    else:
        patient_ids = [str(cfg.patient_id)] if cfg.patient_id is not None else [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    results = []
    failed = []
    for pid in patient_ids:
        try:
            print('\n' + '=' * 70)
            print(f'Training true Bayesian PINN for patient {pid}')
            results.append(train_bpinn_single_patient(cfg, str(pid)))
        except Exception as e:
            print(f'\nERROR while training patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if len(results) > 0:
        summary_path = os.path.join(cfg.out_dir, 'bpinn_summary_all_patients.csv')
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f'Saved Bayesian PINN summary to: {summary_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'bpinn_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved failed cases to: {failed_path}')
if __name__ == '__main__':
    config = BayesPINNConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, out_dir='outputs_bpinn')
    train_bpinn(config)
