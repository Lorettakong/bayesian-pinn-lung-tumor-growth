import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Datasets import load_patient_timeseries, get_available_patient_ids, split_patient_train_holdout
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
ONE_COL_WIDTH_IN = 3.5
TWO_COL_WIDTH_IN = 7.16
PREDICTIVE_HEIGHT_IN = 2.65
PANEL_HEIGHT_IN = 5.55
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9.5
SUPTITLE_SIZE = 10.5
TICK_SIZE = 8.5
LEGEND_SIZE = 8
PANEL_CAPTION_SIZE = 8.5
ANNOTATION_SIZE = 8

@dataclass
class HMCConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    map_dir: str = 'outputs'
    use_map_init: bool = True
    nls_dir: str = 'outputs_nls'
    use_nls_init: bool = False
    sigma_d: float = 0.2
    mu_log_alpha: float = math.log(0.2)
    sd_log_alpha: float = 0.5
    mu_log_beta: float = math.log(0.05)
    sd_log_beta: float = 0.5
    y0_prior_sd: float = 0.5
    step_size: float = 0.01
    num_leapfrog_steps: int = 20
    num_samples: int = 400
    burn_in: int = 100
    thin: int = 1
    mass_scale: float = 1.0
    use_obs_init: bool = True
    seed: int = 42
    out_dir: str = 'outputs_hmc'
    hist_bins: int = 30
    overlay_prior: bool = True
    scatter_alpha: float = 0.5
    scatter_size: float = 14.0

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
    return 't'

def kinetic_energy(momentums: List[torch.Tensor], mass_scale: float) -> torch.Tensor:
    K = 0.0
    for p in momentums:
        K = K + torch.sum(p * p) / (2.0 * mass_scale)
    return K

def zero_grads(params: List[torch.nn.Parameter]):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

@torch.no_grad()
def copy_params(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]

@torch.no_grad()
def load_params(params: List[torch.nn.Parameter], values: List[torch.Tensor]):
    for p, v in zip(params, values):
        p.copy_(v)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-12)
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))

def lognormal_pdf(x: np.ndarray, mu_log: float, sd_log: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd_log = max(float(sd_log), 1e-12)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    xm = x[mask]
    out[mask] = np.exp(-0.5 * ((np.log(xm) - mu_log) / sd_log) ** 2) / (xm * sd_log * np.sqrt(2.0 * np.pi))
    return out

def safe_quantile(arr: np.ndarray, q: float) -> float:
    if len(arr) == 0:
        return np.nan
    return float(np.quantile(arr, q))

def safe_mean(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))

def safe_std(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return np.nan
    return float(np.std(arr))

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

def load_map_init_if_available(cfg: HMCConfig, patient_id: str) -> dict | None:
    ckpt_path = os.path.join(cfg.map_dir, f'map_model_{patient_id}.pt')
    if not cfg.use_map_init:
        return None
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return {'alpha': float(ckpt.get('alpha', np.nan)), 'beta': float(ckpt.get('beta', np.nan)), 'y0_obs': float(ckpt.get('y0_obs', np.nan)), 't0_obs': float(ckpt.get('t0_obs', np.nan)), 'ckpt_path': ckpt_path}

def load_nls_init_if_available(cfg: HMCConfig, patient_id: str) -> dict | None:
    if not cfg.use_nls_init:
        return None
    summary_path = os.path.join(cfg.nls_dir, f'nls_summary_{patient_id}.csv')
    if not os.path.exists(summary_path):
        return None
    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    return {'alpha': float(row.get('alpha_nls', np.nan)), 'beta': float(row.get('beta_nls', np.nan)), 'y0_obs': float(row.get('y0_nls', np.nan)), 't0_obs': float(row.get('t0', np.nan)), 'summary_path': summary_path}

def infer_time_axis_label(time_col: str) -> str:
    key = str(time_col).lower().strip()
    if key in {'time_days', 'days_from_baseline'}:
        return 'Time since baseline (days)'
    if key == 'time_months':
        return 'Time since baseline (months)'
    if key in {'t_rel', 't_rel_used'}:
        return 'Time since baseline'
    return f'{time_col} (a.u.)'

def infer_rate_unit_label(time_col: str) -> str:
    key = str(time_col).lower().strip()
    if key in {'time_days', 'days_from_baseline'}:
        return 'day$^{-1}$'
    if key == 'time_months':
        return 'month$^{-1}$'
    return 'a.u.$^{-1}$'

def style_axes(ax):
    ax.set_facecolor('#F8FAFC')
    ax.grid(True, alpha=0.18, color='#475569', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    ax.tick_params(labelsize=TICK_SIZE)

def add_panel_caption(ax, text: str):
    ax.text(0.5, -0.24, text, transform=ax.transAxes, ha='center', va='top', fontsize=PANEL_CAPTION_SIZE)

def plot_posterior_hist_with_prior(samples: np.ndarray, param_name: str, patient_id: str, save_path: str, bins: int=30, prior_kind: str | None=None, prior_params: Dict[str, float] | None=None):
    plt.figure(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
    plt.hist(samples, bins=bins, density=True, alpha=0.78, label=f'Posterior {param_name}')
    if prior_kind is not None and prior_params is not None and (len(samples) > 0):
        if prior_kind == 'lognormal':
            x_min = max(1e-08, float(np.min(samples) * 0.8))
            x_max = float(np.max(samples) * 1.2) if np.max(samples) > 0 else x_min + 1.0
        else:
            x_min = float(np.min(samples))
            x_max = float(np.max(samples))
            if abs(x_max - x_min) < 1e-12:
                x_max = x_min + 1.0
        x_grid = np.linspace(x_min, x_max, 400)
        if prior_kind == 'lognormal':
            y_prior = lognormal_pdf(x_grid, mu_log=prior_params['mu_log'], sd_log=prior_params['sd_log'])
            plt.plot(x_grid, y_prior, linestyle='--', linewidth=1.2, label=f'Prior {param_name}')
        elif prior_kind == 'normal':
            y_prior = normal_pdf(x_grid, mu=prior_params['mu'], sd=prior_params['sd'])
            plt.plot(x_grid, y_prior, linestyle='--', linewidth=1.2, label=f'Prior {param_name}')
    plt.xlabel(param_name, fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Density', fontsize=AXIS_LABEL_SIZE)
    plt.title(f'Posterior of {param_name} | patient {patient_id}', fontsize=TITLE_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_alpha_beta_joint(alpha_samples: np.ndarray, beta_samples: np.ndarray, patient_id: str, save_path: str, scatter_alpha: float=0.5, scatter_size: float=14.0):
    plt.figure(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
    plt.scatter(alpha_samples, beta_samples, s=scatter_size, alpha=scatter_alpha, label='Posterior samples')
    if len(alpha_samples) > 0:
        alpha_mean = float(np.mean(alpha_samples))
        beta_mean = float(np.mean(beta_samples))
        plt.scatter([alpha_mean], [beta_mean], marker='x', s=60, linewidths=1.5, label='Posterior mean')
    corr = safe_corr(alpha_samples, beta_samples)
    title = f'Joint posterior of alpha and beta | patient {patient_id}'
    if not np.isnan(corr):
        title += f'\nCorr(alpha, beta) = {corr:.3f}'
    plt.xlabel('alpha', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('beta', fontsize=AXIS_LABEL_SIZE)
    plt.title(title, fontsize=TITLE_SIZE, fontweight='bold')
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def draw_hist_with_prior_on_axis(ax, samples: np.ndarray, posterior_label: str, xlabel: str, prior_kind: str | None=None, prior_params: Dict[str, float] | None=None, bins: int=30):
    style_axes(ax)
    ax.hist(samples, bins=bins, density=True, alpha=0.82, color='#5FA8D3', edgecolor='white', linewidth=0.35, label=posterior_label)
    if prior_kind is not None and prior_params is not None and (len(samples) > 0):
        if prior_kind == 'lognormal':
            x_min = max(1e-08, float(np.min(samples) * 0.8))
            x_max = float(np.max(samples) * 1.2) if np.max(samples) > 0 else x_min + 1.0
        else:
            x_min = float(np.min(samples))
            x_max = float(np.max(samples))
            if abs(x_max - x_min) < 1e-12:
                x_max = x_min + 1.0
        x_grid = np.linspace(x_min, x_max, 400)
        if prior_kind == 'lognormal':
            y_prior = lognormal_pdf(x_grid, mu_log=prior_params['mu_log'], sd_log=prior_params['sd_log'])
        else:
            y_prior = normal_pdf(x_grid, mu=prior_params['mu'], sd=prior_params['sd'])
        ax.plot(x_grid, y_prior, linestyle='--', linewidth=1.1, color='#E9B872', label=posterior_label.replace('Posterior', 'Prior'))
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Density', fontsize=AXIS_LABEL_SIZE)
    ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE)

def draw_alpha_beta_joint_on_axis(ax, alpha_samples: np.ndarray, beta_samples: np.ndarray, alpha_label: str, beta_label: str, scatter_alpha: float=0.5, scatter_size: float=14.0):
    style_axes(ax)
    ax.scatter(alpha_samples, beta_samples, s=scatter_size, alpha=scatter_alpha, color='#5FA8D3', edgecolors='none', label='Posterior samples')
    if len(alpha_samples) > 0:
        alpha_mean = float(np.mean(alpha_samples))
        beta_mean = float(np.mean(beta_samples))
        ax.scatter([alpha_mean], [beta_mean], marker='x', s=65, linewidths=1.6, color='#F97316', label='Posterior mean', zorder=4)
    corr = safe_corr(alpha_samples, beta_samples)
    if not np.isnan(corr):
        ax.text(0.03, 0.97, f'$\\mathrm{{Corr}}(\\alpha,\\beta) = {corr:.3f}$', transform=ax.transAxes, ha='left', va='top', fontsize=ANNOTATION_SIZE, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.92})
    ax.set_xlabel(alpha_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(beta_label, fontsize=AXIS_LABEL_SIZE)
    ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE, loc='upper right')

def plot_posterior_panel_2x2(patient_id: str, alpha_samples: np.ndarray, beta_samples: np.ndarray, y0_samples: np.ndarray, save_path: str, time_col: str, bins: int=30, overlay_prior: bool=True, scatter_alpha: float=0.5, scatter_size: float=14.0, prior_alpha: Dict[str, float] | None=None, prior_beta: Dict[str, float] | None=None, prior_y0: Dict[str, float] | None=None):
    rate_unit = infer_rate_unit_label(time_col)
    alpha_label = f'Growth parameter, $\\alpha$ ({rate_unit})'
    beta_label = f'Inhibition parameter, $\\beta$ ({rate_unit})'
    y0_label = 'Initial log-volume, $y_0$ (unitless)'
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.suptitle(f'Posterior distributions | Patient {patient_id}', fontsize=SUPTITLE_SIZE, fontweight='bold', y=0.97)
    draw_alpha_beta_joint_on_axis(axes[0, 0], alpha_samples=alpha_samples, beta_samples=beta_samples, alpha_label=alpha_label, beta_label=beta_label, scatter_alpha=scatter_alpha, scatter_size=scatter_size)
    add_panel_caption(axes[0, 0], '(a) Joint posterior of $\\alpha$ and $\\beta$')
    draw_hist_with_prior_on_axis(axes[0, 1], samples=alpha_samples, posterior_label='Posterior $\\alpha$', xlabel=alpha_label, prior_kind='lognormal' if overlay_prior else None, prior_params=prior_alpha if overlay_prior else None, bins=bins)
    add_panel_caption(axes[0, 1], '(b) Posterior of $\\alpha$')
    draw_hist_with_prior_on_axis(axes[1, 0], samples=beta_samples, posterior_label='Posterior $\\beta$', xlabel=beta_label, prior_kind='lognormal' if overlay_prior else None, prior_params=prior_beta if overlay_prior else None, bins=bins)
    add_panel_caption(axes[1, 0], '(c) Posterior of $\\beta$')
    draw_hist_with_prior_on_axis(axes[1, 1], samples=y0_samples, posterior_label='Posterior $y_0$', xlabel=y0_label, prior_kind='normal' if overlay_prior else None, prior_params=prior_y0 if overlay_prior else None, bins=bins)
    add_panel_caption(axes[1, 1], '(d) Posterior of $y_0$')
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.985, hspace=0.5, wspace=0.3)
    fig.savefig(save_path, dpi=240, bbox_inches='tight')
    plt.close(fig)

class BayesianGompertz(torch.nn.Module):
    """
    Parameters:
        raw_alpha  -> alpha = softplus(raw_alpha)
        raw_beta   -> beta  = softplus(raw_beta)
        y0         -> unrestricted real number

    Closed-form model in log-volume space:
        y(t) = alpha/beta + (y0 - alpha/beta) * exp(-beta * (t - t0))
    """

    def __init__(self, init_alpha: float=0.2, init_beta: float=0.05, init_y0: float=0.0, eps: float=1e-08):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        self.eps = eps
        self.raw_alpha = torch.nn.Parameter(torch.tensor(self._inv_softplus(init_alpha), dtype=torch.float32))
        self.raw_beta = torch.nn.Parameter(torch.tensor(self._inv_softplus(init_beta), dtype=torch.float32))
        self.y0 = torch.nn.Parameter(torch.tensor(float(init_y0), dtype=torch.float32))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        y = max(float(y), 1e-08)
        if y > 20:
            return y
        return math.log(math.expm1(y))

    @property
    def alpha(self):
        return self.softplus(self.raw_alpha) + self.eps

    @property
    def beta(self):
        return self.softplus(self.raw_beta) + self.eps

    def forward(self, t: torch.Tensor, t0: float) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(dtype=torch.float32)
        beta = self.beta
        alpha = self.alpha
        y_inf = alpha / beta
        y = y_inf + (self.y0 - y_inf) * torch.exp(-beta * (t - t0))
        return y

def get_hmc_target_params(model: BayesianGompertz) -> List[torch.nn.Parameter]:
    return [model.raw_alpha, model.raw_beta, model.y0]

def nll_data_gaussian(y_hat: torch.Tensor, y_obs: torch.Tensor, sigma_d: float) -> torch.Tensor:
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)
    if y_obs.dim() == 1:
        y_obs = y_obs.unsqueeze(1)
    sigma_d = max(float(sigma_d), 1e-08)
    n = y_hat.shape[0]
    resid = y_obs - y_hat
    quad = 0.5 * torch.sum((resid / sigma_d) ** 2)
    log_term = 0.5 * n * math.log(2.0 * math.pi * sigma_d ** 2)
    return quad + log_term

def nll_log_normal_prior_positive(x: torch.Tensor, mu_log: float, sd_log: float) -> torch.Tensor:
    eps = 1e-12
    x = torch.clamp(x, min=eps)
    log_x = torch.log(x)
    sd_log = max(float(sd_log), 1e-08)
    return 0.5 * ((log_x - mu_log) / sd_log) ** 2 + 0.5 * math.log(2.0 * math.pi * sd_log ** 2) + log_x

def nll_normal_prior(x: torch.Tensor, mu: torch.Tensor, sd: float) -> torch.Tensor:
    sd = max(float(sd), 1e-08)
    return 0.5 * ((x - mu) / sd) ** 2 + 0.5 * math.log(2.0 * math.pi * sd ** 2)

def total_energy(model: BayesianGompertz, t_data: torch.Tensor, y_obs: torch.Tensor, t0: float, cfg: HMCConfig) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    y_hat = model(t_data, t0=t0)
    U_data = nll_data_gaussian(y_hat=y_hat, y_obs=y_obs, sigma_d=cfg.sigma_d)
    U_alpha = nll_log_normal_prior_positive(model.alpha, mu_log=cfg.mu_log_alpha, sd_log=cfg.sd_log_alpha)
    U_beta = nll_log_normal_prior_positive(model.beta, mu_log=cfg.mu_log_beta, sd_log=cfg.sd_log_beta)
    y0_center = y_obs[0].view_as(model.y0)
    U_y0 = nll_normal_prior(model.y0, mu=y0_center, sd=cfg.y0_prior_sd)
    U_prior = U_alpha + U_beta + U_y0
    U_total = U_data + U_prior
    return (U_total, {'U_total': U_total, 'U_data': U_data, 'U_prior': U_prior, 'U_alpha': U_alpha, 'U_beta': U_beta, 'U_y0': U_y0, 'y_hat': y_hat})

def grad_U(model: BayesianGompertz, params: List[torch.nn.Parameter], t_data: torch.Tensor, y_obs: torch.Tensor, t0: float, cfg: HMCConfig) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    U, out = total_energy(model, t_data, y_obs, t0, cfg)
    zero_grads(params)
    U.backward()
    return (U, out)

def leapfrog(model: BayesianGompertz, params: List[torch.nn.Parameter], p_list: List[torch.Tensor], t_data: torch.Tensor, y_obs: torch.Tensor, t0: float, cfg: HMCConfig) -> tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    eps = cfg.step_size
    L = cfg.num_leapfrog_steps
    m = cfg.mass_scale
    U, out = grad_U(model, params, t_data, y_obs, t0, cfg)
    with torch.no_grad():
        for p, q in zip(p_list, params):
            p -= 0.5 * eps * q.grad
    for i in range(L):
        with torch.no_grad():
            for q, p in zip(params, p_list):
                q += eps * p / m
        U, out = grad_U(model, params, t_data, y_obs, t0, cfg)
        with torch.no_grad():
            for p, q in zip(p_list, params):
                if i != L - 1:
                    p -= eps * q.grad
                else:
                    p -= 0.5 * eps * q.grad
    with torch.no_grad():
        p_list = [-p for p in p_list]
    return (p_list, out)

def hmc_sample_single_patient(model: BayesianGompertz, t_data: torch.Tensor, y_obs: torch.Tensor, t0: float, cfg: HMCConfig) -> Dict[str, Any]:
    params = get_hmc_target_params(model)
    device = next(model.parameters()).device
    samples_raw_alpha = []
    samples_raw_beta = []
    samples_alpha = []
    samples_beta = []
    samples_y0 = []
    accepted = 0
    U_current, out_current = total_energy(model, t_data, y_obs, t0, cfg)
    for s in range(cfg.num_samples):
        p_current = [torch.randn_like(q, device=device) * math.sqrt(cfg.mass_scale) for q in params]
        q_current = copy_params(params)
        K_current = kinetic_energy(p_current, cfg.mass_scale)
        H_current = U_current.detach() + K_current.detach()
        p_proposed, out_prop = leapfrog(model=model, params=params, p_list=[p.detach().clone() for p in p_current], t_data=t_data, y_obs=y_obs, t0=t0, cfg=cfg)
        U_proposed, out_proposed = total_energy(model, t_data, y_obs, t0, cfg)
        K_proposed = kinetic_energy(p_proposed, cfg.mass_scale)
        H_proposed = U_proposed.detach() + K_proposed.detach()
        log_accept_ratio = (H_current - H_proposed).item()
        accept = math.log(random.random()) < log_accept_ratio
        if accept:
            accepted += 1
            U_current = U_proposed.detach()
            out_current = out_proposed
        else:
            load_params(params, q_current)
            U_current = U_current.detach()
        if s >= cfg.burn_in and (s - cfg.burn_in) % cfg.thin == 0:
            samples_raw_alpha.append(float(model.raw_alpha.detach().cpu().item()))
            samples_raw_beta.append(float(model.raw_beta.detach().cpu().item()))
            samples_alpha.append(float(model.alpha.detach().cpu().item()))
            samples_beta.append(float(model.beta.detach().cpu().item()))
            samples_y0.append(float(model.y0.detach().cpu().item()))
        if (s + 1) % 20 == 0 or s == 0:
            acc_rate = accepted / (s + 1)
            print(f'Iter {s + 1:4d}/{cfg.num_samples} | U={U_current.item():.4f} | alpha={model.alpha.item():.6f} | beta={model.beta.item():.6f} | y0={model.y0.item():.6f} | acc_rate={acc_rate:.3f}')
    return {'raw_alpha': np.array(samples_raw_alpha), 'raw_beta': np.array(samples_raw_beta), 'alpha': np.array(samples_alpha), 'beta': np.array(samples_beta), 'y0': np.array(samples_y0), 'accept_rate': accepted / cfg.num_samples}

def gompertz_log_closed_form(t_grid: np.ndarray, alpha: float, beta: float, y0: float, t0: float) -> np.ndarray:
    beta = max(float(beta), 1e-08)
    alpha = float(alpha)
    y0 = float(y0)
    t0 = float(t0)
    y_inf = alpha / beta
    return y_inf + (y0 - y_inf) * np.exp(-beta * (t_grid - t0))

def predict_from_posterior(alpha_samples: np.ndarray, beta_samples: np.ndarray, y0_samples: np.ndarray, t_grid: np.ndarray, t0: float) -> Dict[str, np.ndarray]:
    y_preds = []
    v_preds = []
    for a, b, y0 in zip(alpha_samples, beta_samples, y0_samples):
        y = gompertz_log_closed_form(t_grid=t_grid, alpha=float(a), beta=float(b), y0=float(y0), t0=float(t0))
        v = np.exp(y)
        y_preds.append(y)
        v_preds.append(v)
    y_preds = np.asarray(y_preds)
    v_preds = np.asarray(v_preds)
    return {'y_mean': y_preds.mean(axis=0), 'y_std': y_preds.std(axis=0), 'y_q025': np.quantile(y_preds, 0.025, axis=0), 'y_q975': np.quantile(y_preds, 0.975, axis=0), 'v_mean': v_preds.mean(axis=0), 'v_std': v_preds.std(axis=0), 'v_q025': np.quantile(v_preds, 0.025, axis=0), 'v_q975': np.quantile(v_preds, 0.975, axis=0)}

def run_hmc_single_patient(cfg: HMCConfig, patient_id: str) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patient_id = str(patient_id)
    print('\n' + '=' * 70)
    print(f'Running HMC for patient {patient_id}')
    print(f'Using device: {device}')
    split = split_patient_train_holdout(cfg.csv_path, patient_id=patient_id, keep_only_ok=True, holdout_rule='last_timepoint')
    patient_id_loaded = str(split['patient_id'])
    df = split['full_df'].copy()
    _, t_data_full, y_obs_full, _ = load_patient_timeseries(cfg.csv_path, patient_id=patient_id_loaded, keep_only_ok=True)
    time_col = get_time_column_name(df)
    time_axis_label = infer_time_axis_label(time_col)
    rate_unit = infer_rate_unit_label(time_col)
    t_data = split['t_train'].to(device=device, dtype=torch.float32)
    y_obs = split['y_train'].to(device=device, dtype=torch.float32)
    t_data_full = t_data_full.to(device=device, dtype=torch.float32)
    y_obs_full = y_obs_full.to(device=device, dtype=torch.float32)
    t_train_np = t_data.detach().cpu().numpy().flatten()
    y_train_np = y_obs.detach().cpu().numpy().flatten()
    t_obs_np = t_data_full.detach().cpu().numpy().flatten()
    y_obs_np = y_obs_full.detach().cpu().numpy().flatten()
    v_obs_np = np.exp(y_obs_np)
    t0 = float(t_train_np[0])
    t_min = float(t_obs_np.min())
    t_max = float(t_obs_np.max())
    nls_init = load_nls_init_if_available(cfg, patient_id)
    map_init = load_map_init_if_available(cfg, patient_id)
    if nls_init is not None:
        init_alpha = float(nls_init['alpha'])
        init_beta = float(nls_init['beta'])
        init_y0 = float(nls_init['y0_obs']) if not np.isnan(nls_init['y0_obs']) else float(y_obs_np[0])
        init_source = 'nls'
        init_source_path = str(nls_init['summary_path'])
        print(f'Using NLS init from: {init_source_path}')
    elif map_init is not None:
        init_alpha = float(map_init['alpha'])
        init_beta = float(map_init['beta'])
        init_y0 = float(map_init['y0_obs']) if not np.isnan(map_init['y0_obs']) else float(y_obs_np[0])
        init_source = 'map'
        init_source_path = str(map_init['ckpt_path'])
        print(f'Using MAP init from: {init_source_path}')
    else:
        init_alpha = math.exp(cfg.mu_log_alpha)
        init_beta = math.exp(cfg.mu_log_beta)
        init_y0 = float(y_obs_np[0]) if cfg.use_obs_init else 0.0
        init_source = 'prior_obs'
        init_source_path = ''
        print('No NLS/MAP init found. Falling back to prior/observation init.')
    model = BayesianGompertz(init_alpha=init_alpha, init_beta=init_beta, init_y0=init_y0).to(device)
    hmc_results = hmc_sample_single_patient(model=model, t_data=t_data, y_obs=y_obs, t0=t0, cfg=cfg)
    alpha_samples = hmc_results['alpha']
    beta_samples = hmc_results['beta']
    y0_samples = hmc_results['y0']
    accept_rate = hmc_results['accept_rate']
    alpha_beta_corr = safe_corr(alpha_samples, beta_samples)
    posterior_summary = {'patient_id': patient_id, 'accept_rate': float(accept_rate), 'n_posterior_samples': int(len(alpha_samples)), 'alpha_mean': safe_mean(alpha_samples), 'alpha_std': safe_std(alpha_samples), 'alpha_q025': safe_quantile(alpha_samples, 0.025), 'alpha_q50': safe_quantile(alpha_samples, 0.5), 'alpha_q975': safe_quantile(alpha_samples, 0.975), 'beta_mean': safe_mean(beta_samples), 'beta_std': safe_std(beta_samples), 'beta_q025': safe_quantile(beta_samples, 0.025), 'beta_q50': safe_quantile(beta_samples, 0.5), 'beta_q975': safe_quantile(beta_samples, 0.975), 'y0_mean': safe_mean(y0_samples), 'y0_std': safe_std(y0_samples), 'y0_q025': safe_quantile(y0_samples, 0.025), 'y0_q50': safe_quantile(y0_samples, 0.5), 'y0_q975': safe_quantile(y0_samples, 0.975), 'alpha_beta_corr': alpha_beta_corr, 'init_source': init_source, 'init_source_path': init_source_path, 'init_alpha': init_alpha, 'init_beta': init_beta, 'init_y0': init_y0}
    npz_path = os.path.join(cfg.out_dir, f'hmc_samples_{patient_id}.npz')
    np.savez(npz_path, raw_alpha=hmc_results['raw_alpha'], raw_beta=hmc_results['raw_beta'], alpha=alpha_samples, beta=beta_samples, y0=y0_samples, accept_rate=accept_rate, alpha_beta_corr=alpha_beta_corr)
    print(f'Saved raw HMC samples to: {npz_path}')
    if len(alpha_samples) > 0:
        t_grid_np = np.linspace(t_min, t_max, 200)
        posterior_pred = predict_from_posterior(alpha_samples=alpha_samples, beta_samples=beta_samples, y0_samples=y0_samples, t_grid=t_grid_np, t0=t0)
        obs_pred = predict_from_posterior(alpha_samples=alpha_samples, beta_samples=beta_samples, y0_samples=y0_samples, t_grid=t_obs_np, t0=t0)
        rmse_log_post_mean = rmse(y_obs_np, obs_pred['y_mean'])
        rmse_volume_post_mean = rmse(v_obs_np, obs_pred['v_mean'])
        coverage_log_95 = empirical_coverage(y_obs_np, obs_pred['y_q025'], obs_pred['y_q975'])
        coverage_vol_95 = empirical_coverage(v_obs_np, obs_pred['v_q025'], obs_pred['v_q975'])
        posterior_summary['rmse_log_post_mean'] = rmse_log_post_mean
        posterior_summary['rmse_volume_post_mean'] = rmse_volume_post_mean
        posterior_summary['coverage_log_95'] = coverage_log_95
        posterior_summary['coverage_vol_95'] = coverage_vol_95
    else:
        posterior_pred = None
        rmse_log_post_mean = np.nan
        rmse_volume_post_mean = np.nan
        coverage_log_95 = np.nan
        coverage_vol_95 = np.nan
    summary_path = os.path.join(cfg.out_dir, f'hmc_summary_{patient_id}.csv')
    pd.DataFrame([posterior_summary]).to_csv(summary_path, index=False)
    print(f'Saved HMC summary to: {summary_path}')
    if len(alpha_samples) > 0:
        alpha_hist_path = os.path.join(cfg.out_dir, f'hmc_alpha_hist_{patient_id}.pdf')
        plot_posterior_hist_with_prior(samples=alpha_samples, param_name='alpha', patient_id=patient_id, save_path=alpha_hist_path, bins=cfg.hist_bins, prior_kind='lognormal' if cfg.overlay_prior else None, prior_params={'mu_log': cfg.mu_log_alpha, 'sd_log': cfg.sd_log_alpha} if cfg.overlay_prior else None)
        beta_hist_path = os.path.join(cfg.out_dir, f'hmc_beta_hist_{patient_id}.pdf')
        plot_posterior_hist_with_prior(samples=beta_samples, param_name='beta', patient_id=patient_id, save_path=beta_hist_path, bins=cfg.hist_bins, prior_kind='lognormal' if cfg.overlay_prior else None, prior_params={'mu_log': cfg.mu_log_beta, 'sd_log': cfg.sd_log_beta} if cfg.overlay_prior else None)
        y0_hist_path = os.path.join(cfg.out_dir, f'hmc_y0_hist_{patient_id}.pdf')
        plot_posterior_hist_with_prior(samples=y0_samples, param_name='y0', patient_id=patient_id, save_path=y0_hist_path, bins=cfg.hist_bins, prior_kind='normal' if cfg.overlay_prior else None, prior_params={'mu': float(y_obs_np[0]), 'sd': cfg.y0_prior_sd} if cfg.overlay_prior else None)
        alpha_beta_joint_path = os.path.join(cfg.out_dir, f'hmc_alpha_beta_joint_{patient_id}.pdf')
        plot_alpha_beta_joint(alpha_samples=alpha_samples, beta_samples=beta_samples, patient_id=patient_id, save_path=alpha_beta_joint_path, scatter_alpha=cfg.scatter_alpha, scatter_size=cfg.scatter_size)
        posterior_panel_path = os.path.join(cfg.out_dir, f'hmc_posterior_panel_{patient_id}.pdf')
        plot_posterior_panel_2x2(patient_id=patient_id, alpha_samples=alpha_samples, beta_samples=beta_samples, y0_samples=y0_samples, save_path=posterior_panel_path, time_col=time_col, bins=cfg.hist_bins, overlay_prior=cfg.overlay_prior, scatter_alpha=cfg.scatter_alpha, scatter_size=cfg.scatter_size, prior_alpha={'mu_log': cfg.mu_log_alpha, 'sd_log': cfg.sd_log_alpha}, prior_beta={'mu_log': cfg.mu_log_beta, 'sd_log': cfg.sd_log_beta}, prior_y0={'mu': float(y_obs_np[0]), 'sd': cfg.y0_prior_sd})
    else:
        alpha_hist_path = ''
        beta_hist_path = ''
        y0_hist_path = ''
        alpha_beta_joint_path = ''
        posterior_panel_path = ''
    if posterior_pred is not None:
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
        style_axes(ax)
        ax.scatter(t_obs_np, y_obs_np, color='#F97316', edgecolor='white', linewidth=0.5, s=18, label='Observed $\\log V_{\\mathrm{obs}}$', zorder=4)
        ax.plot(t_grid_np, posterior_pred['y_mean'], color='#1D4ED8', linewidth=1.5, label='Posterior mean', zorder=3)
        ax.fill_between(t_grid_np, posterior_pred['y_q025'], posterior_pred['y_q975'], color='#93C5FD', alpha=0.42, label='95% CrI', zorder=1)
        ax.set_xlabel(time_axis_label, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Log tumor volume, $\\log V(t)$ (unitless)', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f'Posterior predictive log-volume | Patient {patient_id}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE)
        fig.tight_layout()
        log_plot_path = os.path.join(cfg.out_dir, f'hmc_log_fit_{patient_id}.pdf')
        fig.savefig(log_plot_path, dpi=240, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, PREDICTIVE_HEIGHT_IN))
        style_axes(ax)
        ax.scatter(t_obs_np, v_obs_np, color='#F97316', edgecolor='white', linewidth=0.5, s=18, label='Observed $V_{\\mathrm{obs}}$', zorder=4)
        ax.plot(t_grid_np, posterior_pred['v_mean'], color='#1D4ED8', linewidth=1.5, label='Posterior mean', zorder=3)
        ax.fill_between(t_grid_np, posterior_pred['v_q025'], posterior_pred['v_q975'], color='#93C5FD', alpha=0.42, label='95% CrI', zorder=1)
        ax.set_xlabel(time_axis_label, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Tumor volume, $V(t)$ (cm$^3$)', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f'Posterior predictive tumor volume | Patient {patient_id}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=LEGEND_SIZE)
        fig.tight_layout()
        volume_plot_path = os.path.join(cfg.out_dir, f'hmc_volume_fit_{patient_id}.pdf')
        fig.savefig(volume_plot_path, dpi=240, bbox_inches='tight')
        plt.close(fig)
        posterior_grid_path = os.path.join(cfg.out_dir, f'hmc_posterior_grid_{patient_id}.csv')
        pd.DataFrame({'t': t_grid_np, 'y_mean': posterior_pred['y_mean'], 'y_std': posterior_pred['y_std'], 'y_q025': posterior_pred['y_q025'], 'y_q975': posterior_pred['y_q975'], 'v_mean': posterior_pred['v_mean'], 'v_std': posterior_pred['v_std'], 'v_q025': posterior_pred['v_q025'], 'v_q975': posterior_pred['v_q975']}).to_csv(posterior_grid_path, index=False)
    else:
        log_plot_path = ''
        volume_plot_path = ''
        posterior_grid_path = ''
    print('HMC done.')
    print('Accept rate:', accept_rate)
    if len(alpha_samples) > 0:
        print(f'alpha mean/std: {alpha_samples.mean():.6f} / {alpha_samples.std():.6f}')
        print(f'beta  mean/std: {beta_samples.mean():.6f} / {beta_samples.std():.6f}')
        print(f'y0    mean/std: {y0_samples.mean():.6f} / {y0_samples.std():.6f}')
        print(f'RMSE log post mean   : {rmse_log_post_mean:.6f}')
        print(f'RMSE volume post mean: {rmse_volume_post_mean:.6f}')
        print(f'Coverage log 95%     : {coverage_log_95:.6f}')
        print(f'Coverage vol 95%     : {coverage_vol_95:.6f}')
        if not np.isnan(alpha_beta_corr):
            print(f'corr(alpha, beta): {alpha_beta_corr:.6f}')
    return {'patient_id': patient_id, 'accept_rate': float(accept_rate), 'n_posterior_samples': int(len(alpha_samples)), 'alpha_mean': posterior_summary['alpha_mean'], 'alpha_std': posterior_summary['alpha_std'], 'alpha_q025': posterior_summary['alpha_q025'], 'alpha_q50': posterior_summary['alpha_q50'], 'alpha_q975': posterior_summary['alpha_q975'], 'beta_mean': posterior_summary['beta_mean'], 'beta_std': posterior_summary['beta_std'], 'beta_q025': posterior_summary['beta_q025'], 'beta_q50': posterior_summary['beta_q50'], 'beta_q975': posterior_summary['beta_q975'], 'y0_mean': posterior_summary['y0_mean'], 'y0_std': posterior_summary['y0_std'], 'y0_q025': posterior_summary['y0_q025'], 'y0_q50': posterior_summary['y0_q50'], 'y0_q975': posterior_summary['y0_q975'], 'alpha_beta_corr': posterior_summary['alpha_beta_corr'], 'rmse_log_post_mean': posterior_summary.get('rmse_log_post_mean', np.nan), 'rmse_volume_post_mean': posterior_summary.get('rmse_volume_post_mean', np.nan), 'coverage_log_95': posterior_summary.get('coverage_log_95', np.nan), 'coverage_vol_95': posterior_summary.get('coverage_vol_95', np.nan), 'samples_npz': npz_path, 'summary_csv': summary_path, 'alpha_hist_pdf': alpha_hist_path, 'beta_hist_pdf': beta_hist_path, 'y0_hist_pdf': y0_hist_path, 'alpha_beta_joint_pdf': alpha_beta_joint_path, 'posterior_panel_pdf': posterior_panel_path, 'log_fit_pdf': log_plot_path, 'volume_fit_pdf': volume_plot_path, 'posterior_grid_csv': posterior_grid_path}

def run_hmc(cfg: HMCConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients to run HMC on:')
    for pid in patient_ids:
        print(' -', pid)
    results = []
    failed = []
    for pid in patient_ids:
        try:
            result = run_hmc_single_patient(cfg, pid)
            results.append(result)
        except Exception as e:
            print(f'\nERROR while running HMC for patient {pid}: {e}')
            failed.append({'patient_id': pid, 'error': str(e)})
    if len(results) > 0:
        summary_all_path = os.path.join(cfg.out_dir, 'hmc_summary_all_patients.csv')
        pd.DataFrame(results).to_csv(summary_all_path, index=False)
        print(f'\nSaved HMC overall summary to: {summary_all_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'hmc_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved HMC failed cases to: {failed_path}')
    print('\nAll HMC runs finished.')
    print(f'Successful patients: {len(results)}')
    print(f'Failed patients    : {len(failed)}')
if __name__ == '__main__':
    config = HMCConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, map_dir='outputs', out_dir='outputs_hmc')
    run_hmc(config)
