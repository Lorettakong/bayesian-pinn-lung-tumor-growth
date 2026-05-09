import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Datasets import load_patient_timeseries, get_available_patient_ids, split_patient_train_holdout

@dataclass
class PINNDetConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    use_time_normalization: bool = True
    lr: float = 0.001
    epochs: int = 5000
    print_every: int = 200
    seed: int = 42
    sigma_d: float = 0.2
    y0_prior_sd: float = 0.5
    w_data: float = 1.0
    w_y0: float = 1.0
    out_dir: str = 'outputs_pinn_det'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def gaussian_penalty(x: torch.Tensor, center: torch.Tensor, sd: float) -> torch.Tensor:
    sd = max(float(sd), 1e-08)
    return 0.5 * ((x - center) / sd) ** 2

class DeterministicPINNAblation(nn.Module):

    def __init__(self, hidden_dim: int, num_hidden_layers: int, use_time_normalization: bool=True):
        super().__init__()
        self.use_time_normalization = use_time_normalization
        layers = []
        in_dim = 1
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer('time_norm_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('time_norm_max', torch.tensor(1.0, dtype=torch.float32))

    def set_time_normalization_stats(self, t_min: float, t_max: float):
        self.time_norm_min.fill_(float(t_min))
        self.time_norm_max.fill_(float(t_max))

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        if not self.use_time_normalization:
            return t
        t_min = self.time_norm_min.to(device=t.device, dtype=t.dtype)
        t_max = self.time_norm_max.to(device=t.device, dtype=t.dtype)
        denom = torch.clamp(t_max - t_min, min=1e-06)
        return 2.0 * (t - t_min) / denom - 1.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = self._normalize_time(t.to(dtype=torch.float32))
        return self.net(x)

def train_single_patient(cfg: PINNDetConfig, patient_id: str) -> Dict[str, Any]:
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
    model = DeterministicPINNAblation(hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers, use_time_normalization=cfg.use_time_normalization).to(device)
    model.set_time_normalization_stats(t_min=t_min, t_max=t_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    sigma_d = max(float(cfg.sigma_d), 1e-08)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_hat = model(t_data)
        y0_hat = model(t0_obs)
        resid = y_obs - y_hat
        u_data = 0.5 * torch.sum((resid / sigma_d) ** 2) + 0.5 * y_hat.shape[0] * np.log(2.0 * np.pi * sigma_d ** 2)
        u_y0 = gaussian_penalty(y0_hat, y0_obs, cfg.y0_prior_sd).sum()
        loss = cfg.w_data * u_data + cfg.w_y0 * u_y0
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        history.append({'epoch': epoch, 'loss_total': loss_val, 'U_data': float(u_data.item()), 'U_y0': float(u_y0.item())})
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % cfg.print_every == 0 or epoch == 1:
            print(f'Epoch {epoch:5d} | loss={loss_val:.4f} | U_data={float(u_data.item()):.4f} | U_y0={float(u_y0.item()):.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred_obs = model(t_data_full).detach().cpu().numpy().reshape(-1)
        t_grid = torch.linspace(t_full_min, t_full_max, 200, device=device, dtype=torch.float32).view(-1, 1)
        y_pred_grid = model(t_grid).detach().cpu().numpy().reshape(-1)
    t_obs_np = t_data_full.detach().cpu().numpy().reshape(-1)
    y_obs_np = y_obs_full.detach().cpu().numpy().reshape(-1)
    v_obs_np = np.exp(y_obs_np)
    v_pred_obs = np.exp(y_pred_obs)
    v_pred_grid = np.exp(y_pred_grid)
    rmse_log = rmse(y_obs_np, y_pred_obs)
    rmse_volume = rmse(v_obs_np, v_pred_obs)
    pred_df = df.copy()
    pred_df['y_obs'] = y_obs_np
    pred_df['y_pred'] = y_pred_obs
    pred_df['V_obs'] = v_obs_np
    pred_df['V_pred'] = v_pred_obs
    pred_path = os.path.join(cfg.out_dir, f'pinn_det_predictions_{patient_id_loaded}.csv')
    pred_df.to_csv(pred_path, index=False)
    obs_df = pd.DataFrame({'patient_id': patient_id_loaded, 't_obs': t_obs_np, 'y_obs': y_obs_np, 'y_pred_mean': y_pred_obs, 'V_obs': v_obs_np, 'V_pred_mean': v_pred_obs})
    obs_path = os.path.join(cfg.out_dir, f'pinn_det_observed_points_{patient_id_loaded}.csv')
    obs_df.to_csv(obs_path, index=False)
    grid_df = pd.DataFrame({'t': t_grid.detach().cpu().numpy().reshape(-1), 'y_pred': y_pred_grid, 'V_pred': v_pred_grid})
    grid_path = os.path.join(cfg.out_dir, f'pinn_det_grid_{patient_id_loaded}.csv')
    grid_df.to_csv(grid_path, index=False)
    hist_path = os.path.join(cfg.out_dir, f'pinn_det_loss_history_{patient_id_loaded}.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    ckpt_path = os.path.join(cfg.out_dir, f'pinn_det_model_{patient_id_loaded}.pt')
    torch.save({'patient_id': patient_id_loaded, 'model_state': model.state_dict(), 'config': asdict(cfg), 'best_epoch': best_epoch, 'best_loss': best_loss, 't_min': t_min, 't_max': t_max, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume}, ckpt_path)
    return {'patient_id': patient_id_loaded, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume, 'best_epoch': best_epoch, 'best_loss': best_loss, 'predictions_csv': pred_path, 'observed_points_csv': obs_path, 'grid_csv': grid_path, 'loss_history_csv': hist_path, 'ckpt_path': ckpt_path}

def train_pinn_deterministic(cfg: PINNDetConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    else:
        patient_ids = [str(cfg.patient_id)] if cfg.patient_id is not None else [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    rows = []
    failed = []
    for pid in patient_ids:
        print('\n' + '=' * 70)
        print(f'Training deterministic PINN ablation for patient {pid}')
        try:
            rows.append(train_single_patient(cfg, str(pid)))
        except Exception as e:
            print(f'ERROR while training patient {pid}: {e}')
            failed.append({'patient_id': str(pid), 'error': str(e)})
    if rows:
        summary_path = os.path.join(cfg.out_dir, 'pinn_det_summary_all_patients.csv')
        pd.DataFrame(rows).sort_values('patient_id').reset_index(drop=True).to_csv(summary_path, index=False)
        print(f'Saved summary to: {summary_path}')
    if failed:
        failed_path = os.path.join(cfg.out_dir, 'pinn_det_failed_patients.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved failures to: {failed_path}')
if __name__ == '__main__':
    train_pinn_deterministic(PINNDetConfig())
