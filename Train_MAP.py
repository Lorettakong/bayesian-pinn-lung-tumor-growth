import os
import math
import copy
import random
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Datasets import load_patient_timeseries, get_available_patient_ids, split_patient_train_holdout
from Model import GompertzPINN
from Physics import gompertz_log_residual, build_collocation_points
from Losses import total_energy_U

@dataclass
class TrainConfig:
    csv_path: str = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_id: str | None = None
    run_all_patients: bool = True
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    init_alpha: float = 0.2
    init_beta: float = 0.05
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
    use_nn_weight_prior: bool = False
    nn_weight_prior_std: float = 1.0
    w_nn_prior: float = 0.0
    lr: float = 0.001
    epochs: int = 5000
    print_every: int = 200
    seed: int = 42
    w_data: float = 1.0
    w_phys: float = 1.0
    w_prior: float = 1.0
    w_y0: float = 1.0
    out_dir: str = 'outputs'

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

def train_map_single_patient(cfg: TrainConfig, patient_id: str) -> dict:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    split = split_patient_train_holdout(cfg.csv_path, patient_id=patient_id, keep_only_ok=True, holdout_rule='last_timepoint')
    patient_id = str(split['patient_id'])
    df = split['full_df'].copy()
    _, t_data_full, y_obs_full, _ = load_patient_timeseries(cfg.csv_path, patient_id=patient_id, keep_only_ok=True)
    t_data = split['t_train'].to(device=device, dtype=torch.float32)
    y_obs = split['y_train'].to(device=device, dtype=torch.float32)
    t_data_full = t_data_full.to(device=device, dtype=torch.float32)
    y_obs_full = y_obs_full.to(device=device, dtype=torch.float32)
    t_min = float(t_data.min().item())
    t_max = float(t_data.max().item())
    t_full_min = float(t_data_full.min().item())
    t_full_max = float(t_data_full.max().item())
    time_col = get_time_column_name(df)
    print('\n' + '=' * 70)
    print(f'Training patient {patient_id} | n_train = {len(t_data)} | n_full = {len(t_data_full)}')
    show_cols = [c for c in ['patient_id', 'scan', time_col, 'V_obs'] if c in df.columns]
    print(df[show_cols])
    model = GompertzPINN(hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers, init_alpha=cfg.init_alpha, init_beta=cfg.init_beta, use_time_normalization=cfg.use_time_normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history_total = []
    history_data = []
    history_phys = []
    history_prior = []
    history_nn_prior = []
    history_y0 = []
    first_idx = torch.argmin(t_data.view(-1))
    t0_obs = t_data[first_idx].view(1, 1)
    y0_obs = y_obs[first_idx].view(1, 1)
    best_total = float('inf')
    best_epoch = -1
    best_state = None
    best_snapshot = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        t_phys = build_collocation_points(t_obs=t_data.detach().cpu(), num_points=cfg.n_phys, extend_ratio=cfg.phys_extend_ratio).to(device)
        out = total_energy_U(model=model, t_data=t_data, y_obs=y_obs, t_phys=t_phys, compute_residual_fn=gompertz_log_residual, sigma_d=cfg.sigma_d, sigma_p=cfg.sigma_p, prior_cfg={'mu_log_alpha': cfg.mu_log_alpha, 'sd_log_alpha': cfg.sd_log_alpha, 'mu_log_beta': cfg.mu_log_beta, 'sd_log_beta': cfg.sd_log_beta, 'use_nn_weight_prior': cfg.use_nn_weight_prior, 'nn_weight_prior_std': cfg.nn_weight_prior_std}, w_data=cfg.w_data, w_phys=cfg.w_phys, w_prior=cfg.w_prior, w_nn_prior=cfg.w_nn_prior)
        U_data = out['U_data']
        U_phys = out['U_phys']
        U_prior = out['U_prior']
        U_nn_prior = out['U_nn_prior']
        y0_model = model(t0_obs)
        U_y0 = gaussian_penalty(y0_model, y0_obs, cfg.y0_prior_sd).sum()
        U_total = out['U_total'] + cfg.w_y0 * U_y0
        U_total.backward()
        optimizer.step()
        total_val = float(U_total.item())
        data_val = float(U_data.item())
        phys_val = float(U_phys.item())
        prior_val = float(U_prior.item())
        nn_prior_val = float(U_nn_prior.item())
        y0_val = float(U_y0.item())
        history_total.append(total_val)
        history_data.append(data_val)
        history_phys.append(phys_val)
        history_prior.append(prior_val)
        history_nn_prior.append(nn_prior_val)
        history_y0.append(y0_val)
        if total_val < best_total:
            best_total = total_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_snapshot = {'alpha': float(model.alpha.item()), 'beta': float(model.beta.item()), 'U_total': total_val, 'U_data': data_val, 'U_phys': phys_val, 'U_prior': prior_val, 'U_nn_prior': nn_prior_val, 'U_y0': y0_val}
        if epoch % cfg.print_every == 0 or epoch == 1:
            print(f'Epoch {epoch:5d} | U_total={total_val:.4f} | U_data={data_val:.4f} | U_phys={phys_val:.4f} | U_prior={prior_val:.4f} | U_nn_prior={nn_prior_val:.4f} | U_y0={y0_val:.4f} | alpha={model.alpha.item():.6f} | beta={model.beta.item():.6f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_obs = model(t_data_full)
        v_pred_obs = torch.exp(y_pred_obs)
        t_grid = torch.linspace(t_full_min, t_full_max, 200, device=device, dtype=torch.float32).view(-1, 1)
        y_grid = model(t_grid)
        v_grid = torch.exp(y_grid)
    t_obs_np = t_data_full.detach().cpu().numpy().flatten()
    y_obs_np = y_obs_full.detach().cpu().numpy().flatten()
    y_pred_obs_np = y_pred_obs.detach().cpu().numpy().flatten()
    v_obs_np = np.exp(y_obs_np)
    v_pred_obs_np = v_pred_obs.detach().cpu().numpy().flatten()
    t_grid_np = t_grid.detach().cpu().numpy().flatten()
    y_grid_np = y_grid.detach().cpu().numpy().flatten()
    v_grid_np = v_grid.detach().cpu().numpy().flatten()
    rmse_log = rmse(y_obs_np, y_pred_obs_np)
    rmse_volume = rmse(v_obs_np, v_pred_obs_np)
    ckpt_path = os.path.join(cfg.out_dir, f'map_model_{patient_id}.pt')
    torch.save({'model_state': model.state_dict(), 'patient_id': patient_id, 'alpha': float(model.alpha.item()), 'beta': float(model.beta.item()), 't_min': t_min, 't_max': t_max, 't0_obs': float(t0_obs.item()), 'y0_obs': float(y0_obs.item()), 'best_epoch': best_epoch, 'best_total': best_total, 'best_snapshot': best_snapshot, 'config': asdict(cfg), 'history_total': history_total, 'history_data': history_data, 'history_phys': history_phys, 'history_prior': history_prior, 'history_nn_prior': history_nn_prior, 'history_y0': history_y0, 'rmse_log': rmse_log, 'rmse_volume': rmse_volume}, ckpt_path)
    print(f'Saved MAP checkpoint to: {ckpt_path}')
    pred_df = df.copy()
    pred_df['y_obs'] = y_obs_np
    pred_df['y_pred'] = y_pred_obs_np
    pred_df['V_obs'] = v_obs_np
    pred_df['V_pred'] = v_pred_obs_np
    pred_path = os.path.join(cfg.out_dir, f'predictions_{patient_id}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f'Saved predictions to: {pred_path}')
    hist_df = pd.DataFrame({'epoch': np.arange(1, cfg.epochs + 1), 'U_total': history_total, 'U_data': history_data, 'U_phys': history_phys, 'U_prior': history_prior, 'U_nn_prior': history_nn_prior, 'U_y0': history_y0})
    hist_path = os.path.join(cfg.out_dir, f'loss_history_{patient_id}.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f'Saved loss history to: {hist_path}')
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['U_total'], label='U_total')
    plt.plot(hist_df['epoch'], hist_df['U_data'], label='U_data')
    plt.plot(hist_df['epoch'], hist_df['U_phys'], label='U_phys')
    plt.plot(hist_df['epoch'], hist_df['U_prior'], label='U_prior')
    plt.plot(hist_df['epoch'], hist_df['U_nn_prior'], label='U_nn_prior')
    plt.plot(hist_df['epoch'], hist_df['U_y0'], label='U_y0')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss curves | patient {patient_id}')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(cfg.out_dir, f'loss_curve_{patient_id}.png')
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    plt.figure()
    plt.scatter(t_obs_np, y_obs_np, label='Observed')
    plt.plot(t_grid_np, y_grid_np, label='MAP fit')
    plt.xlabel(time_col)
    plt.ylabel('y = log(V)')
    plt.title(f'MAP fit (log-space) | patient {patient_id}')
    plt.legend()
    plt.tight_layout()
    log_plot_path = os.path.join(cfg.out_dir, f'map_log_fit_{patient_id}.png')
    plt.savefig(log_plot_path, dpi=200)
    plt.close()
    plt.figure()
    plt.scatter(t_obs_np, v_obs_np, label='Observed')
    plt.plot(t_grid_np, v_grid_np, label='MAP fit')
    plt.xlabel(time_col)
    plt.ylabel('V (cm^3)')
    plt.title(f'MAP fit (volume-space) | patient {patient_id}')
    plt.legend()
    plt.tight_layout()
    volume_plot_path = os.path.join(cfg.out_dir, f'map_volume_fit_{patient_id}.png')
    plt.savefig(volume_plot_path, dpi=200)
    plt.close()
    print(f'Training complete for patient {patient_id}.')
    print(f'Best epoch : {best_epoch}')
    print(f'Final alpha: {model.alpha.item():.6f}')
    print(f'Final beta : {model.beta.item():.6f}')
    print(f'RMSE log   : {rmse_log:.6f}')
    print(f'RMSE volume: {rmse_volume:.6f}')
    return {'patient_id': patient_id, 'alpha_map': float(model.alpha.item()), 'beta_map': float(model.beta.item()), 't_min': t_min, 't_max': t_max, 't0_obs': float(t0_obs.item()), 'y0_obs': float(y0_obs.item()), 'n_data': int(len(t_data)), 'best_epoch': int(best_epoch), 'best_U_total': float(best_total), 'best_U_data': float(best_snapshot.get('U_data', np.nan)), 'best_U_phys': float(best_snapshot.get('U_phys', np.nan)), 'best_U_prior': float(best_snapshot.get('U_prior', np.nan)), 'best_U_nn_prior': float(best_snapshot.get('U_nn_prior', np.nan)), 'best_U_y0': float(best_snapshot.get('U_y0', np.nan)), 'final_U_total': float(history_total[-1]), 'final_U_data': float(history_data[-1]), 'final_U_phys': float(history_phys[-1]), 'final_U_prior': float(history_prior[-1]), 'final_U_nn_prior': float(history_nn_prior[-1]), 'final_U_y0': float(history_y0[-1]), 'rmse_log': rmse_log, 'rmse_volume': rmse_volume, 'ckpt_path': ckpt_path, 'predictions_csv': pred_path, 'loss_history_csv': hist_path, 'loss_curve_png': loss_plot_path, 'log_plot_png': log_plot_path, 'volume_plot_png': volume_plot_path}

def train_map(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.run_all_patients:
        patient_ids = get_available_patient_ids(cfg.csv_path, keep_only_ok=True)
    elif cfg.patient_id is None:
        patient_ids = [get_available_patient_ids(cfg.csv_path, keep_only_ok=True)[0]]
    else:
        patient_ids = [str(cfg.patient_id)]
    print('\nPatients to train:')
    for pid in patient_ids:
        print(' -', pid)
    results = []
    failed = []
    for pid in patient_ids:
        try:
            result = train_map_single_patient(cfg, pid)
            results.append(result)
        except Exception as e:
            print(f'\nERROR while training patient {pid}: {e}')
            failed.append({'patient_id': pid, 'error': str(e)})
    if len(results) > 0:
        summary_path = os.path.join(cfg.out_dir, 'map_summary.csv')
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f'\nSaved MAP summary to: {summary_path}')
    if len(failed) > 0:
        failed_path = os.path.join(cfg.out_dir, 'map_failed_cases.csv')
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f'Saved failed cases to: {failed_path}')
    print('\nAll done.')
    print(f'Successful patients: {len(results)}')
    print(f'Failed patients    : {len(failed)}')
if __name__ == '__main__':
    config = TrainConfig(csv_path='/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv', patient_id=None, run_all_patients=True, out_dir='outputs')
    train_map(config)
