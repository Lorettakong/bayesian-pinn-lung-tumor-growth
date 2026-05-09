import math
import torch

def ensure_2d_column(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return x

def get_default_prior_cfg():
    """
    Default prior configuration for alpha and beta in log-space.
    You can tighten sd later (e.g. 0.5) if posterior intervals are too wide.
    """
    return {'mu_log_alpha': 0.0, 'sd_log_alpha': 1.0, 'mu_log_beta': 0.0, 'sd_log_beta': 1.0, 'use_nn_weight_prior': False, 'nn_weight_prior_std': 1.0}

def nll_data_gaussian(y_hat: torch.Tensor, y_obs: torch.Tensor, sigma_d: torch.Tensor | float) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for observed log-volume data.
    """
    y_hat = ensure_2d_column(y_hat)
    y_obs = ensure_2d_column(y_obs)
    if y_hat.shape != y_obs.shape:
        raise ValueError(f'Shape mismatch: y_hat {y_hat.shape} vs y_obs {y_obs.shape}')
    if not torch.is_tensor(sigma_d):
        sigma_d = torch.tensor(float(sigma_d), device=y_hat.device, dtype=y_hat.dtype)
    sigma_d = torch.clamp(sigma_d, min=1e-08)
    resid = y_obs - y_hat
    n = y_hat.shape[0]
    term_quad = 0.5 * torch.sum((resid / sigma_d) ** 2)
    term_log = 0.5 * n * torch.log(2.0 * math.pi * sigma_d ** 2)
    return term_quad + term_log

def nll_phys_gaussian(residual_f: torch.Tensor, sigma_p: torch.Tensor | float) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for physics residuals.
    """
    residual_f = ensure_2d_column(residual_f)
    if not torch.is_tensor(sigma_p):
        sigma_p = torch.tensor(float(sigma_p), device=residual_f.device, dtype=residual_f.dtype)
    sigma_p = torch.clamp(sigma_p, min=1e-08)
    m = residual_f.shape[0]
    term_quad = 0.5 * torch.sum((residual_f / sigma_p) ** 2)
    term_log = 0.5 * m * torch.log(2.0 * math.pi * sigma_p ** 2)
    return term_quad + term_log

def nll_log_normal_prior_alpha_beta(alpha: torch.Tensor, beta: torch.Tensor, mu_log_alpha: float=0.0, sd_log_alpha: float=1.0, mu_log_beta: float=0.0, sd_log_beta: float=1.0) -> torch.Tensor:
    """
    Log-normal prior NLL for positive alpha and beta.
    """
    eps = 1e-12
    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)
    log_alpha = torch.log(alpha)
    log_beta = torch.log(beta)
    sd_log_alpha = max(float(sd_log_alpha), 1e-08)
    sd_log_beta = max(float(sd_log_beta), 1e-08)
    U_alpha = 0.5 * ((log_alpha - mu_log_alpha) / sd_log_alpha) ** 2 + 0.5 * math.log(2.0 * math.pi * sd_log_alpha ** 2) + log_alpha
    U_beta = 0.5 * ((log_beta - mu_log_beta) / sd_log_beta) ** 2 + 0.5 * math.log(2.0 * math.pi * sd_log_beta ** 2) + log_beta
    return U_alpha + U_beta

def nll_gaussian_prior_nn_weights(model, std: float=1.0) -> torch.Tensor:
    """
    Optional Gaussian prior / L2-style regularization for neural network weights.
    Only applies to parameters that are not raw_alpha / raw_beta.
    """
    std = max(float(std), 1e-08)
    total = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in ['raw_alpha', 'raw_beta']:
            continue
        term = 0.5 * torch.sum((param / std) ** 2) + 0.5 * param.numel() * math.log(2.0 * math.pi * std ** 2)
        if total is None:
            total = term
        else:
            total = total + term
    if total is None:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)
    return total

def total_energy_U(model, t_data: torch.Tensor, y_obs: torch.Tensor, t_phys: torch.Tensor, compute_residual_fn, sigma_d: float=0.2, sigma_p: float=0.5, prior_cfg: dict | None=None, w_data: float=1.0, w_phys: float=1.0, w_prior: float=1.0, w_nn_prior: float=0.0) -> dict:
    """
    Total MAP objective:

        U_total = w_data * U_data
                + w_phys * U_phys
                + w_prior * U_prior
                + w_nn_prior * U_nn_prior

    Returns a dictionary of all components for logging and debugging.
    """
    if prior_cfg is None:
        prior_cfg = get_default_prior_cfg()
    else:
        tmp = get_default_prior_cfg()
        tmp.update(prior_cfg)
        prior_cfg = tmp
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    t_data = ensure_2d_column(t_data).to(device=device, dtype=torch.float32)
    y_obs = ensure_2d_column(y_obs).to(device=device, dtype=torch.float32)
    t_phys = ensure_2d_column(t_phys).to(device=device, dtype=torch.float32)
    if not hasattr(model, 'alpha') or not hasattr(model, 'beta'):
        raise AttributeError('Model must expose positive parameters: model.alpha and model.beta')
    y_hat = model(t_data)
    U_data = nll_data_gaussian(y_hat=y_hat, y_obs=y_obs, sigma_d=sigma_d)
    residual_f = compute_residual_fn(model, t_phys)
    U_phys = nll_phys_gaussian(residual_f=residual_f, sigma_p=sigma_p)
    U_prior = nll_log_normal_prior_alpha_beta(alpha=model.alpha, beta=model.beta, mu_log_alpha=prior_cfg['mu_log_alpha'], sd_log_alpha=prior_cfg['sd_log_alpha'], mu_log_beta=prior_cfg['mu_log_beta'], sd_log_beta=prior_cfg['sd_log_beta'])
    if prior_cfg.get('use_nn_weight_prior', False):
        U_nn_prior = nll_gaussian_prior_nn_weights(model=model, std=prior_cfg['nn_weight_prior_std'])
    else:
        U_nn_prior = torch.tensor(0.0, device=device, dtype=dtype)
    U_total = w_data * U_data + w_phys * U_phys + w_prior * U_prior + w_nn_prior * U_nn_prior
    return {'U_total': U_total, 'U_data': U_data, 'U_phys': U_phys, 'U_prior': U_prior, 'U_nn_prior': U_nn_prior, 'y_hat': y_hat, 'residual_f': residual_f, 'alpha': model.alpha, 'beta': model.beta}
if __name__ == '__main__':
    import torch.nn as nn

    class TinyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
            self.softplus = nn.Softplus()
            self.raw_alpha = nn.Parameter(torch.tensor(0.1))
            self.raw_beta = nn.Parameter(torch.tensor(-2.0))

        @property
        def alpha(self):
            return self.softplus(self.raw_alpha) + 1e-08

        @property
        def beta(self):
            return self.softplus(self.raw_beta) + 1e-08

        def forward(self, t):
            if t.dim() == 1:
                t = t.unsqueeze(1)
            return self.net(t)

    def gompertz_log_residual(model, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.clone().to(dtype=torch.float32).detach().requires_grad_(True)
        y_hat = model(t)
        dy_dt = torch.autograd.grad(outputs=y_hat, inputs=t, grad_outputs=torch.ones_like(y_hat), create_graph=True, retain_graph=True)[0]
        return dy_dt - (model.alpha - model.beta * y_hat)
    torch.manual_seed(0)
    model = TinyModel()
    t_data = torch.linspace(0, 2, 8).unsqueeze(1)
    y_obs = torch.sin(t_data / 2.0)
    t_phys = torch.linspace(0, 2, 50).unsqueeze(1)
    out = total_energy_U(model=model, t_data=t_data, y_obs=y_obs, t_phys=t_phys, compute_residual_fn=gompertz_log_residual, sigma_d=0.2, sigma_p=0.2, prior_cfg={'mu_log_alpha': 0.0, 'sd_log_alpha': 1.0, 'mu_log_beta': 0.0, 'sd_log_beta': 1.0, 'use_nn_weight_prior': False, 'nn_weight_prior_std': 1.0}, w_data=1.0, w_phys=1.0, w_prior=1.0, w_nn_prior=0.0)
    print('U_total   :', float(out['U_total'].detach()))
    print('U_data    :', float(out['U_data'].detach()))
    print('U_phys    :', float(out['U_phys'].detach()))
    print('U_prior   :', float(out['U_prior'].detach()))
    print('U_nn_prior:', float(out['U_nn_prior'].detach()))
    print('alpha     :', float(model.alpha.detach()))
    print('beta      :', float(model.beta.detach()))
    out['U_total'].backward()
    print('Backward OK. Example grad(raw_alpha):', float(model.raw_alpha.grad.detach()))
