import math
import torch
import torch.nn as nn

class GompertzPINN(nn.Module):
    """
    PINN for Gompertz tumor growth in log-volume space.

    Network output:
        y_hat(t) = log(V(t))

    Gompertz ODE in volume space:
        dV/dt = alpha * V - beta * V * log(V)

    Equivalent ODE in log-volume space:
        y(t) = log(V(t))
        dy/dt = alpha - beta * y
    """

    def __init__(self, hidden_dim: int=64, num_hidden_layers: int=3, init_alpha: float=0.2, init_beta: float=0.05, eps: float=1e-08, use_time_normalization: bool=True):
        super().__init__()
        self.eps = eps
        self.softplus = nn.Softplus()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.use_time_normalization = use_time_normalization
        self.register_buffer('time_norm_min', torch.tensor(float('nan'), dtype=torch.float32))
        self.register_buffer('time_norm_max', torch.tensor(float('nan'), dtype=torch.float32))
        layers = []
        in_dim = 1
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.raw_alpha = nn.Parameter(torch.tensor(self._inv_softplus(init_alpha), dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.tensor(self._inv_softplus(init_beta), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for stable PINN training.
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """
        Approximate inverse of softplus for positive initialization.
        """
        y = max(float(y), 1e-06)
        if y > 20:
            return y
        return math.log(math.expm1(y))

    @property
    def alpha(self):
        return self.softplus(self.raw_alpha) + self.eps

    @property
    def beta(self):
        return self.softplus(self.raw_beta) + self.eps

    def set_time_normalization_stats(self, t_ref: torch.Tensor):
        """
        Persist normalization statistics from the training-time reference points.
        All later calls should reuse the same stats so data, physics, and grids
        are represented in one consistent time coordinate system.
        """
        if t_ref.dim() == 1:
            t_ref = t_ref.unsqueeze(1)
        t_ref = t_ref.to(dtype=torch.float32)
        self.time_norm_min.copy_(torch.min(t_ref).detach())
        self.time_norm_max.copy_(torch.max(t_ref).detach())

    def has_time_normalization_stats(self) -> bool:
        return bool(torch.isfinite(self.time_norm_min).item() and torch.isfinite(self.time_norm_max).item())

    def get_time_normalization_stats(self):
        return {'time_norm_min': float(self.time_norm_min.detach().cpu().item()), 'time_norm_max': float(self.time_norm_max.detach().cpu().item())}

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalize time to roughly [-1, 1] or centered small scale for better stability.
        For very short trajectories like [0,1,2], this changes little.
        For real day-scale time, this helps a lot.
        """
        if not self.use_time_normalization:
            return t
        if self.has_time_normalization_stats():
            t_min = self.time_norm_min.to(device=t.device, dtype=t.dtype)
            t_max = self.time_norm_max.to(device=t.device, dtype=t.dtype)
        else:
            t_min = torch.min(t)
            t_max = torch.max(t)
        denom = torch.clamp(t_max - t_min, min=1e-06)
        t_norm = 2.0 * (t - t_min) / denom - 1.0
        return t_norm

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Predict y_hat(t) = log(V(t)).

        Input:
            t: shape (N,) or (N,1)

        Output:
            y_hat: shape (N,1)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(dtype=torch.float32)
        t_in = self._normalize_time(t)
        y_hat = self.net(t_in)
        return y_hat

    def predict_log_volume(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward(t)

    def predict_volume(self, t: torch.Tensor) -> torch.Tensor:
        y_hat = self.forward(t)
        v_hat = torch.exp(y_hat)
        return torch.clamp(v_hat, min=self.eps)

    def gompertz_rhs_log(self, y: torch.Tensor) -> torch.Tensor:
        """
        dy/dt = alpha - beta * y
        """
        return self.alpha - self.beta * y

    def gompertz_rhs_volume(self, v: torch.Tensor) -> torch.Tensor:
        """
        dV/dt = alpha * V - beta * V * log(V)
        """
        v = torch.clamp(v, min=self.eps)
        return self.alpha * v - self.beta * v * torch.log(v)

    def closed_form_log_solution(self, t: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """
        Closed-form solution of:
            dy/dt = alpha - beta y
        with initial condition y(0)=y0

        y(t) = alpha/beta + (y0 - alpha/beta) * exp(-beta t)

        Notes:
        - This assumes t is relative time from baseline.
        - Useful later for HMC / posterior predictive checks.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(dtype=torch.float32)
        y0 = y0.to(dtype=torch.float32)
        steady = self.alpha / self.beta
        return steady + (y0 - steady) * torch.exp(-self.beta * t)

    def get_param_dict(self):
        """
        Convenient for logging / saving MAP summaries.
        """
        return {'alpha': float(self.alpha.detach().cpu().item()), 'beta': float(self.beta.detach().cpu().item()), 'hidden_dim': int(self.hidden_dim), 'num_hidden_layers': int(self.num_hidden_layers), 'use_time_normalization': bool(self.use_time_normalization), **self.get_time_normalization_stats()}
if __name__ == '__main__':
    model = GompertzPINN()
    t = torch.linspace(0, 2, 10).unsqueeze(1)
    y_hat = model(t)
    v_hat = model.predict_volume(t)
    print('y_hat shape:', y_hat.shape)
    print('v_hat shape:', v_hat.shape)
    print('alpha:', model.alpha.item())
    print('beta:', model.beta.item())
    print('param_dict:', model.get_param_dict())
