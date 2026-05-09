import torch

def ensure_2d_column(x: torch.Tensor) -> torch.Tensor:
    """
    Make sure tensor has shape (N, 1).
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return x

def prepare_time_for_grad(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure t is float32, shape (N,1), and requires grad.
    """
    t = ensure_2d_column(t)
    t = t.to(dtype=torch.float32)
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)
    return t

def compute_dy_dt(y_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute dy_hat/dt using autograd.

    Inputs:
        y_hat: shape (N,1)
        t:     shape (N,1), must require grad

    Output:
        dy_dt: shape (N,1)
    """
    y_hat = ensure_2d_column(y_hat)
    t = ensure_2d_column(t)
    if not t.requires_grad:
        raise ValueError('Input t must have requires_grad=True before calling compute_dy_dt().')
    dy_dt = torch.autograd.grad(outputs=y_hat, inputs=t, grad_outputs=torch.ones_like(y_hat), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return dy_dt

def gompertz_log_rhs(model, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Gompertz RHS in log-volume space:
        dy/dt = alpha - beta * y
    """
    y_hat = ensure_2d_column(y_hat)
    return model.alpha - model.beta * y_hat

def gompertz_log_residual(model, t: torch.Tensor) -> torch.Tensor:
    """
    Physics residual in log-volume space:
        r(t) = dy_hat/dt - (alpha - beta * y_hat)
    """
    t = prepare_time_for_grad(t)
    y_hat = model(t)
    dy_dt = compute_dy_dt(y_hat, t)
    rhs = gompertz_log_rhs(model, y_hat)
    residual = dy_dt - rhs
    return residual

def physics_loss_mse(model, t_collocation: torch.Tensor) -> torch.Tensor:
    """
    Mean squared physics residual on collocation points.
    """
    r = gompertz_log_residual(model, t_collocation)
    return torch.mean(r ** 2)

def build_collocation_points(t_obs: torch.Tensor, num_points: int=50, extend_ratio: float=0.0) -> torch.Tensor:
    """
    Build collocation points spanning the observed time range.

    Args:
        t_obs: observed time points, shape (N,) or (N,1)
        num_points: number of collocation points
        extend_ratio: optionally extend range beyond observation window

    Returns:
        t_collocation: shape (num_points, 1)
    """
    t_obs = ensure_2d_column(t_obs).to(dtype=torch.float32)
    t_min = torch.min(t_obs)
    t_max = torch.max(t_obs)
    span = torch.clamp(t_max - t_min, min=1e-06)
    t_start = t_min - extend_ratio * span
    t_end = t_max + extend_ratio * span
    t_collocation = torch.linspace(float(t_start.detach().cpu()), float(t_end.detach().cpu()), num_points, dtype=torch.float32).unsqueeze(1)
    return t_collocation
if __name__ == '__main__':
    from Model import GompertzPINN
    model = GompertzPINN()
    t_phys = torch.linspace(0.0, 2.0, 50).unsqueeze(1)
    res = gompertz_log_residual(model, t_phys)
    loss = physics_loss_mse(model, t_phys)
    print('Residual shape:', tuple(res.shape))
    print('Physics loss:', float(loss.detach().cpu()))
    print('alpha:', float(model.alpha.detach().cpu()))
    print('beta:', float(model.beta.detach().cpu()))
    t_obs = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    t_colloc = build_collocation_points(t_obs, num_points=20)
    print('Collocation shape:', tuple(t_colloc.shape))
    print('Collocation first/last:', float(t_colloc[0]), float(t_colloc[-1]))
