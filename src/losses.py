import torch
import torch.nn as nn
from typing import Callable, Dict
from src.utils import compute_gradients

def fp_loss(
        model: nn.Module,
        x: torch.Tensor,
        x_boundary: torch.Tensor,
        mu_fn: Callable[[torch.Tensor], torch.Tensor],
        D_fn: Callable[[torch.Tensor], torch.Tensor],
        dx: float,
        use_normalization: bool = True,
        x_norm: torch.Tensor = None,
        a1: float = 1.0,
        a2: float = 1.0,
        a3: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Computes the total loss for the DL-FP method.
    Loss = a1 * E_pde + a2 * E_norm + a3 * E_bound
    """
    assert x.requires_grad, (
        "Collocation tensor `x` must have requires_grad=True. "
        "Call x.requires_grad_(True) before passing to fp_loss."
    )

    # ----- E1: PDE Residual -----
    p = model(x)
    mu_p = mu_fn(x) * p
    D_p = D_fn(x) * p

    d_mu_p_dx = compute_gradients(mu_p, x)
    d_D_p_dx = compute_gradients(D_p, x)
    d2_D_p_dx2 = compute_gradients(d_D_p_dx, x)

    fpe_residual = -d_mu_p_dx + 0.5 * d2_D_p_dx2
    loss_pde = torch.mean(fpe_residual ** 2)

    # ----- E2: Normalization -----
    if use_normalization:
        integral_p = torch.sum(p) * dx
        loss_norm = (integral_p - 1.0) ** 2
    else:
        loss_norm = torch.tensor(0.0, device=x.device)


    # ----- E3: Boundary -----
    p_boundary = model(x_boundary)
    loss_bound = torch.mean(p_boundary ** 2)

    # ----- Total -----
    total = a1 * loss_pde + a2 * loss_norm + a3 * loss_bound

    return {
        'pde': loss_pde,
        'norm': loss_norm,
        'bound': loss_bound,
        'total': total,
    }