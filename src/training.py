import torch
import numpy as np
import warnings
from typing import Dict, Any, Optional
from src.losses import fp_loss
from src.models import FPNet


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_grid(domain: tuple, target_dx: float, device: torch.device):
    """Uniform grid of collocation points with exact step size."""
    num_coll = int(round((domain[1] - domain[0]) / target_dx)) + 1
    x_grid = torch.linspace(
        domain[0], domain[1], num_coll,
        dtype=torch.float32, device=device,
    ).view(-1, 1)
    x_grid.requires_grad_(True)
    actual_dx = (domain[1] - domain[0]) / (num_coll - 1)
    return x_grid, actual_dx, num_coll


def _make_boundary(domain: tuple, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[domain[0]], [domain[1]]],
        dtype=torch.float32, device=device,
    )


def _compute_losses(model, x_grid, x_boundary, problem,
                    actual_dx, use_normalization, penalty_factors):
    return fp_loss(
        model=model,
        x=x_grid,
        x_boundary=x_boundary,
        mu_fn=problem.mu,
        D_fn=problem.D,
        dx=actual_dx,
        use_normalization=use_normalization,
        a1=penalty_factors['a1'],
        a2=penalty_factors['a2'],
        a3=penalty_factors['a3'],
    )


def _build_optimizer_config(optimizer_config: dict) -> dict:
    opt_type = optimizer_config.get('type', 'adam')
    if opt_type == 'adam':
        return {
            'type': 'adam',
            'epochs': 30000,
            'lr': 1e-3,
            **optimizer_config,
        }
    elif opt_type == 'lbfgs':
        return {
            'type': 'lbfgs',
            'epochs': 5000,
            'lr': 1.0,
            'max_iter': 20,
            'history_size': 50,
            **optimizer_config,
        }
    elif opt_type == 'hybrid':
        return {
            'type': 'hybrid',
            'adam_epochs': 5000,
            'adam_lr': 1e-3,
            'lbfgs_epochs': 25000,
            'lbfgs_lr': 1.0,
            'lbfgs_max_iter': 20,
            'lbfgs_history_size': 50,
            **optimizer_config,
        }
    else:
        raise ValueError(
            f"optimizer_config['type'] має бути 'adam', 'lbfgs' або 'hybrid'."
            f" Отримано: '{opt_type}'"
        )


def _print_header(optimizer_config, model_config, penalty_factors,
                  use_normalization, seed, device, domain,
                  actual_dx, num_coll):
    opt_type = optimizer_config['type']
    if opt_type == 'adam':
        opt_str = (f"Adam | epochs={optimizer_config['epochs']}, "
                   f"lr={optimizer_config['lr']}")
    elif opt_type == 'lbfgs':
        opt_str = (f"L-BFGS | epochs={optimizer_config['epochs']}, "
                   f"lr={optimizer_config['lr']}, "
                   f"max_iter={optimizer_config['max_iter']}")
    else:
        opt_str = (f"Hybrid | Adam {optimizer_config['adam_epochs']} ep"
                   f" → L-BFGS {optimizer_config['lbfgs_epochs']} ep")
    print(
        f"Optimizer:  {opt_str}\n"
        f"Model:      layers={model_config['hidden_layers']}, "
        f"neurons={model_config['neurons']}, "
        f"transform={model_config['output_transform']}\n"
        f"seed={seed} | device={device} | normalization={use_normalization}\n"
        f"Penalties:  a1={penalty_factors['a1']}, "
        f"a2={penalty_factors['a2']}, a3={penalty_factors['a3']}\n"
        f"Grid:       domain={domain}, dx={actual_dx:.6f}, N={num_coll}\n"
        + "-" * 60
    )


def _log_epoch(epoch, total_epochs, losses_dict, phase):
    print(
        f"[{phase}] Epoch {epoch:05d}/{total_epochs} | "
        f"Total: {losses_dict['total']:.4e} | "
        f"PDE: {losses_dict['pde']:.4e} | "
        f"Norm: {losses_dict['norm']:.4e} | "
        f"Bound: {losses_dict['bound']:.4e}"
    )


# training loops --------------------------------------------------------------------

def _train_adam(model, x_grid, x_boundary, problem, actual_dx,
                use_normalization, penalty_factors,
                epochs, lr, history, print_every,
                epoch_offset=0, total_epochs=None, phase="Adam"):
    """
    Training with Adam optimizer
    """
    if total_epochs is None:
        total_epochs = epoch_offset + epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        if x_grid.grad is not None:
            x_grid.grad.zero_()
        losses = _compute_losses(
            model, x_grid, x_boundary, problem,
            actual_dx, use_normalization, penalty_factors,
        )
        losses['total'].backward()
        optimizer.step()

        s = {k: v.item() for k, v in losses.items()}
        history['total'].append(s['total'])
        history['pde'].append(s['pde'])
        history['norm'].append(s['norm'])
        history['bound'].append(s['bound'])
        history['phase'].append(phase)

        if epoch % print_every == 0 or epoch == epochs - 1:
            _log_epoch(epoch_offset + epoch, total_epochs, s, phase)


def _train_lbfgs(model, x_grid, x_boundary, problem, actual_dx,
                 use_normalization, penalty_factors,
                 epochs, lr, max_iter, history_size, history, print_every,
                 epoch_offset=0, total_epochs=None, phase="L-BFGS"):
    """
    Training with L-BFGS optimizer
    """
    if total_epochs is None:
        total_epochs = epoch_offset + epochs

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn='strong_wolfe',
    )
    model.train()

    _last: Dict[str, float] = {}

    def closure():
        optimizer.zero_grad()

        if x_grid.grad is not None:
            x_grid.grad.zero_()
        losses = _compute_losses(
            model, x_grid, x_boundary, problem,
            actual_dx, use_normalization, penalty_factors,
        )
        losses['total'].backward()
        _last['total'] = losses['total'].item()
        _last['pde'] = losses['pde'].item()
        _last['norm'] = losses['norm'].item()
        _last['bound'] = losses['bound'].item()
        return losses['total']

    for epoch in range(epochs):
        optimizer.step(closure)

        history['total'].append(_last.get('total', float('nan')))
        history['pde'].append(_last.get('pde', float('nan')))
        history['norm'].append(_last.get('norm', float('nan')))
        history['bound'].append(_last.get('bound', float('nan')))
        history['phase'].append(phase)

        if epoch % print_every == 0 or epoch == epochs - 1:
            _log_epoch(epoch_offset + epoch, total_epochs, _last, phase)



def train_model(
        problem,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        dx: float = 0.01,
        use_normalization: bool = True,
        penalty_factors: Optional[Dict[str, float]] = None,
        seed: int = 42,
        device: Optional[torch.device] = None,
        print_every: int = 5000,
        init_model=None,
        train_config: Optional[Dict[str, Any]] = None,  # [застарів]
) -> Dict[str, Any]:
    """
    Train a neural network model to solve the given problem using the specified configurations.
    """

    if train_config is not None:
        warnings.warn(
            "\n[train_model] 'train_config' is deprecated. "
            "Use 'optimizer_config' and 'dx' instead.\n",
            DeprecationWarning,
            stacklevel=2,
        )
        if optimizer_config is None:
            optimizer_config = {
                'type': 'adam',
                'epochs': train_config.get('epochs', 30000),
                'lr': train_config.get('lr', 1e-3),
            }
        if 'dx' in train_config and dx == 0.01:
            dx = train_config['dx']

    # ----- Defaults -----
    if model_config is None:
        model_config = {}
    model_config = {
        'input_dim': 1,
        'hidden_layers': 4,
        'neurons': 20,
        'output_transform': 'abs',
        **model_config,
    }

    if optimizer_config is None:
        optimizer_config = {}
    optimizer_config = _build_optimizer_config(optimizer_config)

    if penalty_factors is None:
        penalty_factors = {}
    penalty_factors = {'a1': 1.0, 'a2': 1.0, 'a3': 1.0, **penalty_factors}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Initialization -----
    _set_seed(seed, device)
    model = FPNet(**model_config).to(device)

    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
        print("Weights initialized from pretrained model.")

    domain = problem.domain
    x_grid, actual_dx, num_coll = _make_grid(domain, dx, device)
    x_boundary = _make_boundary(domain, device)

    history: Dict[str, list] = {
        'total': [], 'pde': [], 'norm': [], 'bound': [], 'phase': []
    }

    _print_header(optimizer_config, model_config, penalty_factors,
                  use_normalization, seed, device, domain, actual_dx, num_coll)

    # ----- Training -----
    opt_type = optimizer_config['type']

    if opt_type == 'adam':
        total_epochs = optimizer_config['epochs']
        _train_adam(
            model, x_grid, x_boundary, problem, actual_dx,
            use_normalization, penalty_factors,
            epochs=total_epochs,
            lr=optimizer_config['lr'],
            history=history,
            print_every=print_every,
            epoch_offset=0,
            total_epochs=total_epochs,
            phase="Adam",
        )

    elif opt_type == 'lbfgs':
        total_epochs = optimizer_config['epochs']
        _train_lbfgs(
            model, x_grid, x_boundary, problem, actual_dx,
            use_normalization, penalty_factors,
            epochs=total_epochs,
            lr=optimizer_config['lr'],
            max_iter=optimizer_config['max_iter'],
            history_size=optimizer_config['history_size'],
            history=history,
            print_every=print_every,
            epoch_offset=0,
            total_epochs=total_epochs,
            phase="L-BFGS",
        )

    elif opt_type == 'hybrid':
        adam_ep = optimizer_config['adam_epochs']
        lbfgs_ep = optimizer_config['lbfgs_epochs']
        total_epochs = adam_ep + lbfgs_ep

        print(f"=== Phase 1/2: Adam ({adam_ep} epochs) ===")
        _train_adam(
            model, x_grid, x_boundary, problem, actual_dx,
            use_normalization, penalty_factors,
            epochs=adam_ep,
            lr=optimizer_config['adam_lr'],
            history=history,
            print_every=print_every,
            epoch_offset=0,
            total_epochs=total_epochs,
            phase="Adam",
        )

        print(f"\n=== Phase 2/2: L-BFGS ({lbfgs_ep} epochs) ===")
        _train_lbfgs(
            model, x_grid, x_boundary, problem, actual_dx,
            use_normalization, penalty_factors,
            epochs=lbfgs_ep,
            lr=optimizer_config['lbfgs_lr'],
            max_iter=optimizer_config['lbfgs_max_iter'],
            history_size=optimizer_config['lbfgs_history_size'],
            history=history,
            print_every=print_every,
            epoch_offset=adam_ep,
            total_epochs=total_epochs,
            phase="L-BFGS",
        )

    print("\nTraining completed!")

    return {
        'model': model,
        'history': history,
        'config': {
            'model_config': model_config,
            'optimizer_config': optimizer_config,
            'train_config': {
                'epochs': total_epochs,
                'lr': optimizer_config.get('lr', optimizer_config.get('adam_lr')),
                'dx': dx,
            },
            'use_normalization': use_normalization,
            'penalty_factors': penalty_factors,
            'seed': seed,
            'device': str(device),
            'actual_dx': actual_dx,
            'num_collocation': num_coll,
            'total_epochs': total_epochs,
        },
    }