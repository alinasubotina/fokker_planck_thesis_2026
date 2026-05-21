import torch
import numpy as np

def compute_gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the first derivative of y with respect to x using PyTorch autograd.
    """
    grad_y = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad_y


def acc_L2(y_pred: np.ndarray, y_exact: np.ndarray) -> float:
    """
    Computes the L2 accuracy metric as defined in Xu et al. (2020).
    Formula: acc_L2 = 1 - (||y_pred - y_exact||_2 / ||y_exact||_2)
    """
    diff_norm = np.linalg.norm(y_pred - y_exact, ord=2)
    exact_norm = np.linalg.norm(y_exact, ord=2)
    accuracy = 1.0 - (diff_norm / exact_norm)

    return float(accuracy)