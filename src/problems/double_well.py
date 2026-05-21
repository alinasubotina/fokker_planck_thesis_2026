import torch
import numpy as np
from scipy.integrate import quad

class DoubleWellProblem:
    """
    1D double-well potential problem from Xu et al. (2020), Example 1.
    SDE:    dx = (alpha*x - beta*x^3) dt + sigma dW
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.5, sigma: float = 0.5,
                 domain: tuple = (-2.2, 2.2)):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.domain = domain

        c_inv, _ = quad(self._unnormalized_p, domain[0], domain[1])
        self.C_norm = 1.0 / c_inv

    def mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x - self.beta * x**3

    def D(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (self.sigma ** 2)

    def potential(self, x):
        return -self.alpha * x ** 2 / 2 + self.beta * x ** 4 / 4

    def _unnormalized_p(self, x: float):
        return np.exp(-self.potential(x) / (self.sigma ** 2 / 2))

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        return self.C_norm * self._unnormalized_p(x)