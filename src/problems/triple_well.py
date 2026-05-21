import torch
import numpy as np
from scipy.integrate import quad


class TripleWellProblem:
    """
    1D triple-well potential problem.
    """

    def __init__(
            self,
            a: float = 0.2,
            b: float = 1.5,
            c: float = 3.0,
            d: float = 0.0,
            sigma: float = 2.0,
            domain: tuple = (-4.0, 4.0),
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sigma = sigma
        self.domain = domain

        c_inv, _ = quad(self._unnormalized_p, domain[0], domain[1])
        self.C_norm = 1.0 / c_inv

    def mu(self, x: torch.Tensor) -> torch.Tensor:
        return -6 * self.a * x**5 + 4 * self.b * x**3 - 2 * self.c * x - self.d

    def D(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * (self.sigma ** 2)

    def potential(self, x):
        return (
            self.a * x**6
            - self.b * x**4
            + self.c * x**2
            + self.d * x
        )

    def potential_prime(self, x):
        return 6 * self.a * x**5 - 4 * self.b * x**3 + 2 * self.c * x + self.d

    def potential_double_prime(self, x):
        return 30 * self.a * x**4 - 12 * self.b * x**2 + 2 * self.c

    def _unnormalized_p(self, x):
        return np.exp(-self.potential(x) / (self.sigma ** 2 / 2))

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        return self.C_norm * self._unnormalized_p(x)

