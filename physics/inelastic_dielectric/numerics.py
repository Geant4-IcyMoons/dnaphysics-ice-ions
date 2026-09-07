"""Shared quadrature; retained without numerical changes."""
import numpy as np

def _simpson_integrate(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    if n != x.size:
        raise ValueError("x and y must have the same length")
    if n == 2:
        return 0.5 * (y[0] + y[1]) * (x[1] - x[0])

    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        return float(np.sum(0.5 * dx * (y[:-1] + y[1:])))

    h = dx[0]
    if n % 2 == 1:
        return float(
            (h / 3.0)
            * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
        )

    n1 = n - 1
    simpson_part = (h / 3.0) * (
        y[0]
        + y[n1 - 1]
        + 4.0 * np.sum(y[1 : n1 - 1 : 2])
        + 2.0 * np.sum(y[2 : n1 - 2 : 2])
    )
    trap_part = 0.5 * (y[-2] + y[-1]) * (x[-1] - x[-2])
    return float(simpson_part + trap_part)
