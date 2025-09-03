from __future__ import annotations
import numpy as np
from .integrators import integrate


def run_system(system, y0, t_span=(0.0, 5.0), h=0.01, method="rk4"):
    return integrate(system.deriv, np.array(y0, dtype=float), t_span, h, method)


def landing_time_from_trajectory(ts, ys):
    """Estimate landing time where z crosses 0 using linear interpolation.

    Assumes ys contains state with z-position at column index 2.
    Returns None if never crosses.
    """
    z = ys[:, 2]
    cross_indices = np.where((z[:-1] > 0) & (z[1:] <= 0))[0]
    if cross_indices.size == 0:
        return None
    i = int(cross_indices[0]) + 1
    t0, t1 = ts[i - 1], ts[i]
    z0, z1 = z[i - 1], z[i]
    if z1 == z0:
        return t1
    alpha = -z0 / (z1 - z0)
    return t0 + alpha * (t1 - t0)