from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

Vector = np.ndarray
DerivFunc = Callable[[float, Vector], Vector]


def euler_step(f: DerivFunc, t: float, y: Vector, h: float) -> Vector:
    """Forward Euler step: y_{n+1} = y_n + h * f(t_n, y_n)."""
    return y + h * f(t, y)


def rk4_step(f: DerivFunc, t: float, y: Vector, h: float) -> Vector:
    """Classic Runge–Kutta 4th order step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: DerivFunc,
    y0: Vector,
    t_span: Tuple[float, float],
    h: float,
    method: str = "rk4",
):
    """Integrate an ODE system dy/dt=f(t,y) over t_span with step h.

    Returns
    -------
    ts : (n,) array of time points
    ys : (n, d) array of states
    """
    t0, tf = t_span
    if h <= 0:
        raise ValueError("Time step h must be positive")
    if tf <= t0:
        raise ValueError("t_span must have tf > t0")

    n = int(np.ceil((tf - t0) / h)) + 1
    ys = np.zeros((n, y0.size), dtype=float)
    ts = np.zeros(n, dtype=float)
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)

    step = rk4_step if method.lower() == "rk4" else euler_step

    for i in range(n):
        ts[i] = t
        ys[i] = y
        current_h = min(h, tf - t)
        if i < n - 1:
            y = step(f, t, y, current_h)
            t += current_h
    return ts, ys