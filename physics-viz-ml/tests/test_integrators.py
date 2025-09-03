import numpy as np
from engine.integrators import integrate

# dy/dt = a, simple linear motion. y = y0 + a t

def test_linear_motion():
    a = 2.0

    def f(t, y):
        return np.array([a])

    ts, ys = integrate(f, np.array([0.0]), (0.0, 1.0), 0.01, method="rk4")
    y_true = a * ts
    err = np.abs(ys[:, 0] - y_true).max()
    assert err < 1e-3


def test_harmonic_oscillator_phase():
    # y=[x, v], dx/dt=v, dv/dt=-x
    def f(t, y):
        x, v = y
        return np.array([v, -x])

    ts, ys = integrate(f, np.array([1.0, 0.0]), (0.0, 2 * np.pi), 0.01, method="rk4")
    x = ys[:, 0]
    # After one period, x should be close to initial value
    assert abs(x[-1] - 1.0) < 1e-2