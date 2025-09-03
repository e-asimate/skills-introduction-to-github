from __future__ import annotations
import numpy as np

GRAV = 9.81


class Projectile3D:
    """Projectile with optional quadratic drag in 3D.

    State y = [x, y, z, vx, vy, vz].
    """

    def __init__(self, mass: float = 0.145, drag_coeff: float = 0.0, rho: float = 1.225, area: float = 0.0042):
        self.m = float(mass)
        self.cd = float(drag_coeff)
        self.rho = float(rho)
        self.A = float(area)

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        x, y_pos, z, vx, vy, vz = y
        v = np.array([vx, vy, vz], dtype=float)
        speed = np.linalg.norm(v)
        drag = 0.5 * self.rho * self.cd * self.A * speed * v
        ax, ay, az = -(drag / self.m)
        az -= GRAV
        return np.array([vx, vy, vz, ax, ay, az], dtype=float)


class MassSpring:
    """1D mass-spring-damper. State y=[x, v]."""

    def __init__(self, k: float = 10.0, m: float = 1.0, c: float = 0.1):
        self.k, self.m, self.c = float(k), float(m), float(c)

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        x, v = y
        a = -(self.k / self.m) * x - (self.c / self.m) * v
        return np.array([v, a], dtype=float)


class DoublePendulum:
    """Chaotic double pendulum (planar). State y=[th1, w1, th2, w2]."""

    def __init__(self, m1: float = 1.0, m2: float = 1.0, L1: float = 1.0, L2: float = 1.0):
        self.m1, self.m2, self.L1, self.L2 = float(m1), float(m2), float(L1), float(L2)

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        th1, w1, th2, w2 = y
        m1, m2, L1, L2 = self.m1, self.m2, self.L1, self.L2
        g = GRAV
        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        den2 = (L2 / L1) * den1
        a1 = (
            m2 * L1 * w1 ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(th2) * np.cos(delta)
            + m2 * L2 * w2 ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
        ) / den1
        a2 = (
            -m2 * L2 * w2 ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * (g * np.sin(th1) * np.cos(delta) - L1 * w1 ** 2 * np.sin(delta) - g * np.sin(th2))
        ) / den2
        return np.array([w1, a1, w2, a2], dtype=float)