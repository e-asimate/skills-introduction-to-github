from __future__ import annotations
import numpy as np
import pandas as pd

def mass_spring_to_xyz(ys: np.ndarray) -> np.ndarray:
    # 1D motion along X axis; embed into 3D as (x, 0, 0)
    x = ys[:, 0]
    zeros = np.zeros_like(x)
    return np.stack([x, zeros, zeros], axis=1)

def double_pendulum_to_xyz(ys: np.ndarray, L1: float, L2: float):
    # Convert angles to Cartesian (planar in XZ, Y=0)
    th1, th2 = ys[:, 0], ys[:, 2]
    x1 = L1 * np.sin(th1); z1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2); z2 = z1 - L2 * np.cos(th2)
    y0 = np.zeros_like(x1)
    bob1 = np.stack([x1, y0, z1], axis=1)
    bob2 = np.stack([x2, y0, z2], axis=1)
    return bob1, bob2

def build_predicted_parabola(v0: float, theta_deg: float, phi_deg: float,
                             landing_time: float, max_height: float,
                             range_x: float, range_y: float,
                             g_default: float = 9.81, n: int = 200):
    # Approximate 3D path from ML summary for projectile
    theta = np.deg2rad(theta_deg); phi = np.deg2rad(phi_deg)
    vx0 = v0 * np.cos(theta) * np.cos(phi)
    vy0 = v0 * np.cos(theta) * np.sin(phi)
    vz0 = v0 * np.sin(theta)
    lt = max(landing_time, 1e-6)
    g_eff = g_default
    if max_height and max_height > 1e-6 and vz0 > 1e-6:
        g_eff = max(vz0**2 / (2.0 * max_height), 1e-6)
    vx_eff = range_x / lt if abs(lt) > 1e-9 else vx0
    vy_eff = range_y / lt if abs(lt) > 1e-9 else vy0
    ts = np.linspace(0.0, lt, n)
    x = vx_eff * ts; y = vy_eff * ts; z = vz0 * ts - 0.5 * g_eff * ts**2
    return ts, np.stack([x, y, z], axis=1)
Wire into app/streamlit_app.py
Update imports:
from engine.utils import to_dataframe, projectile_energy, mass_spring_energy, mass_spring_to_xyz, double_pendulum_to_xyz, build_predicted_parabola
from viz.plots import traj3d, timeseries, energy_plot, traj3d_animated, double_pendulum_animated
import plotly.io as pio  # for HTML export


def to_dataframe(ts: np.ndarray, ys: np.ndarray, labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(ys, columns=labels)
    df.insert(0, "t", ts)
    return df


def projectile_energy(ys: np.ndarray, mass: float, g: float = 9.81) -> dict[str, np.ndarray]:
    vx, vy, vz = ys[:, 3], ys[:, 4], ys[:, 5]
    z = ys[:, 2]
    ke = 0.5 * mass * (vx ** 2 + vy ** 2 + vz ** 2)
    pe = mass * g * np.maximum(z, 0.0)
    return {"KE": ke, "PE": pe, "TE": ke + pe}


def mass_spring_energy(ys: np.ndarray, m: float, k: float) -> dict[str, np.ndarray]:
    x, v = ys[:, 0], ys[:, 1]
    ke = 0.5 * m * v ** 2
    pe = 0.5 * k * x ** 2
    return {"KE": ke, "PE": pe, "TE": ke + pe}
