from __future__ import annotations
import numpy as np
import pandas as pd


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