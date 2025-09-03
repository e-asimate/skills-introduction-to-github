import numpy as np
import pandas as pd
from engine.systems import Projectile3D
from engine.simulate import run_system, landing_time_from_trajectory


def generate_projectile_dataset(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        mass = rng.uniform(0.05, 0.3)
        cd = rng.uniform(0.0, 0.6)
        area = rng.uniform(0.001, 0.008)
        v0 = rng.uniform(5.0, 70.0)
        theta = rng.uniform(5.0, 80.0)
        phi = rng.uniform(-180.0, 180.0)
        vx = v0 * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        vy = v0 * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        vz = v0 * np.sin(np.deg2rad(theta))
        sys = Projectile3D(mass=mass, drag_coeff=cd, area=area)
        ts, ys = run_system(sys, [0, 0, 0, vx, vy, vz], t_span=(0, 15), h=0.01)
        lt = landing_time_from_trajectory(ts, ys)
        xr, yr = ys[-1, 0], ys[-1, 1]
        max_z = ys[:, 2].max()
        rows.append(
            dict(
                mass=mass,
                cd=cd,
                area=area,
                v0=v0,
                theta=theta,
                phi=phi,
                landing_time=lt if lt is not None else np.nan,
                range_x=xr,
                range_y=yr,
                max_height=max_z,
            )
        )
    df = pd.DataFrame(rows).dropna()
    return df