import plotly.graph_objects as go
import numpy as np

def traj3d_animated(ts, xyz, title="Animated 3D Trajectory", show_ground=True, max_frames=200):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    n = len(ts)
    if n <= 1:
        fig = go.Figure()
        return fig
    stride = max(1, n // max_frames)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    # Initial traces
    line0 = go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="lines", name="path")
    head0 = go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="markers", name="head")

    frames = []
    for i in indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=x[: i + 1], y=y[: i + 1], z=z[: i + 1], mode="lines", name="path"),
                    go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]], mode="markers", name="head"),
                ],
                name=str(i),
            )
        )

    data = [line0, head0]
    if show_ground:
        xr = np.linspace(min(0, x.min()), max(0, x.max()), 2)
        yr = np.linspace(min(0, y.min()), max(0, y.max()), 2)
        X, Y = np.meshgrid(xr, yr)
        Z = np.zeros_like(X)
        ground = go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.25, name="ground")
        data.append(ground)

    layout = go.Layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05, y=1.15, xanchor="left", yanchor="top",
                direction="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[[f.name for f in frames],
                               {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate",
                         args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                         label=str(i))
                    for i in indices
                ],
                x=0.05, y=1.05, xanchor="left", yanchor="top",
            )
        ],
    )
    fig = go.Figure(data=data, layout=layout, frames=frames)
    return fig


def double_pendulum_animated(ts, xyz1, xyz2, title="Double Pendulum (Animated)"):
    # xyz1: bob1 positions; xyz2: bob2 positions; both shape (n, 3)
    x1, y1, z1 = xyz1[:, 0], xyz1[:, 1], xyz1[:, 2]
    x2, y2, z2 = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    n = len(ts)
    stride = max(1, n // 200)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    # rods as line segments between origin->bob1 and bob1->bob2
    rod1_0 = go.Scatter3d(x=[0, x1[0]], y=[0, y1[0]], z=[0, z1[0]], mode="lines", name="rod1")
    rod2_0 = go.Scatter3d(x=[x1[0], x2[0]], y=[y1[0], y2[0]], z=[z1[0], z2[0]], mode="lines", name="rod2")
    bob1_0 = go.Scatter3d(x=[x1[0]], y=[y1[0]], z=[z1[0]], mode="markers", name="bob1")
    bob2_0 = go.Scatter3d(x=[x2[0]], y=[y2[0]], z=[z2[0]], mode="markers", name="bob2")
    path2_0 = go.Scatter3d(x=[x2[0]], y=[y2[0]], z=[z2[0]], mode="lines", name="path2")

    frames = []
    for i in indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=[0, x1[i]], y=[0, y1[i]], z=[0, z1[i]], mode="lines", name="rod1"),
                    go.Scatter3d(x=[x1[i], x2[i]], y=[y1[i], y2[i]], z=[z1[i], z2[i]], mode="lines", name="rod2"),
                    go.Scatter3d(x=[x1[i]], y=[y1[i]], z=[z1[i]], mode="markers", name="bob1"),
                    go.Scatter3d(x=[x2[i]], y=[y2[i]], z=[z2[i]], mode="markers", name="bob2"),
                    go.Scatter3d(x=x2[: i + 1], y=y2[: i + 1], z=z2[: i + 1], mode="lines", name="path2"),
                ],
                name=str(i),
            )
        )

    layout = go.Layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05, y=1.15, xanchor="left", yanchor="top",
                direction="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[[f.name for f in frames],
                               {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                ],
            )
        ],
    )
    fig = go.Figure(data=[rod1_0, rod2_0, bob1_0, bob2_0, path2_0], layout=layout, frames=frames)
    return fig
Add coordinate builders to engine/utils.py Append:
import numpy as np

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
