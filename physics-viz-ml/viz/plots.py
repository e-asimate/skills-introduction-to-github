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
