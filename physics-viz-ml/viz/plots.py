import numpy as np
import plotly.graph_objects as go

def traj3d(ts, xyz, title="3D Trajectory", show_ground=True):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    path = go.Scatter3d(x=x, y=y, z=z, mode="lines", name="path")
    pts = go.Scatter3d(
        x=[x[0], x[-1]], y=[y[0], y[-1]], z=[z[0], z[-1]], mode="markers", name="start/end"
    )
    data = [path, pts]
    if show_ground:
        xr = np.linspace(min(0, x.min()), max(0, x.max()), 2)
        yr = np.linspace(min(0, y.min()), max(0, y.max()), 2)
        X, Y = np.meshgrid(xr, yr)
        Z = np.zeros_like(X)
        ground = go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.25, name="ground")
        data.append(ground)
    layout = go.Layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), title=title
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def timeseries(ts, ys, labels, title="Timeseries"):
    traces = []
    for i, lab in enumerate(labels):
        traces.append(go.Scatter(x=ts, y=ys[:, i], mode="lines", name=lab))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, xaxis_title="t (s)")
    return fig


def energy_plot(ts, energy_dict, title="Energy"):
    traces = []
    for name, arr in energy_dict.items():
        traces.append(go.Scatter(x=ts, y=arr, mode="lines", name=name))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, xaxis_title="t (s)", yaxis_title="Energy (J)")
    return fig
