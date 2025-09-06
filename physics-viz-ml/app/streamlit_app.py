import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine.systems import Projectile3D, MassSpring, DoublePendulum
from engine.simulate import run_system, landing_time_from_trajectory
from engine.utils import to_dataframe, projectile_energy, mass_spring_energy
from viz.plots import traj3d, timeseries, energy_plot

st.set_page_config(page_title="Physics Viz + ML", layout="wide")
st.title("Physics Visualization & Prediction Engine")

with st.sidebar:
    st.header("Controls")
    sim_choice = st.selectbox("System", ["Projectile (3D)", "Mass–Spring", "Double Pendulum"]) 
    method = st.selectbox("Integrator", ["rk4", "euler"])  
    T = st.slider("Simulation time (s)", 1.0, 20.0, 8.0)
    h = st.select_slider("Time step (s)", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.01)
    st.markdown("---")
    preset = st.selectbox("Presets", ["Custom", "High Drag", "No Drag", "Short Run", "Accurate (small dt)"])
    if preset == "High Drag":
        h = 0.01
    elif preset == "No Drag":
        h = 0.01
    elif preset == "Short Run":
        T = 4.0
    elif preset == "Accurate (small dt)":
        h = 0.005

# Attempt to load ML model if available
model_info = None
try:
    model_info = joblib.load("model_rf.pkl")
except Exception:
    pass

if sim_choice == "Projectile (3D)":
    st.subheader("Projectile (with optional drag)")
    col = st.columns(4)
    with col[0]:
        mass = st.number_input("Mass (kg)", 0.01, 5.0, 0.145, 0.005)
        cd = st.number_input("Drag coefficient C_d", 0.0, 2.0, 0.25, 0.01)
        area = st.number_input("Area (m²)", 0.0001, 0.01, 0.0042, 0.0001)
    with col[1]:
        v0 = st.number_input("Speed |v0| (m/s)", 0.0, 200.0, 30.0, 0.1)
        theta = st.slider("Elevation θ (deg)", 0, 89, 35)
        phi = st.slider("Azimuth φ (deg)", -180, 180, 0)
    with col[2]:
        show_energy = st.checkbox("Show energy overlay", value=True)
        export_png = st.checkbox("Enable PNG export", value=False)
    with col[3]:
        st.markdown(" ")
        st.markdown(" ")
        run_btn = st.button("Run Simulation", use_container_width=True)

    vx = v0 * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    vy = v0 * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    vz = v0 * np.sin(np.deg2rad(theta))

    proj = Projectile3D(mass=mass, drag_coeff=cd, area=area)
    y0 = np.array([0, 0, 0, vx, vy, vz])

    if run_btn:
        ts, ys = run_system(proj, y0, t_span=(0.0, T), h=h, method=method)
        lt = landing_time_from_trajectory(ts, ys)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig3d = traj3d(ts, ys[:, :3], title="Projectile Trajectory")
            st.plotly_chart(fig3d, use_container_width=True)
        with col2:
            st.plotly_chart(timeseries(ts, ys[:, 3:], labels=["vx", "vy", "vz"], title="Velocity Components"), use_container_width=True)
            st.metric("Estimated landing time (s)", f"{lt:.3f}" if lt else "n/a")
            if show_energy:
                e = projectile_energy(ys, mass)
                st.plotly_chart(energy_plot(ts, e, title="Energy vs Time"), use_container_width=True)
        df = to_dataframe(ts, ys, labels=["x", "y", "z", "vx", "vy", "vz"])
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name="projectile_run.csv")
        if export_png:
            try:
                import plotly.io as pio

                png = pio.to_image(fig3d, format="png", scale=2)
                st.download_button("Download Trajectory PNG", png, file_name="trajectory.png")
            except Exception as e:
                st.info("PNG export requires kaleido. Install and restart if unavailable.")

    # ML quick prediction panel
    st.markdown("---")
    st.subheader("Predict (ML) vs Simulate")
    if model_info is not None:
        pipe = model_info["pipeline"]
        X = pd.DataFrame([[mass, cd, area, v0, theta, phi]], columns=["mass", "cd", "area", "v0", "theta", "phi"])
        pred = pipe.predict(X)[0]
        st.write({"landing_time": float(pred[0]), "range_x": float(pred[1]), "range_y": float(pred[2]), "max_height": float(pred[3])})
    else:
        st.caption("Train and save model_rf.pkl to enable instant predictions.")

elif sim_choice == "Mass–Spring":
    st.subheader("Mass–Spring–Damper")
    col = st.columns(3)
    with col[0]:
        m = st.number_input("Mass (kg)", 0.01, 10.0, 1.0, 0.01)
        k = st.number_input("Spring k (N/m)", 0.1, 1000.0, 10.0, 0.1)
        c = st.number_input("Damping c (N·s/m)", 0.0, 50.0, 0.1, 0.01)
    with col[1]:
        x0 = st.number_input("Initial x (m)", -5.0, 5.0, 1.0, 0.1)
        v0 = st.number_input("Initial v (m/s)", -20.0, 20.0, 0.0, 0.1)
    with col[2]:
        show_energy = st.checkbox("Show energy overlay", value=True)
        run_btn = st.button("Run Simulation", use_container_width=True)

    if run_btn:
        sys = MassSpring(k=k, m=m, c=c)
        ts, ys = run_system(sys, [x0, v0], t_span=(0.0, T), h=h, method=method)
        st.plotly_chart(timeseries(ts, ys, labels=["x", "v"], title="Mass–Spring State"), use_container_width=True)
        if show_energy:
            e = mass_spring_energy(ys, m, k)
            st.plotly_chart(energy_plot(ts, e, title="Energy vs Time"), use_container_width=True)
        df = to_dataframe(ts, ys, labels=["x", "v"])
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name="mass_spring_run.csv")

else:
    st.subheader("Double Pendulum (planar)")
    col = st.columns(3)
    with col[0]:
        L1 = st.number_input("L1 (m)", 0.1, 5.0, 1.0, 0.1)
        L2 = st.number_input("L2 (m)", 0.1, 5.0, 1.0, 0.1)
    with col[1]:
        th1 = st.slider("θ1 (deg)", -180, 180, 120)
        th2 = st.slider("θ2 (deg)", -180, 180, -10)
    with col[2]:
        w1 = st.number_input("ω1 (rad/s)", -20.0, 20.0, 0.0, 0.1)
        w2 = st.number_input("ω2 (rad/s)", -20.0, 20.0, 0.0, 0.1)
        run_btn = st.button("Run Simulation", use_container_width=True)

    if run_btn:
        sys = DoublePendulum(L1=L1, L2=L2)
        y0 = np.array([np.deg2rad(th1), w1, np.deg2rad(th2), w2])
        ts, ys = run_system(sys, y0, t_span=(0.0, T), h=h, method=method)
        st.plotly_chart(timeseries(ts, ys, labels=["θ1", "ω1", "θ2", "ω2"], title="Double Pendulum State"), use_container_width=True)

st.markdown("---")
with st.expander("Help: How to use this app"):
    st.markdown(
        "- Pick a system, set initial conditions, and click Run.\n"
        "- For projectiles, toggle energy overlay and compare drag vs no-drag.\n"
        "- Use small time steps (e.g., 0.005–0.01) for better accuracy.\n"
        "- Use Download CSV to export data for analysis."
    )
