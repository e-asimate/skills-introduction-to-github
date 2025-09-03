# Physics Visualization & Prediction Engine

Interactive 3D physics simulator with ML prediction. Includes RK4/Euler integrators, projectile motion, mass–spring oscillator, and a double pendulum. Streamlit + Plotly UI, dataset generation, and baseline ML models.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Structure
```
physics-viz-ml/
├─ app/
├─ engine/
├─ viz/
├─ ml/
├─ data/
├─ tests/
```

## ML
- Generate dataset: use `ml/dataset.py:generate_projectile_dataset`.
- Train baseline: `python ml/train_sklearn.py` creates `model_rf.pkl`.
- (Optional) PyTorch MLP: `python ml/train_torch.py` saves `mlp.pt`.

## Notes
- For PNG export of plots, install Kaleido (already in requirements).
- Time step size strongly affects accuracy; prefer 0.005–0.01.