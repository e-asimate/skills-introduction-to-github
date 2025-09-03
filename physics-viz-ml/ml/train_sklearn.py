import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["mass", "cd", "area", "v0", "theta", "phi"]
TARGETS = ["landing_time", "range_x", "range_y", "max_height"]


def train_random_forest(df: pd.DataFrame, out_path: str = "model_rf.pkl") -> float:
    X = df[FEATURES]
    y = df[TARGETS]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=0))
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(Xtr, ytr)
    score = model.score(Xte, yte)
    joblib.dump({"pipeline": model, "score": score}, out_path)
    return score


if __name__ == "__main__":
    from dataset import generate_projectile_dataset

    df = generate_projectile_dataset(n=3000)
    score = train_random_forest(df)
    print("R^2 on holdout:", score)