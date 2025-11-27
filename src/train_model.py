# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_CSV = "data/simulated_sensor_data.csv"
MODEL_PATH = "models/regression_model.joblib"
METRICS_PATH = "models/metrics.txt"

def train(save_model=True):
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    # Features and target
    X = df[["soil_moisture","soil_ph","air_temp","humidity","irrigation_ml","fertilizer_kg"]]
    y = df["yield_ton_per_ha"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if save_model:
        joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        f.write(f"MAE:{mae}\nMSE:{mse}\nRMSE:{rmse}\nR2:{r2}\n")
    print("Training complete. Metrics:")
    print(f"MAE={mae:.4f}, MSE={mse:.6f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    return pipe, {"mae":mae,"mse":mse,"rmse":rmse,"r2":r2}

if __name__ == "__main__":
    train()
