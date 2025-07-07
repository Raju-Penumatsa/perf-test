from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class AnomalyDetector:
    """
    AnomalyDetector to detect anomaly in data, this uses sklearn IsolationForest model
    """
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, data: np.ndarray):
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        return self.model.predict(data)  # 1 = normal, -1 = anomaly

    def detect_anomalies(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        X = df[features].values
        self.fit(X)
        preds = self.predict(X)
        df["anomaly"] = preds
        return df

