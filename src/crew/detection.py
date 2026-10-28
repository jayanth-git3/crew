"""Baseline Anomaly Detection Layer for CREW.
Uses dual RandomForestRegressors (Energy & Water) trained on normal baseline data
to predict expected resource consumption and flag deviations > 20%.

Deliverable: DataFrame with [predicted, deviation_pct, flagged] columns.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from crew.config import DEVIATION_FLAG_THRESHOLD_PCT


class CampusResourceDetector:
    """Detection pipeline implementing dual RandomForestRegressors for Energy & Water."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.energy_model: Optional[RandomForestRegressor] = None
        self.water_model: Optional[RandomForestRegressor] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_cols: list[str] = ["building_type", "occupancy", "hour"]

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encodes building_type along with numerical features (occupancy, hour)."""
        cat_features = df[["building_type"]].values
        num_features = df[["occupancy", "hour"]].values

        if fit:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_cat = self.encoder.fit_transform(cat_features)
        else:
            if self.encoder is None:
                raise ValueError("Encoder has not been fitted yet.")
            encoded_cat = self.encoder.transform(cat_features)

        return np.hstack([encoded_cat, num_features])

    def fit(self, df: pd.DataFrame) -> "CampusResourceDetector":
        """Trains dual RandomForestRegressors strictly on baseline normal readings."""
        # Train strictly on normal (non-anomalous) rows so model learns clean expected profile
        baseline_df = df[~df["known_inefficient"]].copy()

        # Fit feature encoder
        self._prepare_features(baseline_df, fit=True)

        # Train Energy model
        energy_data = baseline_df[baseline_df["metric"] == "energy"]
        if len(energy_data) > 0:
            X_energy = self._prepare_features(energy_data, fit=False)
            y_energy = energy_data["reading"].values
            self.energy_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.energy_model.fit(X_energy, y_energy)

        # Train Water model
        water_data = baseline_df[baseline_df["metric"] == "water"]
        if len(water_data) > 0:
            X_water = self._prepare_features(water_data, fit=False)
            y_water = water_data["reading"].values
            self.water_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.water_model.fit(X_water, y_water)

        return self

    def predict_and_flag(
        self,
        df: pd.DataFrame,
        threshold_pct: float = DEVIATION_FLAG_THRESHOLD_PCT,
    ) -> pd.DataFrame:
        """Predicts expected consumption and flags deviations > threshold_pct.

        Adds columns:
        - predicted: float
        - deviation_pct: float
        - flagged: bool
        """
        if self.energy_model is None or self.water_model is None or self.encoder is None:
            raise RuntimeError("Models must be trained or loaded before calling predict_and_flag.")

        result_df = df.copy()
        result_df["predicted"] = 0.0

        # Predict for Energy rows
        energy_idx = result_df[result_df["metric"] == "energy"].index
        if len(energy_idx) > 0:
            X_energy = self._prepare_features(result_df.loc[energy_idx], fit=False)
            preds_energy = self.energy_model.predict(X_energy)
            result_df.loc[energy_idx, "predicted"] = np.round(preds_energy, 2)

        # Predict for Water rows
        water_idx = result_df[result_df["metric"] == "water"].index
        if len(water_idx) > 0:
            X_water = self._prepare_features(result_df.loc[water_idx], fit=False)
            preds_water = self.water_model.predict(X_water)
            result_df.loc[water_idx, "predicted"] = np.round(preds_water, 2)

        # Prevent division by zero
        pred_safe = np.where(result_df["predicted"] <= 0, 0.01, result_df["predicted"])
        result_df["deviation_pct"] = np.round(
            ((result_df["reading"] - result_df["predicted"]) / pred_safe) * 100.0,
            2,
        )

        # Flag row if actual reading is more than threshold_pct above predicted
        result_df["flagged"] = result_df["deviation_pct"] > threshold_pct

        return result_df

    def save(self, filepath: str = "models/detector.joblib") -> str:
        """Persists trained detector artifact."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return str(p.resolve())

    @staticmethod
    def load(filepath: str = "models/detector.joblib") -> "CampusResourceDetector":
        """Loads persisted detector artifact."""
        return joblib.load(filepath)


def run_detection_pipeline(
    data_path: str = "data/campus_data.csv",
    model_path: str = "models/detector.joblib",
    threshold_pct: float = DEVIATION_FLAG_THRESHOLD_PCT,
) -> pd.DataFrame:
    """End-to-end detection convenience runner."""
    df = pd.read_csv(data_path)
    detector = CampusResourceDetector()
    detector.fit(df)
    detector.save(model_path)
    flagged_df = detector.predict_and_flag(df, threshold_pct=threshold_pct)
    return flagged_df


if __name__ == "__main__":
    flagged = run_detection_pipeline()
    total_flags = flagged["flagged"].sum()
    print(f"Detection completed. Total rows: {len(flagged)}, Flagged anomalies: {total_flags}")
    print("\nSample flagged rows:")
    print(flagged[flagged["flagged"]][["building_name", "hour", "metric", "reading", "predicted", "deviation_pct"]].head(10))
