"""Singular Unified ML Anomaly Detection Layer for CREW.
Trains a single unified RandomForestRegressor across multi-domain telemetry
(Energy, Water, Waste) learning cross-resource interactions across facility categories
(College, Hostel, Canteen) to predict consumption and flag deviations > 20%.

Deliverable: DataFrame with [predicted, deviation_pct, flagged] columns.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from crew.config import DEVIATION_FLAG_THRESHOLD_PCT


class CampusResourceDetector:
    """Singular unified ML detector predicting across Energy, Water, and Waste telemetry."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """One-hot encodes facility category, building, and metric along with numerical features."""
        # Ensure facility_category is populated if loading older data
        cat_df = df.copy()
        if "facility_category" not in cat_df.columns:
            cat_df["facility_category"] = cat_df["building_type"].map({
                "academic": "college",
                "lab": "college",
                "hostel": "hostel",
                "canteen": "canteen",
            }).fillna("college")

        cat_features = cat_df[["building_id", "building_type", "facility_category", "metric"]].values
        num_features = cat_df[["occupancy", "hour"]].values

        if fit:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_cat = self.encoder.fit_transform(cat_features)
        else:
            if self.encoder is None:
                raise ValueError("Encoder has not been fitted yet.")
            encoded_cat = self.encoder.transform(cat_features)

        return np.hstack([encoded_cat, num_features])

    def fit(self, df: pd.DataFrame) -> "CampusResourceDetector":
        """Trains a singular unified RandomForestRegressor across all resource domains."""
        # Filter to normal baseline data so model learns clean expected profile
        baseline_df = df[~df["known_inefficient"]].copy()

        # Prepare unified feature matrix
        X_all = self._prepare_features(baseline_df, fit=True)
        y_all = baseline_df["reading"].values

        # 80/20 train/test evaluation split
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X_all, y_all, baseline_df, test_size=0.20, random_state=self.random_state
        )

        # Train singular unified regressor
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=18,
            min_samples_split=4,
            max_samples=0.5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate overall accuracy
        y_pred = self.model.predict(X_test)
        self.metrics["overall"] = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 2),
            "sample_count": len(baseline_df),
        }

        # Calculate per-domain breakdown accuracy
        test_df_eval = df_test.copy()
        test_df_eval["y_true"] = y_test
        test_df_eval["y_pred"] = y_pred

        for m in ["energy", "water", "waste"]:
            m_sub = test_df_eval[test_df_eval["metric"] == m]
            if len(m_sub) > 0:
                self.metrics[m] = {
                    "r2": round(float(r2_score(m_sub["y_true"], m_sub["y_pred"])), 4),
                    "mae": round(float(mean_absolute_error(m_sub["y_true"], m_sub["y_pred"])), 2),
                    "rmse": round(float(root_mean_squared_error(m_sub["y_true"], m_sub["y_pred"])), 2),
                    "sample_count": len(m_sub),
                }

        return self

    def predict_and_flag(
        self,
        df: pd.DataFrame,
        threshold_pct: float = DEVIATION_FLAG_THRESHOLD_PCT,
    ) -> pd.DataFrame:
        """Predicts expected consumption using the singular unified model and flags deviations."""
        if self.model is None or self.encoder is None:
            raise RuntimeError("Model must be trained or loaded before calling predict_and_flag.")

        result_df = df.copy()
        X = self._prepare_features(result_df, fit=False)
        preds = self.model.predict(X)

        result_df["predicted"] = np.round(preds, 2)
        pred_safe = np.where(result_df["predicted"] <= 0, 0.01, result_df["predicted"])
        result_df["deviation_pct"] = np.round(
            ((result_df["reading"] - result_df["predicted"]) / pred_safe) * 100.0,
            2,
        )
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

    @classmethod
    def load_or_train(
        cls,
        data_path: str = "data/campus_data.csv",
        model_path: str = "models/detector.joblib",
    ) -> "CampusResourceDetector":
        """Loads cached model if present; otherwise fits and saves."""
        if Path(model_path).exists():
            try:
                return cls.load(model_path)
            except Exception:
                pass

        df = pd.read_csv(data_path)
        detector = cls()
        detector.fit(df)
        detector.save(model_path)
        return detector


def run_detection_pipeline(
    data_path: str = "data/campus_data.csv",
    model_path: str = "models/detector.joblib",
    threshold_pct: float = DEVIATION_FLAG_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Convenience pipeline runner."""
    detector = CampusResourceDetector.load_or_train(data_path=data_path, model_path=model_path)
    df = pd.read_csv(data_path)
    return detector.predict_and_flag(df, threshold_pct=threshold_pct)
