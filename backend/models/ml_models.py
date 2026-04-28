"""
BizMind ML Models
Standalone ML model classes used by agents for predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class ChurnPredictor:
    """
    Random Forest-based churn predictor.
    Predicts which customers are likely to churn in the next 30 days.
    """
    def __init__(self):
        self.model = None
        self.feature_names = [
            "last_purchase_days_ago",
            "avg_order_value",
            "purchase_frequency",
            "total_orders",
            "nps_score",
        ]
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X = df[self.feature_names].fillna(0)
        y = df["is_churned"].astype(int)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
        )
        self.model.fit(X, y)
        self.is_trained = True

        # Cross-validation score
        scores = cross_val_score(self.model, X, y, cv=5, scoring="roc_auc")

        return {
            "accuracy": round(self.model.score(X, y), 3),
            "roc_auc_cv": round(scores.mean(), 3),
            "feature_importances": dict(zip(
                self.feature_names,
                self.model.feature_importances_.round(3)
            )),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        X = df[self.feature_names].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        result = df[["customer_id", "segment"]].copy()
        result["churn_probability"] = proba.round(3)
        result["churn_label"] = (proba > 0.5).astype(int)
        result = result.sort_values("churn_probability", ascending=False)
        return result

    def get_high_risk(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        predictions = self.predict(df)
        return predictions[predictions["churn_probability"] >= threshold]


class DemandForecaster:
    """
    XGBoost-based demand forecaster.
    Predicts daily units sold for each product.
    """
    def __init__(self):
        self.models = {}  # one model per product
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
        df["day_num"] = (df["date"] - df["date"].min()).dt.days
        return df

    def train(self, sales_df: pd.DataFrame) -> Dict:
        try:
            import xgboost as xgb
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor as xgb

        feature_cols = ["day_of_week", "month", "day_of_month", "week_of_year",
                        "is_weekend", "is_holiday_season", "day_num"]

        df = self._build_features(sales_df)
        products = df["product_name"].unique()
        results = {}

        for product in products:
            prod_data = df[df["product_name"] == product].groupby("date").agg(
                units_sold=("units_sold", "sum")
            ).reset_index()
            prod_data = self._build_features(prod_data)

            X = prod_data[feature_cols]
            y = prod_data["units_sold"]

            try:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X, y)
                self.models[product] = model
                results[product] = {"status": "trained", "samples": len(y)}
            except Exception as e:
                results[product] = {"status": "error", "error": str(e)}

        self.is_trained = True
        return {"products_trained": len(self.models), "details": results}

    def forecast(self, product_name: str, days_ahead: int = 30) -> List[Dict]:
        if product_name not in self.models:
            raise ValueError(f"No model trained for product: {product_name}")

        import datetime
        model = self.models[product_name]
        base_date = datetime.datetime.now()
        forecasts = []

        for i in range(days_ahead):
            future_date = base_date + datetime.timedelta(days=i + 1)
            features = {
                "day_of_week": future_date.weekday(),
                "month": future_date.month,
                "day_of_month": future_date.day,
                "week_of_year": future_date.isocalendar()[1],
                "is_weekend": int(future_date.weekday() >= 5),
                "is_holiday_season": int(future_date.month in [11, 12]),
                "day_num": 200 + i,
            }
            X = pd.DataFrame([features])
            pred = max(0, float(model.predict(X)[0]))
            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_units": round(pred, 1),
            })

        return forecasts


class RevenueForecaster:
    """
    Polynomial trend + seasonality revenue forecaster.
    Uses historical data to project future revenue with confidence intervals.
    """
    def __init__(self):
        self.coeffs = None
        self.weekly_factors = {}
        self.base_date = None
        self.std_residual = 0

    def fit(self, sales_df: pd.DataFrame):
        df = sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["revenue"].sum().reset_index().sort_values("date")

        self.base_date = daily["date"].min()
        daily["day_num"] = (daily["date"] - self.base_date).dt.days
        daily["day_of_week"] = daily["date"].dt.dayofweek

        x = daily["day_num"].values
        y = daily["revenue"].values

        self.coeffs = np.polyfit(x, y, 2)
        trend = np.poly1d(self.coeffs)
        residuals = y - trend(x)
        self.std_residual = np.std(residuals)

        overall_mean = daily["revenue"].mean()
        weekly_raw = daily.groupby("day_of_week")["revenue"].mean()
        self.weekly_factors = (weekly_raw / overall_mean).to_dict()
        return self

    def predict(self, days_ahead: int = 30, confidence: float = 0.95) -> List[Dict]:
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")

        import datetime
        trend = np.poly1d(self.coeffs)
        base = datetime.datetime.now()

        # Z-score for confidence interval
        z = 1.96 if confidence == 0.95 else 1.645

        forecasts = []
        for i in range(days_ahead):
            future_date = base + datetime.timedelta(days=i + 1)
            day_num = (future_date - datetime.datetime(
                self.base_date.year, self.base_date.month, self.base_date.day
            )).days
            factor = self.weekly_factors.get(future_date.weekday(), 1.0)
            point = max(0, float(trend(day_num) * factor))

            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_revenue": round(point, 2),
                "lower_bound": round(max(0, point - z * self.std_residual), 2),
                "upper_bound": round(point + z * self.std_residual, 2),
            })

        return forecasts

    def summary(self) -> Dict:
        if self.coeffs is None:
            return {}
        return {
            "model": "Polynomial degree 2 + weekly seasonality",
            "trend_coefficients": self.coeffs.tolist(),
            "std_residual": round(self.std_residual, 2),
            "weekly_factors": {int(k): round(v, 3) for k, v in self.weekly_factors.items()},
        }


class AnomalyDetector:
    """
    Simple Z-score based anomaly detector for sales and KPI data.
    Flags data points that deviate significantly from expected values.
    """
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold

    def detect(self, series: pd.Series, label: str = "value") -> List[Dict]:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return []

        anomalies = []
        for i, (idx, val) in enumerate(series.items()):
            z_score = abs((val - mean) / std)
            if z_score > self.threshold:
                anomalies.append({
                    "index": str(idx),
                    "value": round(float(val), 2),
                    "z_score": round(float(z_score), 2),
                    "direction": "spike" if val > mean else "dip",
                    "deviation_pct": round(abs(val - mean) / mean * 100, 1),
                })
        return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
