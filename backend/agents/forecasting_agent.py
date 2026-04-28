"""
Forecasting Agent — Predicts revenue, demand, and customer growth using ML models.
"""
import json
from typing import Dict, List
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from data import get_sales_data, get_customer_data


class ForecastingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ForecastBot",
            role="Predictive Analytics & Forecasting Specialist",
            expertise="Revenue forecasting, demand prediction, trend analysis, time-series modeling, growth projections"
        )

    def _forecast_revenue(self, df: pd.DataFrame, days_ahead: int = 30) -> Dict:
        """Simple linear regression + seasonality forecast."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["revenue"].sum().reset_index()
        daily = daily.sort_values("date")

        # Feature engineering
        daily["day_num"] = (daily["date"] - daily["date"].min()).dt.days
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month

        # Simple polynomial trend + weekly seasonality
        x = daily["day_num"].values
        y = daily["revenue"].values

        # Fit polynomial trend
        coeffs = np.polyfit(x, y, 2)
        trend = np.poly1d(coeffs)

        # Forecast next N days
        last_day = daily["day_num"].max()
        future_days = np.arange(last_day + 1, last_day + days_ahead + 1)
        forecast_values = trend(future_days)

        # Add weekly seasonality from historical data
        weekly_pattern = daily.groupby("day_of_week")["revenue"].mean()
        overall_mean = daily["revenue"].mean()
        weekly_factors = (weekly_pattern / overall_mean).to_dict()

        forecast_dates = [
            (daily["date"].max() + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(days_ahead)
        ]

        forecast_final = []
        for i, (date, val) in enumerate(zip(forecast_dates, forecast_values)):
            day_of_week = (daily["date"].max() + pd.Timedelta(days=i+1)).dayofweek
            factor = weekly_factors.get(day_of_week, 1.0)
            adjusted = max(0, val * factor)
            forecast_final.append({"date": date, "predicted_revenue": round(float(adjusted), 2)})

        monthly_forecast = sum(f["predicted_revenue"] for f in forecast_final)
        last_month_actual = daily.tail(30)["revenue"].sum()
        growth_pct = (monthly_forecast - last_month_actual) / last_month_actual * 100

        return {
            "daily_forecast": forecast_final,
            "monthly_forecast_total": round(monthly_forecast, 2),
            "last_month_actual": round(last_month_actual, 2),
            "projected_growth_pct": round(growth_pct, 1),
        }

    def _forecast_churn(self, df: pd.DataFrame) -> Dict:
        """Predict churn using logistic-style scoring."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            features = ["last_purchase_days_ago", "avg_order_value", "purchase_frequency",
                        "total_orders", "nps_score"]
            X = df[features].fillna(0)
            y = df["is_churned"].astype(int)

            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, y)

            importances = dict(zip(features, clf.feature_importances_.round(3)))
            predicted_churn_rate = clf.predict_proba(X)[:, 1].mean() * 100

            return {
                "predicted_churn_rate_pct": round(predicted_churn_rate, 1),
                "feature_importances": importances,
                "model": "RandomForest",
            }
        except Exception:
            return {
                "predicted_churn_rate_pct": round(df["churn_risk"].mean() * 100, 1),
                "feature_importances": {},
                "model": "Statistical",
            }

    def analyze(self) -> Dict:
        sales_df = get_sales_data()
        customer_df = get_customer_data()

        # Revenue forecast
        rev_forecast = self._forecast_revenue(sales_df, days_ahead=30)

        # Churn forecast
        churn_forecast = self._forecast_churn(customer_df)

        # Customer growth projection
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        monthly_revenue = sales_df.groupby(sales_df["date"].dt.to_period("M"))["revenue"].sum()
        revenue_trend = monthly_revenue.pct_change().mean() * 100

        # Demand by product (next 30 days)
        product_demand = sales_df.groupby("product_name")["units_sold"].mean() * 30
        product_demand = product_demand.round(0).astype(int).sort_values(ascending=False).to_dict()

        stats_summary = f"""
Forecasting Analysis:

Revenue Forecast (Next 30 Days):
- Projected Revenue: ${rev_forecast['monthly_forecast_total']:,.0f}
- Last Month Actual: ${rev_forecast['last_month_actual']:,.0f}
- Projected Growth: {rev_forecast['projected_growth_pct']:+.1f}%

Churn Prediction:
- Predicted Churn Rate: {churn_forecast['predicted_churn_rate_pct']:.1f}%
- Key Churn Drivers: {json.dumps(churn_forecast['feature_importances'])}

Revenue Trend (Monthly avg): {revenue_trend:+.1f}%

Expected Monthly Demand (Units) by Product:
{json.dumps(product_demand)}

Historical Revenue by Month:
{json.dumps({str(k): round(float(v), 2) for k, v in monthly_revenue.tail(6).items()})}
"""

        result = self.think(
            prompt="Based on these forecasts, what are the most important predictions the CEO should act on? What risks do the forecasts reveal?",
            context=stats_summary
        )

        result["computed_metrics"] = {
            "revenue_forecast": rev_forecast,
            "churn_prediction": churn_forecast,
            "avg_monthly_revenue_growth_pct": round(float(revenue_trend), 1),
            "product_demand_forecast": product_demand,
            "monthly_history": {str(k): round(float(v), 2) for k, v in monthly_revenue.items()},
        }

        return result
