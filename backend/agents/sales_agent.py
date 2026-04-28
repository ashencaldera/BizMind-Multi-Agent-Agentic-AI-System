"""
Sales Agent — Analyzes sales trends, revenue, top products, and seasonality.
"""
import json
from typing import Dict
import pandas as pd
from agents.base_agent import BaseAgent
from data import get_sales_data


class SalesAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SalesBot",
            role="Sales Analytics Specialist",
            expertise="Revenue analysis, sales trends, product performance, regional analysis, seasonality detection"
        )

    def analyze(self) -> Dict:
        df = get_sales_data()

        # Compute key sales statistics
        total_revenue = df["revenue"].sum()
        total_profit = df["profit"].sum()
        total_units = df["units_sold"].sum()
        avg_margin = (total_profit / total_revenue * 100)

        # Last 30 vs previous 30 days
        df["date"] = pd.to_datetime(df["date"])
        recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
        prev = df[(df["date"] < df["date"].max() - pd.Timedelta(days=30)) &
                  (df["date"] >= df["date"].max() - pd.Timedelta(days=60))]

        recent_rev = recent["revenue"].sum()
        prev_rev = prev["revenue"].sum()
        mom_growth = ((recent_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0

        # Top products
        top_products = (df.groupby("product_name")["revenue"]
                        .sum().sort_values(ascending=False).head(3).to_dict())

        # Bottom products
        bottom_products = (df.groupby("product_name")["revenue"]
                           .sum().sort_values(ascending=True).head(3).to_dict())

        # Regional breakdown
        regional = df.groupby("region")["revenue"].sum().to_dict()

        # Category performance
        category = df.groupby("category")["profit"].sum().sort_values(ascending=False).to_dict()

        # Weekly trend (last 8 weeks)
        df["week"] = df["date"].dt.isocalendar().week
        weekly = df.groupby("week")["revenue"].sum().tail(8).to_dict()

        stats_summary = f"""
Sales Analysis Data (Last 180 Days):
- Total Revenue: ${total_revenue:,.0f}
- Total Profit: ${total_profit:,.0f}
- Profit Margin: {avg_margin:.1f}%
- Units Sold: {total_units:,}
- Month-over-Month Growth: {mom_growth:.1f}%
- Top Products by Revenue: {json.dumps({k: f"${v:,.0f}" for k,v in top_products.items()})}
- Underperforming Products: {json.dumps({k: f"${v:,.0f}" for k,v in bottom_products.items()})}
- Regional Revenue: {json.dumps({k: f"${v:,.0f}" for k,v in regional.items()})}
- Category Profitability: {json.dumps({k: f"${v:,.0f}" for k,v in category.items()})}
- Recent Month Revenue: ${recent_rev:,.0f}
- Previous Month Revenue: ${prev_rev:,.0f}
"""

        result = self.think(
            prompt="Analyze these sales metrics. Identify trends, risks, and strategic opportunities. What actions should the CEO take?",
            context=stats_summary
        )

        # Attach computed metrics directly
        result["computed_metrics"] = {
            "total_revenue": round(total_revenue, 2),
            "total_profit": round(total_profit, 2),
            "profit_margin_pct": round(avg_margin, 1),
            "total_units": int(total_units),
            "mom_growth_pct": round(mom_growth, 1),
            "top_products": top_products,
            "regional_revenue": regional,
            "category_profits": category,
            "weekly_revenue": {str(k): round(v, 2) for k, v in weekly.items()},
        }

        return result
