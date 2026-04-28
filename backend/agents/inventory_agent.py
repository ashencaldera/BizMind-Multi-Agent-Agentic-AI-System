"""
Inventory Agent — Tracks stock levels, predicts demand, prevents overstock/understock.
"""
import json
from typing import Dict
import pandas as pd
from agents.base_agent import BaseAgent
from data import get_inventory_data, get_sales_data


class InventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="StockBot",
            role="Inventory & Supply Chain Specialist",
            expertise="Stock management, demand forecasting, reorder optimization, carrying cost reduction, stockout prevention"
        )

    def analyze(self) -> Dict:
        inv = get_inventory_data()
        sales = get_sales_data()

        total_stock_value = inv["stock_value"].sum()
        critical_items = inv[inv["status"] == "Critical"]
        low_items = inv[inv["status"] == "Low"]
        overstock_items = inv[inv["status"] == "Overstock"]
        optimal_items = inv[inv["status"] == "Optimal"]

        # Days of stock for critical items
        critical_details = critical_items[
            ["product_name", "current_stock", "reorder_point", "days_of_stock_remaining", "daily_sales_avg"]
        ].to_dict("records")

        # Overstock details
        overstock_details = overstock_items[
            ["product_name", "current_stock", "max_capacity", "stock_value", "turnover_rate"]
        ].to_dict("records")

        # Turnover by category
        inv_with_cat = inv.copy()
        cat_turnover = inv_with_cat.groupby("category")["turnover_rate"].mean().round(2).to_dict()

        # Total carrying costs (estimate: 20% of stock value annually)
        annual_carrying_cost = total_stock_value * 0.20
        monthly_carrying_cost = annual_carrying_cost / 12

        # Stockout risk score
        stockout_risk_count = len(inv[inv["days_of_stock_remaining"] < 7])

        stats_summary = f"""
Inventory Analysis:
- Total Stock Value: ${total_stock_value:,.0f}
- Monthly Carrying Cost: ${monthly_carrying_cost:,.0f}
- Status Breakdown:
  * Critical (need reorder NOW): {len(critical_items)} products
  * Low Stock: {len(low_items)} products
  * Overstock: {len(overstock_items)} products
  * Optimal: {len(optimal_items)} products
- Products with <7 days stock: {stockout_risk_count}

Critical Items (URGENT):
{json.dumps(critical_details, indent=2)}

Overstock Items (Cash Tied Up):
{json.dumps(overstock_details, indent=2)}

Turnover Rate by Category:
{json.dumps(cat_turnover)}
"""

        result = self.think(
            prompt="Analyze inventory health. Which products need immediate action? How can we reduce carrying costs while preventing stockouts?",
            context=stats_summary
        )

        result["computed_metrics"] = {
            "total_stock_value": round(total_stock_value, 2),
            "monthly_carrying_cost": round(monthly_carrying_cost, 2),
            "critical_count": int(len(critical_items)),
            "low_count": int(len(low_items)),
            "overstock_count": int(len(overstock_items)),
            "optimal_count": int(len(optimal_items)),
            "stockout_risk_count": int(stockout_risk_count),
            "status_breakdown": inv[["product_name", "status", "current_stock",
                                      "days_of_stock_remaining", "stock_value"]].to_dict("records"),
            "category_turnover": cat_turnover,
        }

        return result
