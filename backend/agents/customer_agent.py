"""
Customer Agent — Analyzes customer segments, churn risk, LTV, and behavior.
"""
import json
from typing import Dict
import pandas as pd
from agents.base_agent import BaseAgent
from data import get_customer_data


class CustomerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CustBot",
            role="Customer Intelligence Specialist",
            expertise="Customer segmentation, churn prediction, lifetime value analysis, retention strategies, behavioral patterns"
        )

    def analyze(self) -> Dict:
        df = get_customer_data()

        total_customers = len(df)
        churned = df["is_churned"].sum()
        churn_rate = churned / total_customers * 100
        avg_ltv = df["lifetime_value"].mean()
        avg_nps = df["nps_score"].mean()

        # Segment breakdown
        segment_stats = df.groupby("segment").agg(
            count=("customer_id", "count"),
            avg_ltv=("lifetime_value", "mean"),
            avg_churn_risk=("churn_risk", "mean"),
            avg_order_value=("avg_order_value", "mean"),
        ).round(2).to_dict("index")

        # High churn risk (>0.5)
        high_risk = df[df["churn_risk"] > 0.5]
        high_risk_count = len(high_risk)
        high_risk_ltv_at_risk = high_risk["lifetime_value"].sum()

        # Top customers by LTV
        top_customers = df.nlargest(5, "lifetime_value")[
            ["customer_id", "segment", "lifetime_value", "churn_risk"]
        ].to_dict("records")

        # Preferred channels
        channel_dist = df["preferred_channel"].value_counts().to_dict()

        # NPS distribution
        promoters = len(df[df["nps_score"] >= 9])
        passives = len(df[(df["nps_score"] >= 7) & (df["nps_score"] < 9)])
        detractors = len(df[df["nps_score"] <= 6])
        nps = round((promoters - detractors) / total_customers * 100, 1)

        stats_summary = f"""
Customer Intelligence Data ({total_customers} customers):
- Total Customers: {total_customers}
- Churned Customers: {churned} ({churn_rate:.1f}%)
- High Churn Risk (>50%): {high_risk_count} customers
- LTV at Risk from Churn: ${high_risk_ltv_at_risk:,.0f}
- Average Lifetime Value: ${avg_ltv:,.0f}
- Net Promoter Score: {nps}
- Average NPS Score: {avg_nps:.1f}/10
- Promoters: {promoters} | Passives: {passives} | Detractors: {detractors}

Segment Performance:
{json.dumps(segment_stats, indent=2)}

Top Revenue Customers:
{json.dumps(top_customers, indent=2)}

Preferred Acquisition Channels:
{json.dumps(channel_dist)}
"""

        result = self.think(
            prompt="Analyze customer health. Who is at risk? What retention strategies should we deploy? Which segments deserve more investment?",
            context=stats_summary
        )

        result["computed_metrics"] = {
            "total_customers": total_customers,
            "churn_rate_pct": round(churn_rate, 1),
            "high_risk_customers": int(high_risk_count),
            "ltv_at_risk": round(high_risk_ltv_at_risk, 2),
            "avg_lifetime_value": round(avg_ltv, 2),
            "net_promoter_score": nps,
            "segment_breakdown": segment_stats,
            "channel_distribution": channel_dist,
        }

        return result
