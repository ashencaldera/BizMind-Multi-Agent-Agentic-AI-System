"""
Marketing Agent — Evaluates campaign ROI, channel performance, and ad spend efficiency.
"""
import json
from typing import Dict
import pandas as pd
from agents.base_agent import BaseAgent
from data import get_marketing_data


class MarketingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MarketBot",
            role="Marketing Analytics Specialist",
            expertise="Campaign performance, ROI analysis, channel attribution, customer acquisition cost, conversion optimization"
        )

    def analyze(self) -> Dict:
        df = get_marketing_data()

        total_spend = df["spend"].sum()
        total_revenue = df["revenue"].sum()
        total_conversions = df["conversions"].sum()
        overall_roi = (total_revenue - total_spend) / total_spend * 100
        overall_cpa = total_spend / max(1, total_conversions)
        avg_ctr = df["ctr"].mean()

        # By campaign
        by_campaign = df.groupby("campaign").agg(
            total_spend=("spend", "sum"),
            total_revenue=("revenue", "sum"),
            total_conversions=("conversions", "sum"),
            avg_roi=("roi", "mean"),
            avg_ctr=("ctr", "mean"),
            avg_cpa=("cpa", "mean"),
        ).round(2).sort_values("avg_roi", ascending=False).to_dict("index")

        # By channel
        by_channel = df.groupby("channel").agg(
            total_spend=("spend", "sum"),
            total_revenue=("revenue", "sum"),
            avg_roi=("roi", "mean"),
            total_conversions=("conversions", "sum"),
        ).round(2).sort_values("avg_roi", ascending=False).to_dict("index")

        # Best vs worst campaigns
        campaign_roi = df.groupby("campaign")["roi"].mean()
        best_campaign = campaign_roi.idxmax()
        worst_campaign = campaign_roi.idxmin()

        # Weekly trend
        df["date"] = pd.to_datetime(df["date"])
        df["week"] = df["date"].dt.isocalendar().week
        weekly_roi = df.groupby("week")["roi"].mean().round(1).tail(8).to_dict()

        stats_summary = f"""
Marketing Performance Data (Last 90 Days):
- Total Ad Spend: ${total_spend:,.0f}
- Total Revenue Generated: ${total_revenue:,.0f}
- Overall ROI: {overall_roi:.1f}%
- Total Conversions: {total_conversions:,}
- Average Cost Per Acquisition: ${overall_cpa:.2f}
- Average Click-Through Rate: {avg_ctr:.2f}%

Campaign Performance (sorted by ROI):
{json.dumps(by_campaign, indent=2)}

Channel Performance:
{json.dumps(by_channel, indent=2)}

Best Performing Campaign: {best_campaign} (ROI: {campaign_roi[best_campaign]:.1f}%)
Worst Performing Campaign: {worst_campaign} (ROI: {campaign_roi[worst_campaign]:.1f}%)

Weekly ROI Trend:
{json.dumps({str(k): v for k,v in weekly_roi.items()})}
"""

        result = self.think(
            prompt="Analyze marketing performance. Where should we cut spend? Where should we double down? What channels deliver the best ROI? How can we reduce CPA?",
            context=stats_summary
        )

        result["computed_metrics"] = {
            "total_spend": round(total_spend, 2),
            "total_revenue": round(total_revenue, 2),
            "overall_roi_pct": round(overall_roi, 1),
            "total_conversions": int(total_conversions),
            "avg_cpa": round(overall_cpa, 2),
            "avg_ctr_pct": round(avg_ctr, 2),
            "best_campaign": best_campaign,
            "worst_campaign": worst_campaign,
            "by_channel": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_channel.items()},
            "by_campaign": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_campaign.items()},
        }

        return result
