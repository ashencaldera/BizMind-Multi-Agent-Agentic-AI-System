"""
BizMind API Routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import time

from api.schemas import (
    QuestionRequest, ScenarioRequest, AgentRunRequest,
    AnalysisResponse, QuestionResponse, HealthResponse
)
from agents import (
    OrchestratorAgent, SalesAgent, CustomerAgent,
    InventoryAgent, MarketingAgent, ForecastingAgent
)
from core.config import config

router = APIRouter()

# Singleton orchestrator (shared state)
_orchestrator = OrchestratorAgent()
_individual_agents = {
    "sales": SalesAgent(),
    "customer": CustomerAgent(),
    "inventory": InventoryAgent(),
    "marketing": MarketingAgent(),
    "forecasting": ForecastingAgent(),
}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "operational",
        "version": config.VERSION,
        "agents_available": list(_individual_agents.keys()) + ["strategy", "report", "action", "orchestrator"],
    }


@router.post("/analyze/full")
async def run_full_analysis():
    """Run the complete multi-agent analysis pipeline."""
    try:
        result = _orchestrator.run_full_analysis()
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/agent/{agent_name}")
async def run_single_agent(agent_name: str):
    """Run a single agent analysis."""
    if agent_name not in _individual_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. Available: {list(_individual_agents.keys())}")
    try:
        result = _individual_agents[agent_name].analyze()
        return {"success": True, "agent": agent_name, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question")
async def ask_question(req: QuestionRequest):
    """Ask the orchestrator a natural language business question."""
    try:
        answer = _orchestrator.answer_question(req.question)
        return {"success": True, "question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate")
async def simulate_scenario(req: ScenarioRequest):
    """Simulate a what-if business scenario."""
    try:
        result = _orchestrator.simulate_scenario(req.scenario)
        return {"success": True, "scenario": req.scenario, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/sales")
async def get_sales_data():
    """Get raw sales data for dashboard charts."""
    from data import get_sales_data
    import pandas as pd
    df = get_sales_data()
    df["date"] = pd.to_datetime(df["date"])

    # Daily revenue summary
    daily = df.groupby("date").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        units=("units_sold", "sum")
    ).reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    # Product revenue
    by_product = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False).reset_index()

    # Regional revenue
    by_region = df.groupby("region")["revenue"].sum().reset_index()

    return {
        "daily_revenue": daily.to_dict("records"),
        "by_product": by_product.to_dict("records"),
        "by_region": by_region.to_dict("records"),
    }


@router.get("/data/customers")
async def get_customer_data():
    """Get customer data for dashboard."""
    from data import get_customer_data
    df = get_customer_data()

    segment_stats = df.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_ltv=("lifetime_value", "mean"),
        avg_churn_risk=("churn_risk", "mean"),
    ).round(2).reset_index()

    churn_dist = df.groupby("is_churned")["customer_id"].count().reset_index()

    return {
        "total": len(df),
        "segment_stats": segment_stats.to_dict("records"),
        "churn_distribution": churn_dist.to_dict("records"),
        "avg_ltv": round(df["lifetime_value"].mean(), 2),
        "avg_nps": round(df["nps_score"].mean(), 2),
    }


@router.get("/data/inventory")
async def get_inventory_data():
    """Get inventory data for dashboard."""
    from data import get_inventory_data
    df = get_inventory_data()
    return {
        "items": df.to_dict("records"),
        "total_value": round(df["stock_value"].sum(), 2),
        "status_counts": df["status"].value_counts().to_dict(),
    }


@router.get("/data/marketing")
async def get_marketing_data():
    """Get marketing data for dashboard."""
    from data import get_marketing_data
    import pandas as pd
    df = get_marketing_data()

    by_channel = df.groupby("channel").agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
        avg_roi=("roi", "mean"),
    ).round(2).reset_index()

    by_campaign = df.groupby("campaign").agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        avg_roi=("roi", "mean"),
    ).round(2).reset_index()

    return {
        "by_channel": by_channel.to_dict("records"),
        "by_campaign": by_campaign.to_dict("records"),
        "total_spend": round(df["spend"].sum(), 2),
        "total_revenue": round(df["revenue"].sum(), 2),
        "overall_roi": round((df["revenue"].sum() - df["spend"].sum()) / df["spend"].sum() * 100, 1),
    }
