#!/usr/bin/env python3
"""
BizMind CLI — Run agents and test the system from the terminal.

Usage:
    python run.py full          → Full pipeline (all agents)
    python run.py sales         → Sales agent only
    python run.py customer      → Customer agent only
    python run.py inventory     → Inventory agent only
    python run.py marketing     → Marketing agent only
    python run.py forecast      → Forecasting agent only
    python run.py ask "why did profit drop?"  → Ask a question
    python run.py simulate "what if we cut prices 10%?"
"""
import sys
import json
import time
from pathlib import Path

# Make sure we can import backend modules
sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    OrchestratorAgent,
    SalesAgent,
    CustomerAgent,
    InventoryAgent,
    MarketingAgent,
    ForecastingAgent,
)


def print_json(data: dict, indent: int = 2):
    print(json.dumps(data, indent=indent, default=str))


def print_header(title: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def run_agent(name: str, agent):
    print_header(f"{agent.name} — {agent.role}")
    t = time.time()
    result = agent.analyze()
    elapsed = round(time.time() - t, 2)

    print(f"Status  : {result.get('status', '—')}")
    print(f"Summary : {result.get('summary', '—')}")
    print(f"Time    : {elapsed}s\n")

    print("Key Insights:")
    for insight in result.get("insights", [])[:3]:
        print(f"  💡 {insight}")

    print("\nRisks:")
    for risk in result.get("risks", [])[:2]:
        print(f"  ⚠  {risk}")

    print("\nTop Recommendations:")
    for rec in result.get("recommendations", [])[:2]:
        action = rec.get("action", rec) if isinstance(rec, dict) else rec
        priority = rec.get("priority", "") if isinstance(rec, dict) else ""
        print(f"  ⚡ [{priority.upper()}] {action}")

    return result


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    AGENTS = {
        "sales": SalesAgent,
        "customer": CustomerAgent,
        "inventory": InventoryAgent,
        "marketing": MarketingAgent,
        "forecast": ForecastingAgent,
    }

    if command == "full":
        print_header("BizMind — Full Multi-Agent Analysis")
        orchestrator = OrchestratorAgent()
        result = orchestrator.run_full_analysis(verbose=True)

        print_header("FINAL RESULTS")
        strategy = result.get("strategy", {})
        print(f"Business Health  : {result.get('business_health_score')}/100 ({result.get('health_label')})")
        print(f"Analysis Time    : {result.get('total_time_seconds')}s")
        print(f"\nExecutive Summary:\n  {strategy.get('executive_summary', '—')}")

        print("\nRoot Causes:")
        for rc in strategy.get("root_causes", [])[:3]:
            print(f"  [{rc['severity'].upper()}] {rc['issue']}")

        print("\nTop Strategic Priorities:")
        for p in strategy.get("strategic_priorities", [])[:3]:
            print(f"  {p['priority']}. {p['action']} → {p['expected_outcome']}")

        print("\nQuick Wins:")
        for win in strategy.get("quick_wins", [])[:3]:
            print(f"  ✓ {win}")

        print("\nAction Plan:")
        actions = result.get("actions", {}).get("actions", [])
        for a in actions[:5]:
            print(f"  [{a.get('priority','?').upper()}] {a.get('action','?')}")
            print(f"        Impact: {a.get('expected_impact','?')} | Timeline: {a.get('timeline','?')}")

    elif command in AGENTS:
        agent = AGENTS[command]()
        result = run_agent(command, agent)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python run.py ask \"your question here\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print_header(f"Ask AI: {question}")
        print("⚠ Note: 'ask' requires a full analysis to be run first via the API.")
        print("Start the FastAPI server and use: POST /api/v1/question")

    elif command == "simulate":
        if len(sys.argv) < 3:
            print("Usage: python run.py simulate \"scenario description\"")
            sys.exit(1)
        scenario = " ".join(sys.argv[2:])
        print_header(f"Simulate: {scenario}")
        print("⚠ Note: Simulation requires a full analysis to be run first via the API.")
        print("Start the FastAPI server and use: POST /api/v1/simulate")

    elif command == "data":
        print_header("Data Preview")
        from data import get_sales_data, get_customer_data, get_inventory_data, get_marketing_data
        print("Sales data:")
        df = get_sales_data()
        print(df.head(3).to_string())
        print(f"\nShape: {df.shape}")
        print(f"\nCustomers: {len(get_customer_data())} records")
        print(f"Inventory: {len(get_inventory_data())} products")
        print(f"Marketing: {len(get_marketing_data())} daily campaign records")

    else:
        print("""
╔══════════════════════════════════════════╗
║         BizMind CLI — Commands           ║
╚══════════════════════════════════════════╝

  python run.py full           Run complete multi-agent pipeline
  python run.py sales          Sales Agent only
  python run.py customer       Customer Agent only
  python run.py inventory      Inventory Agent only
  python run.py marketing      Marketing Agent only
  python run.py forecast       Forecasting Agent only
  python run.py data           Preview data samples
  python run.py ask "..."      Ask a question (via API)
  python run.py simulate "..." Simulate scenario (via API)

  Start API server:
    cd backend && uvicorn main:app --reload --port 8000
""")


if __name__ == "__main__":
    main()
