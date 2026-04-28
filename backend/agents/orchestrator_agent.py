"""
Orchestrator Agent — Master controller that coordinates all BizMind agents.
Uses Groq API.
"""
import json
import re
import time
import traceback
from typing import Dict, Optional
from openai import OpenAI
from core.config import config

from agents.sales_agent import SalesAgent
from agents.customer_agent import CustomerAgent
from agents.inventory_agent import InventoryAgent
from agents.marketing_agent import MarketingAgent
from agents.forecasting_agent import ForecastingAgent
from agents.strategy_agent import StrategyAgent
from agents.report_agent import ReportAgent
from agents.action_agent import ActionAgent


class OrchestratorAgent:
    def __init__(self):
        self.name = "Orchestrator"
        try:
            self.client = OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            print(f"  ✓ Orchestrator client ready")
        except Exception as e:
            print(f"  ✗ Orchestrator client FAILED: {e}")
            self.client = None

        self._agents = {
            "sales": SalesAgent(),
            "customer": CustomerAgent(),
            "inventory": InventoryAgent(),
            "marketing": MarketingAgent(),
            "forecasting": ForecastingAgent(),
        }
        self._strategy_agent = StrategyAgent()
        self._report_agent = ReportAgent()
        self._action_agent = ActionAgent()
        self._last_results: Optional[Dict] = None

    def _clean_json(self, raw: str) -> str:
        cleaned = raw.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        if not cleaned.startswith('{'):
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
        return cleaned

    def run_full_analysis(self, verbose: bool = False) -> Dict:
        pipeline_start = time.time()
        results = {}
        timings = {}

        # Phase 1: Specialized agents
        print("🔍 [Orchestrator] Running specialized agents...")
        for agent_name, agent in self._agents.items():
            print(f"  ▶ Running {agent.name}...")
            t_start = time.time()
            results[agent_name] = agent.analyze()
            timings[agent_name] = round(time.time() - t_start, 2)
            status = results[agent_name].get("status", "unknown")
            print(f"  ✅ {agent.name} complete ({timings[agent_name]}s) — Status: {status}")

        # Phase 2: Strategy
        print("🧠 [Orchestrator] Synthesizing strategy...")
        t_start = time.time()
        strategy_result = self._strategy_agent.synthesize(results)
        timings["strategy"] = round(time.time() - t_start, 2)
        health_score = strategy_result.get("business_health_score", 0)
        print(f"  ✅ Strategy complete ({timings['strategy']}s) — Health: {health_score}/100")

        # Phase 3: Report
        print("📄 [Orchestrator] Generating executive report...")
        t_start = time.time()
        report = self._report_agent.generate({**results, "strategy": strategy_result})
        timings["report"] = round(time.time() - t_start, 2)
        print(f"  ✅ Report complete ({timings['report']}s)")

        # Phase 4: Actions
        print("⚙️ [Orchestrator] Generating action plan...")
        t_start = time.time()
        actions = self._action_agent.generate_actions(strategy_result, results)
        timings["actions"] = round(time.time() - t_start, 2)
        print(f"  ✅ Actions complete ({timings['actions']}s) — {actions.get('total_actions', 0)} actions")

        total_time = round(time.time() - pipeline_start, 2)
        print(f"\n🎯 [Orchestrator] Full analysis complete in {total_time}s")

        final = {
            "pipeline_version": "1.0",
            "total_time_seconds": total_time,
            "timings": timings,
            "agent_results": results,
            "strategy": strategy_result,
            "report": report,
            "actions": actions,
            "business_health_score": health_score,
            "health_label": strategy_result.get("health_label", "Unknown"),
        }

        self._last_results = final
        return final

    def answer_question(self, question: str) -> Dict:
        if not self._last_results:
            return {"error": "No analysis data. Run full analysis first.", "answer": "Please run analysis first.", "confidence": "low"}

        if self.client is None:
            return {"error": "Client not initialized", "answer": "Groq client error", "confidence": "low"}

        context_parts = []
        strategy = self._last_results.get("strategy", {})
        context_parts.append(f"Health: {strategy.get('health_label')} ({strategy.get('business_health_score')}/100)")
        context_parts.append(f"Summary: {strategy.get('executive_summary', '')}")

        for agent_name, result in self._last_results.get("agent_results", {}).items():
            if result and isinstance(result, dict):
                context_parts.append(f"\n{agent_name.upper()}: {result.get('summary', '')}")
                metrics = result.get("computed_metrics", result.get("key_metrics", {}))
                if metrics:
                    context_parts.append(f"Metrics: {dict(list(metrics.items())[:4])}")

        full_context = "\n".join(context_parts)

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=800,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": """You are BizMind Orchestrator AI. Answer business questions using data.
Respond ONLY with JSON (no markdown):
{
  "answer": "Direct specific answer",
  "supporting_data": ["data point 1", "data point 2"],
  "recommendation": "What to do",
  "confidence": "high"
}"""},
                    {"role": "user", "content": f"Data:\n{full_context}\n\nQuestion: {question}"}
                ]
            )
            raw = response.choices[0].message.content
            cleaned = self._clean_json(raw)
            return json.loads(cleaned)

        except Exception as e:
            print(f"❌ Orchestrator question error: {e}")
            return {"answer": f"Error: {str(e)}", "supporting_data": [], "recommendation": "Check logs", "confidence": "low"}

    def simulate_scenario(self, scenario: str) -> Dict:
        if not self._last_results:
            return {"error": "No analysis data. Run full analysis first."}

        if self.client is None:
            return {"error": "Client not initialized"}

        strategy = self._last_results.get("strategy", {})

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": """You are a business scenario simulator.
Respond ONLY with JSON (no markdown):
{
  "scenario": "scenario description",
  "estimated_impact": {
    "revenue_change_pct": 10,
    "profit_change_pct": 8,
    "churn_change_pct": -5,
    "customer_satisfaction_change": "positive"
  },
  "risks": ["risk 1", "risk 2"],
  "opportunities": ["opportunity 1"],
  "recommendation": "Clear yes/no recommendation with reasoning",
  "confidence": "medium"
}"""},
                    {"role": "user", "content": f"""Business health: {strategy.get('health_label')} ({strategy.get('business_health_score')}/100)
Summary: {strategy.get('executive_summary', '')}
Priorities: {json.dumps(strategy.get('strategic_priorities', [])[:3])}

Simulate: {scenario}"""}
                ]
            )
            raw = response.choices[0].message.content
            cleaned = self._clean_json(raw)
            return json.loads(cleaned)

        except Exception as e:
            print(f"❌ Orchestrator simulate error: {e}")
            return {"error": str(e), "scenario": scenario}