"""
Action Agent — Converts insights into concrete executable business decisions.
Uses Groq API.
"""
import json
import re
import traceback
from typing import Dict
from openai import OpenAI
from core.config import config


class ActionAgent:
    def __init__(self):
        self.name = "ActionBot"
        try:
            self.client = OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            print(f"  ✓ {self.name} client ready")
        except Exception as e:
            print(f"  ✗ {self.name} client FAILED: {e}")
            self.client = None

    @property
    def system_prompt(self) -> str:
        return """You are ActionBot, the execution AI in BizMind. Convert strategic insights into concrete, specific, measurable business actions.

Respond ONLY with this exact JSON structure (no markdown, no code fences):
{
  "agent": "ActionBot",
  "total_actions": 8,
  "actions": [
    {
      "id": "ACT-001",
      "category": "Pricing",
      "action": "Specific action sentence",
      "details": "How exactly to execute this",
      "expected_impact": "Quantified expected result e.g. +$5k revenue",
      "effort": "Low",
      "priority": "High",
      "timeline": "This Week",
      "kpi": "Metric to track success",
      "owner": "CEO"
    }
  ],
  "simulation_results": {
    "if_all_actions_executed": {
      "revenue_impact_pct": 15,
      "churn_reduction_pct": 8,
      "margin_improvement_pct": 5,
      "estimated_monthly_gain": "$25,000"
    }
  }
}"""

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

    def generate_actions(self, strategy_result: Dict, agent_results: Dict) -> Dict:
        if self.client is None:
            return self._error_result("Groq client not initialized")

        all_recommendations = []
        for agent_name, result in agent_results.items():
            if result and isinstance(result, dict):
                for rec in result.get("recommendations", [])[:2]:
                    if isinstance(rec, dict):
                        all_recommendations.append(f"[{agent_name}] {rec.get('action', str(rec))}")
                    else:
                        all_recommendations.append(f"[{agent_name}] {rec}")

        context = f"""
Business Health: {strategy_result.get('business_health_score', 'N/A')}/100 — {strategy_result.get('health_label', 'Unknown')}

Strategic Priorities:
{json.dumps(strategy_result.get('strategic_priorities', [])[:4], indent=2)}

Quick Wins Available:
{json.dumps(strategy_result.get('quick_wins', [])[:4])}

Agent Recommendations:
{json.dumps(all_recommendations, indent=2)}

Executive Summary: {strategy_result.get('executive_summary', 'N/A')}
"""

        try:
            print(f"  📡 {self.name} → Groq API...")
            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate 8 executable business actions from:\n{context}"}
                ]
            )
            raw = response.choices[0].message.content
            print(f"  📨 {self.name} ← {len(raw)} chars received")

            cleaned = self._clean_json(raw)
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            print(f"❌ ActionBot JSON error: {e}")
            return self._error_result(f"JSON parse failed: {e}")
        except Exception as e:
            print(f"❌ ActionBot API error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "agent": "ActionBot",
            "total_actions": 0,
            "actions": [],
            "simulation_results": {},
            "error": error_msg,
        }