"""
Strategy Agent — Synthesizes all agent insights into master business strategy.
Uses Groq API.
"""
import json
import re
import traceback
from typing import Dict
from openai import OpenAI
from core.config import config


class StrategyAgent:
    def __init__(self):
        self.name = "StrategyBot"
        self.role = "Chief Strategy Officer AI"
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
        return """You are StrategyBot, the Chief Strategy Officer AI in BizMind.
You receive reports from specialized agents and synthesize them into a unified business strategy.

Your job:
1. Find ROOT CAUSES — not just symptoms
2. Identify cross-domain patterns across sales, customers, inventory, and marketing
3. Detect CONFLICTS between agent recommendations and resolve them
4. Produce a prioritized CEO action plan

Respond ONLY with this exact JSON structure (no markdown, no code fences):
{
  "agent": "StrategyBot",
  "business_health_score": 75,
  "health_label": "Good",
  "root_causes": [
    {"issue": "describe issue", "evidence": "data supporting this", "severity": "high"}
  ],
  "cross_domain_insights": ["insight connecting multiple business areas"],
  "agent_conflicts": [
    {"conflict": "describe conflict", "agents_involved": ["Sales", "Marketing"], "resolution": "how to resolve"}
  ],
  "strategic_priorities": [
    {"priority": 1, "action": "specific action", "rationale": "why this", "timeline": "this week", "expected_outcome": "result"}
  ],
  "quick_wins": ["action doable in under 1 week"],
  "risks_to_watch": ["risk description"],
  "executive_summary": "2-3 sentence board-level summary of business situation and direction"
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

    def synthesize(self, agent_results: Dict[str, Dict]) -> Dict:
        if self.client is None:
            return self._error_result("Groq client not initialized")

        briefings = []
        for agent_name, result in agent_results.items():
            if not result:
                continue
            summary = result.get("summary", "No summary")
            status = result.get("status", "unknown")
            insights = result.get("insights", [])
            risks = result.get("risks", [])
            recs = result.get("recommendations", [])

            briefing = f"""
=== {agent_name.upper()} (Status: {status.upper()}) ===
Summary: {summary}
Insights: {json.dumps(insights[:3])}
Risks: {json.dumps(risks[:2])}
Recommendations: {json.dumps([r.get("action", str(r)) if isinstance(r, dict) else str(r) for r in recs[:2]])}"""
            briefings.append(briefing)

        full_briefing = "\n".join(briefings)

        try:
            print(f"  📡 {self.name} → Groq API...")
            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Agent reports:\n{full_briefing}\n\nSynthesize into master CEO strategy."}
                ]
            )
            raw = response.choices[0].message.content
            print(f"  📨 {self.name} ← {len(raw)} chars received")

            cleaned = self._clean_json(raw)
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            print(f"❌ StrategyBot JSON error: {e}\nRaw: {raw[:500] if 'raw' in dir() else 'none'}")
            return self._error_result(f"JSON parse failed: {e}")
        except Exception as e:
            print(f"❌ StrategyBot API error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "agent": "StrategyBot",
            "business_health_score": 50,
            "health_label": "Unknown",
            "root_causes": [],
            "cross_domain_insights": [f"Strategy error: {error_msg}"],
            "agent_conflicts": [],
            "strategic_priorities": [],
            "quick_wins": [],
            "risks_to_watch": [],
            "executive_summary": f"Strategy synthesis error: {error_msg}",
        }