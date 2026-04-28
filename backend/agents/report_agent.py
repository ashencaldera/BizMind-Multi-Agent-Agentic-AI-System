"""
Report Agent — Generates CEO-level weekly intelligence reports.
Uses Groq API.
"""
import json
import re
import traceback
from typing import Dict
from datetime import datetime
from openai import OpenAI
from core.config import config


class ReportAgent:
    def __init__(self):
        self.name = "ReportBot"
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
        return """You are ReportBot, an executive reporting AI. Write concise CEO/board-level reports.

Respond ONLY with this exact JSON structure (no markdown, no code fences):
{
  "report_title": "BizMind Weekly Intelligence Report",
  "generated_at": "now",
  "period": "Last 7 days",
  "executive_summary": "3-4 sentence business overview",
  "sections": [
    {
      "title": "Section Title",
      "status": "green",
      "headline": "One line summary",
      "body": "2-3 sentences of insight",
      "metric": {"label": "Key Metric", "value": "$X", "change": "+Y%"}
    }
  ],
  "action_items": [
    {"owner": "CEO", "action": "specific action", "deadline": "this week", "priority": "high"}
  ],
  "bottom_line": "The single most important thing to do this week"
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

    def generate(self, all_results: Dict) -> Dict:
        if self.client is None:
            return self._error_result("Groq client not initialized")

        summaries = {}
        for agent_name, result in all_results.items():
            if result and isinstance(result, dict):
                summaries[agent_name] = {
                    "status": result.get("status", "unknown"),
                    "summary": result.get("summary", ""),
                    "insights": result.get("insights", [])[:2],
                    "recommendations": result.get("recommendations", [])[:2],
                }

        try:
            print(f"  📡 {self.name} → Groq API...")
            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate weekly report from:\n{json.dumps(summaries, indent=2)}"}
                ]
            )
            raw = response.choices[0].message.content
            print(f"  📨 {self.name} ← {len(raw)} chars received")

            cleaned = self._clean_json(raw)
            result = json.loads(cleaned)
            result["generated_at"] = datetime.now().isoformat()
            return result

        except json.JSONDecodeError as e:
            print(f"❌ ReportBot JSON error: {e}")
            return self._error_result(f"JSON parse failed: {e}")
        except Exception as e:
            print(f"❌ ReportBot API error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "report_title": "BizMind Weekly Intelligence Report",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": f"Report generation error: {error_msg}",
            "sections": [],
            "action_items": [],
            "bottom_line": "Check system configuration.",
        }