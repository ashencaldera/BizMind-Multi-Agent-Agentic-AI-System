"""
Base Agent — All BizMind agents inherit from this class.
Uses Groq API (groq.com) with llama-3.3-70b-versatile model.
"""
import json
import re
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from openai import OpenAI
from core.config import config


class AgentMessage:
    def __init__(self, sender: str, content: str, data: Optional[Dict] = None):
        self.sender = sender
        self.content = content
        self.data = data or {}

    def to_dict(self) -> Dict:
        return {"sender": self.sender, "content": self.content, "data": self.data}


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: str):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.memory: List[Dict] = []
        self.last_insights: Optional[Dict] = None

        try:
            self.client = OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            print(f"  ✓ {self.name} client ready | model={config.MODEL}")
        except Exception as e:
            print(f"  ✗ {self.name} client FAILED: {e}")
            self.client = None

    @property
    def system_prompt(self) -> str:
        return f"""You are {self.name}, an expert {self.role} AI agent in the BizMind business intelligence system.
Your expertise: {self.expertise}

Analyze the business data and respond ONLY with a valid JSON object in exactly this structure:
{{
  "agent": "{self.name}",
  "status": "success",
  "key_metrics": {{"metric_name": "value"}},
  "insights": ["insight 1", "insight 2", "insight 3"],
  "risks": ["risk 1", "risk 2"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "recommendations": [
    {{"action": "specific action", "priority": "high", "expected_impact": "expected result"}}
  ],
  "summary": "One sentence executive summary with key numbers"
}}

CRITICAL: Return ONLY the raw JSON object. No markdown, no code fences, no explanation text before or after."""

    def _clean_json(self, raw: str) -> str:
        """Robustly clean and extract JSON from model response."""
        cleaned = raw.strip()

        # Remove markdown code fences
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # If still doesn't start with {, find the JSON object
        if not cleaned.startswith('{'):
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        return cleaned

    def think(self, prompt: str, context: Optional[str] = None) -> Dict:
        """Core reasoning — sends data to Groq and returns structured JSON analysis."""

        if self.client is None:
            return self._error_result("Groq client not initialized. Check GROQ_API_KEY in .env")

        messages = []
        for mem in self.memory[-2:]:
            messages.append(mem)

        user_content = prompt
        if context:
            user_content = f"Business Data:\n{context}\n\nTask: {prompt}"

        messages.append({"role": "user", "content": user_content})

        try:
            print(f"  📡 {self.name} → Groq API...")

            response = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt}
                ] + messages,
            )

            raw = response.choices[0].message.content
            print(f"  📨 {self.name} ← {len(raw)} chars received")

            cleaned = self._clean_json(raw)
            result = json.loads(cleaned)

            self.memory.append({"role": "user", "content": user_content})
            self.memory.append({"role": "assistant", "content": raw})
            self.last_insights = result
            return result

        except json.JSONDecodeError as e:
            print(f"\n{'='*55}")
            print(f"❌ JSON ERROR — {self.name}")
            print(f"   Parse error: {e}")
            print(f"   Raw response:\n{raw[:800] if 'raw' in dir() else 'no response'}")
            print(f"{'='*55}\n")
            return self._error_result(f"JSON parse failed: {e}")

        except Exception as e:
            print(f"\n{'='*55}")
            print(f"❌ API ERROR — {self.name}")
            print(f"   Type   : {type(e).__name__}")
            print(f"   Message: {str(e)}")
            traceback.print_exc()
            print(f"{'='*55}\n")
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "agent": self.name,
            "status": "error",
            "key_metrics": {},
            "insights": [f"Agent error: {error_msg}"],
            "risks": [],
            "opportunities": [],
            "recommendations": [],
            "summary": f"{self.name} encountered an error: {error_msg}",
        }

    @abstractmethod
    def analyze(self) -> Dict:
        pass

    def receive_message(self, message: AgentMessage) -> Optional[str]:
        return None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "has_insights": self.last_insights is not None,
        }