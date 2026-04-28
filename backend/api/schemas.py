from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class QuestionRequest(BaseModel):
    question: str

class ScenarioRequest(BaseModel):
    scenario: str

class AgentRunRequest(BaseModel):
    agent: str  # "sales" | "customer" | "inventory" | "marketing" | "forecasting" | "all"

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

class QuestionResponse(BaseModel):
    success: bool
    question: str
    answer: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    agents_available: List[str]
