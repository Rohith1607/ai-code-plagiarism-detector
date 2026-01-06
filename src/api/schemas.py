from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=50000)
    language: Optional[str] = None


class AnalyzeResponse(BaseModel):
    plagiarism_percentage: float
    ai_probability: float
    confidence: str