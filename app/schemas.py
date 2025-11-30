# app/schemas.py
from pydantic import BaseModel
from typing import Any, Dict, Optional

class DumpIn(BaseModel):
    dump_header: Dict[str, Any]
    dump_code: Optional[Any] = None
    callstack: Optional[Any] = None
    # allow additional fields
    class Config:
        extra = "allow"

class AnalysisOut(BaseModel):
    dump_id: str
    dump: DumpIn
    ai_summary: Any
    priority: Optional[str]
    raw_llm: Dict[str, Any]
