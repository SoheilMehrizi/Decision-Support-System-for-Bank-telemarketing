from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class KnowledgeExtractionRequest(BaseModel):
    age: Optional[int] = None
    job: Optional[str] = None
    marital: Optional[str] = None
    education: Optional[str] = None
    default: Optional[str] = None   # TODO: refactor
    balance: Optional[float] = None
    housing: Optional[str] = None   # TODO: refactor
    loan: Optional[str] = None
    month: Optional[str] = None