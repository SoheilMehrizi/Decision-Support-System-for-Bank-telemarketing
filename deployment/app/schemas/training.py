from pydantic import BaseModel
from typing import Dict, Any, Optional

class TrainingResult(BaseModel):
    status: str
    step: Optional[str] = None
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None