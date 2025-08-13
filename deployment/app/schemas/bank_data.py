from pydantic import BaseModel, Field
from typing import Optional

# --- Create schema ---
class BankDataCreate(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str
    marital: str 
    education: str
    default: str #TODO: refactor
    balance: float
    housing: str #TODO: refactor
    loan: str #TODO: refactor
    contact: str
    day: int = Field(..., ge=1, le=31)
    month: str
    duration: int = Field(..., ge=0)
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    y: str #TODO: refactor
    index: int
    training_data: bool


# --- Response schema ---
class BankData(BaseModel):
    id: int
    age: int
    job: str
    marital: str
    education: str
    default: str #TODO: refactor
    balance: float
    housing: str #TODO: refactor
    loan: str #TODO: refactor
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    y: str #TODO: refactor
    index: int
    training_data: bool

    class Config:
        from_attributes = True
