from pydantic import BaseModel, Field
from typing import List, Optional, Union


class PredictionRequest(BaseModel):
    estimator: str = Field(
        default="Random_Forest",
        description="Registered model name to use for prediction",
    )
    data: Optional[Union["BankFeatures", List["BankFeatures"]]] = Field(
        default=None,
        description="Single instance or batch of instances to predict. If omitted, predictions run on test split from DB.",
    )


class PredictionResponse(BaseModel):
    estimator: str
    count: int
    predictions: list


class BankFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

