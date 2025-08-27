from fastapi import APIRouter, Depends, status, HTTPException

from sqlalchemy.orm import Session
from database import get_db

from dependencies.auth import get_current_user

from src.pipelines.model_training_pipeline import training_pipeline
from schemas.training import TrainingResult

router = APIRouter(prefix="/ML", tags=["ML"])

@router.post(
    "/train-model/",
    summary="Trigger model training cycle.",
    response_model=TrainingResult,
    status_code=status.HTTP_200_OK,
)
def trigger_model_training(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = training_pipeline()
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)
    return result