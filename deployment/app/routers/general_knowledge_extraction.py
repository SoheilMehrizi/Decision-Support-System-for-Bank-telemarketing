

from utils.token_verification import get_current_user
from fastapi import APIRouter, Depends, status

from sqlalchemy.orm import Session
from database import get_db

from dependencies.auth import get_current_superuser

router = APIRouter()

def train_model():
    # This simulates your model training cycle
    print("Hello world - model training triggered")

@router.post(
    "/general-knowledge/",
    summary="Trigger model training cycle (hello world)",
    status_code=status.HTTP_200_OK,
)
def trigger_model_training(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    train_model()
    return {"message": "Model training triggered successfully."}
