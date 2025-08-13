from fastapi import APIRouter, Depends, status

from sqlalchemy.orm import Session
from database import get_db

from utils.token_verification import get_current_user


router = APIRouter()

def predict():
    # This simulates your model training cycle
    print("Hello world - model training triggered")

@router.post(
    "/predict/",
    summary="Trigger model training cycle (hello world)",
    status_code=status.HTTP_200_OK,
)
def prediction(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    predict()
    return {"message": "Model training triggered successfully."}


#TODO: it should get a distinct schema (For one or a batch)
#TODO: it would return predicted label alongside explicit knowledge