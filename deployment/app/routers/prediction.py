from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from database import get_db
from dependencies.auth import get_current_user

from src.pipelines.prediction_pipeline import (
    load_and_prpare_data,
    predict as pipeline_predict,
)
from schemas.prediction import PredictionRequest, PredictionResponse, BankFeatures

import pandas as pd


router = APIRouter(prefix="/ML", tags=["ML"])


@router.post(
    "/predict/",
    summary="Run predictions using the registered ML model",
    status_code=status.HTTP_200_OK,
    response_model=PredictionResponse,
)
def prediction(
    payload: PredictionRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Decide source of features: user-provided payload or DB test split
    if payload.data is not None:
        # Normalize to list of dicts
        if isinstance(payload.data, list):
            records = [item.model_dump() for item in payload.data]
        else:
            records = [payload.data.model_dump()]

        X = pd.DataFrame.from_records(records)
    else:
        _y_train, _y_test, _X_train, X = load_and_prpare_data()

    y_pred = pipeline_predict(X, estimator=payload.estimator)

    # Ensure JSON serializable
    predictions_list = getattr(y_pred, "tolist", lambda: list(y_pred))()

    return PredictionResponse(
        estimator=payload.estimator,
        count=len(predictions_list),
        predictions=predictions_list,
    )