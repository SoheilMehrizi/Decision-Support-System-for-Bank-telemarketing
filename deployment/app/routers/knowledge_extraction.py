from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from dependencies.auth import get_current_user

from src.pipelines.prediction_pipeline import (
    extract_general_rules_pipeline,
    extract_local_rules_pipeline,
)
from schemas.knowledge_extraction import (
    KnowledgeExtractionRequest,
    KnowledgeExtractionResponse,
)

router = APIRouter(prefix="/ML", tags=["ML"])

@router.post(
    "/general-knowledge/",
    summary="Extract general or local rules from the model",
    response_model=KnowledgeExtractionResponse,
    status_code=status.HTTP_200_OK,
)
def trigger_knowledge_extraction(
    request: KnowledgeExtractionRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Filter out None values → only pass provided conditions
    input_conditions = request.dict(exclude_none=True)

    if input_conditions:  # Local rules if user provided conditions
        result = extract_local_rules_pipeline(input_conditions=input_conditions)
    else:  # General rules otherwise
        result = extract_general_rules_pipeline()

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)

    return result
