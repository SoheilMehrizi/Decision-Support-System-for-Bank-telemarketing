from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Union, List

from schemas.bank_data import BankDataCreate, BankData as BankDataSchema
from repositories.bank_data_repository import BankDataRepository
from database import get_db  
from utils.token_verification import get_current_user

router = APIRouter()



@router.post(
    "/bankdata/",
    status_code=status.HTTP_201_CREATED,
    response_model=Union[BankDataSchema, List[BankDataSchema]],
    summary="Add new bank data (single or batch)",
)
def add_bank_data(
    bank_data: Union[BankDataCreate, List[BankDataCreate]],
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create new bank data record(s) after verifying token.

    Args:
        bank_data: Single BankDataCreate or list of BankDataCreate.
        current_user: Username extracted from token, guaranteed by Depends.
        db: DB session.

    Returns:
        Created BankData instance(s).
    """
    repo = BankDataRepository(db)
    result = repo.create_bank_data(bank_data)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create bank data record(s).",
        )

    return result


@router.post(
    "/bankdata/slide-window/",
    summary="Move 10% test data to training data (slide window)",
    status_code=status.HTTP_200_OK,
)
def trigger_slide_window(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Moves 10% of test data to training data using the sliding window method.
    Requires valid JWT token.
    Returns the number of moved records.
    """
    repo = BankDataRepository(db)
    moved_count = repo.slide_window()

    if moved_count is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform slide window operation.",
        )

    return {"moved_records": moved_count}