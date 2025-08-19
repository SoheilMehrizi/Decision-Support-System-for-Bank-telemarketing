from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import get_db
from repositories.users import UserRepository
from dependencies.auth import get_current_superuser, get_current_user
from utils.jwt import create_access_token
from schemas.users import Token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    repo = UserRepository(db)
    user = repo.authenticate(form_data.username, form_data.password)
    if not user or not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password or insufficient privileges",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/protected-route")
def protected_route(current_user = Depends(get_current_superuser)):
    return {"msg": f"Hello, superuser {current_user.username}!"}


