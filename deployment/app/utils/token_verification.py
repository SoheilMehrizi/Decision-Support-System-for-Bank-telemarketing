from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jwt import verify_access_token
from jose import JWTError
from fastapi import HTTPException

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        username = verify_access_token(token)
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
