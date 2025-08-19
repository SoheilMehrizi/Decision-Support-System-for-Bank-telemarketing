# # from fastapi import Depends, HTTPException, status
# # from fastapi.security import OAuth2PasswordBearer
# # from sqlalchemy.orm import Session
# # from database import get_db
# # from repositories.users import UserRepository
# # from utils.jwt import verify_access_token
# # from models.users import User

# # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# # def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
# #     username = verify_access_token(token)
# #     if username is None:
# #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
# #     repo = UserRepository(db)
# #     user = repo.get_by_username(username)
# #     if user is None:
# #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
# #     if not user.is_active:
# #         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
# #     return user

# # def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
# #     if not current_user.is_superuser:
# #         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
# #     return current_user




# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
# from sqlalchemy.orm import Session
# from database import get_db
# from repositories.users import UserRepository
# from utils.jwt import verify_access_token
# from models.users import User

# # One single oauth2_scheme for the entire app
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
#     username = verify_access_token(token)  # should return payload["sub"]
#     if username is None:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid authentication credentials"
#         )
#     repo = UserRepository(db)
#     user = repo.get_by_username(username)
#     if user is None:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="User not found"
#         )
#     if not user.is_active:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Inactive user"
#         )
#     return user

# def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
#     if not current_user.is_superuser:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions"
#         )
#     return current_user


from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
from repositories.users import UserRepository
from utils.jwt import verify_access_token
from models.users import User

# One single oauth2_scheme for the entire app
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scheme_name="BearerAuth"  # Add this to align with your custom scheme name
)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    username = verify_access_token(token)  # should return payload["sub"]
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    repo = UserRepository(db)
    user = repo.get_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user

def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user