from sqlalchemy.orm import Session
from models.users import User
from utils.security import hash_password, verify_password

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_username(self, username: str) -> User | None:
        return self.db.query(User).filter(User.username == username).first()

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.get_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def create(self, username: str, email: str, password: str, is_superuser: bool = False) -> User:
        hashed_pw = hash_password(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_pw,
            is_superuser=is_superuser,
            is_active=True,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
