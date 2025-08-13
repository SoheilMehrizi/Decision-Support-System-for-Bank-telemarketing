import getpass
from sqlalchemy.orm import Session
from deployment.app.database import SessionLocal
from repositories.users import UserRepository

def main():
    db: Session = SessionLocal()

    username = input("Superuser username: ")
    email = input("Superuser email: ")
    password = getpass.getpass("Superuser password: ")

    repo = UserRepository(db)
    user = repo.create(username=username, email=email, password=password, is_superuser=True)

    print(f"Superuser '{user.username}' created successfully.")

if __name__ == "__main__":
    main()
