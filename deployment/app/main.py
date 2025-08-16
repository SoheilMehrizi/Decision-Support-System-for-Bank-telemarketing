from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from repositories.bank_data_repository import BankDataRepository
from schemas.bank_data import BankDataCreate, BankData
from routers import auth

from models.users import  User

app = FastAPI()

# # Create database tables
Base.metadata.create_all(bind=engine)

app.include_router(auth.router, prefix="/auth")