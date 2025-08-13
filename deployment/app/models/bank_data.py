from sqlalchemy import Column, Integer, String, Boolean, Float, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BankData(Base):
    __tablename__ = "bank_data"

    id = Column(Integer, primary_key=True, index=True)

    age = Column(Integer, nullable=False, index=True)
    job = Column(String, index=True)
    marital = Column(String, index=True)
    education = Column(String)
    default = Column(String)  # TODO: refactor
    balance = Column(Float)
    housing = Column(String) # TODO: refactor
    loan = Column(String) # TODO: refactor
    contact = Column(String)
    day = Column(Integer)
    month = Column(String, index=True)
    duration = Column(Integer)
    campaign = Column(Integer)
    pdays = Column(Integer)
    previous = Column(Integer)
    poutcome = Column(String)
    y = Column(String, index=True)  # TODO: refactor
    training_data = Column(Boolean, index=True)

    __table_args__ = (
        CheckConstraint('age >= 18 AND age <= 100', name='valid_age'),
        CheckConstraint('balance >= -1000000', name='reasonable_balance'),
        CheckConstraint('day >= 1 AND day <= 31', name='valid_day'),
        CheckConstraint('duration >= 0', name='non_negative_duration')
    )

    def __repr__(self):
        return f"<BankData(id={self.id}, age={self.age}, job={self.job}, y={self.y})>"
