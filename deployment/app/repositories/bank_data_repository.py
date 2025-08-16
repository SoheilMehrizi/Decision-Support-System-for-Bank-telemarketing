import random
import pandas as pd
from typing import Union, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging

from models.bank_data import BankData
from schemas.bank_data import BankDataCreate, BankData as BankDataSchema

logger = logging.getLogger(__name__)

class BankDataRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_bank_data(
        self,
        bank_data: Union[BankDataCreate, List[BankDataCreate]]
    ) -> Optional[Union[BankData, List[BankData]]]:
        """
        Create one or multiple BankData records.

        Args:
            bank_data: single BankDataCreate or list of BankDataCreate.

        Returns:
            Created BankData instance if single input,
            or list of BankData instances if batch input,
            or None on failure.
        """
        try:
            if isinstance(bank_data, list):
                db_objs = [BankData(**data.model_dump()) for data in bank_data]
                self.db.add_all(db_objs)
                self.db.commit()
                for obj in db_objs:
                    self.db.refresh(obj)
                return db_objs
            else:
                db_obj = BankData(**bank_data.model_dump())
                self.db.add(db_obj)
                self.db.commit()
                self.db.refresh(db_obj)
                return db_obj
        except SQLAlchemyError as e:
            logger.error(f"Error creating bank data record(s): {e}")
            self.db.rollback()
            return None

    def get_bank_data_train(self) -> List[dict]:
        """
        Fetch all BankData records where training_data == True.
        Serialize each record to dict using Pydantic model's model_dump() method.
        Returns empty list on failure.
        """
        try:
            records = (
                self.db.query(BankData)
                .filter(BankData.training_data == True)  # noqa: E712
                .all()
            )
            return [BankDataSchema.model_validate(row).model_dump() for row in records]
        except SQLAlchemyError as e:
            logger.error(f"Failed to fetch training bank data: {e}")
            self.db.rollback()
            return []


    def get_bank_data_test(self) -> List[dict]:
        """
        Fetch all BankData records where training_data == True.
        Serialize each record to dict using Pydantic model's model_dump() method.
        Returns empty list on failure.
        """
        try:
            records = (
                self.db.query(BankData)
                .filter(BankData.training_data == False)  # noqa: E712
                .all()
            )
            return [BankDataSchema.model_validate(row).model_dump() for row in records]
        except SQLAlchemyError as e:
            logger.error(f"Failed to fetch training bank data: {e}")
            self.db.rollback()
            return []
        
    
    def import_from_dataframe(self, df: pd.DataFrame) -> Optional[int]:
        """
        Import bank data records from a Pandas DataFrame into the database.

        Args:
            df: Pandas DataFrame with columns matching BankData schema.

        Returns:
            Number of records successfully inserted, or None on failure.
        """
        try:
            records = df.to_dict(orient='records')  # List of dicts, each dict = one row
            db_objs = [BankData(**record) for record in records]
            self.db.add_all(db_objs)
            self.db.commit()
            return len(db_objs)
        except SQLAlchemyError as e:
            logger.error(f"Failed to import data from DataFrame: {e}")
            self.db.rollback()
            return None
        
    def slide_window(self) -> Optional[int]:
        """
        Moves 10% of the test data into training data by:
        - Selecting 10% of test data randomly
        - Removing same number of rows randomly from training data
        - Setting training_data=True for selected test data rows

        Returns:
            Number of test rows moved to training data or None on failure.
        """
        try:
            # Step 1: get all test data ids
            test_query = self.db.query(BankData).filter(BankData.training_data == False)
            test_ids = [row.id for row in test_query.all()]
            if not test_ids:
                logger.info("No test data available to slide.")
                return 0

            # Step 2: sample 10% of test ids randomly
            sample_size = max(1, int(len(test_ids) * 0.10))
            sampled_test_ids = random.sample(test_ids, sample_size)

            # Step 3: count that 10% (sample_size)
            count_to_move = len(sampled_test_ids)

            # Step 4: remove same number from train data randomly
            train_query = self.db.query(BankData).filter(BankData.training_data == True)
            train_ids = [row.id for row in train_query.all()]
            if len(train_ids) < count_to_move:
                logger.warning("Not enough training data to remove.")
                return None

            sampled_train_ids = random.sample(train_ids, count_to_move)

            # Step 5: update training_data flag on test sample to True
            self.db.query(BankData).filter(BankData.id.in_(sampled_test_ids))\
                .update({BankData.training_data: True}, synchronize_session=False)

            # Remove from training data means set training_data=False for those train samples
            self.db.query(BankData).filter(BankData.id.in_(sampled_train_ids))\
                .update({BankData.training_data: False}, synchronize_session=False)

            self.db.commit()
            logger.info(f"Moved {count_to_move} records from test to training data using slide_window.")
            return count_to_move

        except SQLAlchemyError as e:
            logger.error(f"Error in slide_window operation: {e}")
            self.db.rollback()
            return None
