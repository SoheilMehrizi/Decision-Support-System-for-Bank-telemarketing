from src.models.model_repository import ModelRepository
from data_ingestion import load_data_from_db
from src.data_preprocessing import cleaning_pipeline_step
from src.knowledge_extraction import extract_surrogate_rules, extract_local_rules
import logging
from typing import Dict, Any
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def load_and_prpare_data():

        # Step 1: Load the Data
    try:
        df_train, df_test = load_data_from_db()
        if df_train.empty or df_test.empty:
            raise ValueError("Training or test dataset is empty.")
    except Exception as e:
        logger.error(f"Error loading data from DB: {e}", exc_info=True)
        return {"status": "error", "step": "load_data", "message": str(e)}

    # Step 2: Cleaning
    try:
        df_train_cleaned = cleaning_pipeline_step(df_train, outlier_removal=True)
        df_test_cleaned = cleaning_pipeline_step(df_test, outlier_removal=True)
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}", exc_info=True)
        return {"status": "error", "step": "cleaning", "message": str(e)}

    try:
        y_train = df_train_cleaned['y']
        X_train = df_train_cleaned.drop(columns=['y'])
        y_test = df_test_cleaned['y']
        X_test = df_test_cleaned.drop(columns=['y'])
    except KeyError as e:
        logger.error(f"Missing target column 'y' in cleaned data: {e}", exc_info=True)
        return {"status": "error", "step": "splitting", "message": "Missing target column 'y'."}
    return (y_train, y_test, X_train, X_test)


def predict(X, estimator = 'Random_Forest'):

    model_repo = ModelRepository()
    RF_pretrained = model_repo.load_model(registered_name = estimator)

    y_pred = RF_pretrained.predict(X)

    return y_pred

def extract_general_rules_pipeline(estimator = 'Random_Forest') -> Any:
    try:
        # Step 1: Load the model
        _, _, X_train, _=load_and_prpare_data()
        X = X_train

        try:
            model_repo = ModelRepository()
            RF_pretrained = model_repo.load_model(registered_name=estimator)
            if RF_pretrained is None:
                raise ValueError(f"No model found for registered name: {estimator}")
        except (SQLAlchemyError, FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading model '{estimator}': {e}", exc_info=True)
            return {"status": "error", "step": "load_model", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
            return {"status": "error", "step": "load_model", "message": str(e)}

        # Step 2: Predict
        try:
            y_pred = RF_pretrained.predict(X)
        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            return {"status": "error", "step": "prediction", "message": str(e)}

        # Step 3: Extract rules
        try:
            rules_df = extract_surrogate_rules(
                pipeline=RF_pretrained,
                X=X,
                y_pred_pipeline=y_pred,
                feature_names=X.columns.tolist(),
                max_depth=4
            )
        except Exception as e:
            logger.error(f"Error extracting surrogate rules: {e}", exc_info=True)
            return {"status": "error", "step": "rule_extraction", "message": str(e)}

        # Step 4: Return success
        return {"status": "success", "rules": rules_df}

    except Exception as e:
        logger.critical(f"Unexpected error in extract_general_rules: {e}", exc_info=True)
        return {"status": "error", "step": "unknown", "message": str(e)}


def extract_local_rules_pipeline(input_conditions: dict ,estimator = 'Random_Forest') -> Any:
    try:
        # Step 1: Load the model
        _, _, X_train, _=load_and_prpare_data()
        X = X_train
        try:
            model_repo = ModelRepository()
            RF_pretrained = model_repo.load_model(registered_name=estimator)
            if RF_pretrained is None:
                raise ValueError(f"No model found for registered name: {estimator}")
        except (SQLAlchemyError, FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading model '{estimator}': {e}", exc_info=True)
            return {"status": "error", "step": "load_model", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
            return {"status": "error", "step": "load_model", "message": str(e)}

        # Step 2: Predict
        try:
            y_pred = RF_pretrained.predict(X)
        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            return {"status": "error", "step": "prediction", "message": str(e)}

        # Step 3: Extract local rules
        try:
            rules_df = extract_local_rules(
                pipeline=RF_pretrained,
                X=X,
                y_pred_pipeline=y_pred,
                input_conditions=input_conditions,  
                feature_names=X.columns.tolist(),
                max_depth=4
            )
        except Exception as e:
            logger.error(f"Error extracting local rules: {e}", exc_info=True)
            return {"status": "error", "step": "rule_extraction", "message": str(e)}

        # Step 4: Return success
        return {"status": "success", "rules": rules_df}

    except Exception as e:
        logger.critical(f"Unexpected error in extract_local_rules: {e}", exc_info=True)
        return {"status": "error", "step": "unknown", "message": str(e)}
