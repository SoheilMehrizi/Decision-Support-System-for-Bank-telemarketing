
from data_ingestion import load_data_from_db
from src.data_preprocessing import cleaning_pipeline_step
from src.models.model_selection  import train_log_compare_models as model_trainer



import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def training_pipeline() -> Dict[str, Any]:
    try:
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

        # Step 3: Identify categorical & numerical columns
        try:
            cat_columns = X_train.select_dtypes(include='object').columns.tolist()
            num_columns = X_train.select_dtypes(include='number').columns.tolist()
        except Exception as e:
            logger.error(f"Error detecting column types: {e}", exc_info=True)
            return {"status": "error", "step": "column_detection", "message": str(e)}

        # Step 4: Train the model
        try:
            results = model_trainer(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                num_columns=num_columns,
                cat_columns=cat_columns,
                models_name=["RandomForest"],
                visualize=False,
            )
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return {"status": "error", "step": "model_training", "message": str(e)}

        return {"status": "success", "results": results}

    except Exception as e:
        logger.critical(f"Unexpected error in training pipeline: {e}", exc_info=True)
        return {"status": "error", "step": "unknown", "message": str(e)}
