
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold

from scipy.stats import loguniform

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from src.models.model_repository import ModelRepository
from configs.config_repository import ConfigRepository
from sklearn.preprocessing import FunctionTransformer

from src.data_preprocessing import create_preprocessor

def train_logistic_regression(X_train, y_train, num_columns, cat_columns,
                               cv_splits= 5, random_state=42):

   """
    Train a Logistic Regression model with hyperparameter tuning and final retraining.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target labels.
        num_columns (list): List of numerical feature column names.
        cat_columns (list): List of categorical feature column names.
        cv_splits (int): Number of cross-validation splits (default=5).
        random_state (int): Seed for reproducibility (default=42).

    Returns:
        final_pipeline (Pipeline): Fully trained pipeline (preprocessing + classifier).
        best_params (dict): Best hyperparameters found during tuning.
        metrics (dict): Dictionary containing cross-validated ROC AUC.
   """

   repo = ConfigRepository(config_path="../configs/models_config.json")
   param_dist = repo.get_config("logistic_regression")

   cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
   lr_model = LogisticRegression(max_iter=1000, random_state=random_state)

   preprocessing_pipeline = create_preprocessor(num_features=num_columns,
                                                 cat_features=cat_columns)

   pipeline = ImbPipeline([
      ('preprocess',  preprocessing_pipeline),
      ("smote", RandomOverSampler(random_state=42)),
      ("clf", lr_model) 
   ])

   halving_search = HalvingRandomSearchCV(
       estimator = pipeline,
       param_distributions=param_dist,
       factor = 3,                    
       resource="n_samples",              
       cv=cv,
       scoring='roc_auc',           
       random_state=random_state,
       n_jobs=-1,
       verbose=1
   )

    
   halving_search.fit(X_train, y_train)
    
   best_model = halving_search.best_estimator_
    
   best_params = halving_search.best_params_
    
   metrics = {"cv_roc_auc": halving_search.best_score_}
    
   trained_clf = best_model.named_steps['clf']
   fitted_preprocessor = best_model.named_steps['preprocess']

   refernce_pipeline = ImbPipeline([
       ('preprocess', fitted_preprocessor),
       ("clf", trained_clf)
   ])

   repo = ModelRepository(experiment_name="Bank_Marketing_Models")
   run_id = repo.log_model(refernce_pipeline, model_name="logistic_regression_model", params=best_params, metrics=metrics)
   registered_model = repo.register_model(run_id, model_name="logistic_regression_model", registered_name="LogReg")
    
   return refernce_pipeline, best_params, metrics