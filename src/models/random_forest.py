from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from src.models.model_repository import ModelRepository
from configs.config_repository import ConfigRepository

from src.data_preprocessing import create_preprocessor

def train_random_forest(X_train, y_train, num_columns, cat_columns,
                         cv_splits= 5, random_state=42):

   """
    Train a Random Forest classifier with preprocessing, class balancing, and 
    hyperparameter tuning using HalvingRandomSearchCV. The final pipeline includes 
    the fitted preprocessor and best estimator and is logged and registered.

    This function performs:
    - Column-wise preprocessing using `create_preprocessor` for numerical and categorical features.
    - Class balancing with `RandomOverSampler` to address imbalanced labels.
    - Hyperparameter tuning using Halving Random Search with cross-validation.
    - Final model logging and registration using ModelRepository.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw input feature data for training.
    y_train : pd.Series
        Target labels. Assumes binary classification.
    num_columns : list of str
        List of numerical feature column names in `X_train`.
    cat_columns : list of str
        List of categorical feature column names in `X_train`.
    cv_splits : int, optional (default=5)
        Number of cross-validation splits for tuning.
    random_state : int, optional (default=42)
        Random seed for reproducibility in training, cross-validation, and oversampling.

    Returns
    -------
    tuple
        A 3-tuple containing:
        - reference_pipeline : imblearn.pipeline.Pipeline
            Final pipeline with fitted preprocessor and best Random Forest model.
        - best_params : dict
            Best hyperparameters found during HalvingRandomSearchCV.
        - metrics : dict
            Dictionary containing the best cross-validation AUC score.
   """
   repo = ConfigRepository(config_path="../configs/models_config.json")
   param_dist = repo.get_config('random_forest')

   cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
   rf_model = RandomForestClassifier(random_state=random_state)
    
   preprocessing_pipeline = create_preprocessor(num_features=num_columns, cat_features=cat_columns)
    
   pipeline = ImbPipeline([
       ('preprocess', preprocessing_pipeline),
       ("smote", RandomOverSampler(random_state=42)),
       ("clf", rf_model) 
   ])
    
   halving_search = HalvingRandomSearchCV(
   estimator=pipeline,
   param_distributions=param_dist,
   factor=3,
   resource="n_samples",
   cv=cv,
   scoring='roc_auc',
   random_state=42,
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
   run_id = repo.log_model(refernce_pipeline, model_name="random_forest_model", params=best_params, metrics=metrics)
   registered_model = repo.register_model(run_id, model_name="random_forest_model", registered_name="Random_Forest")

   return refernce_pipeline, halving_search.best_params_, metrics
