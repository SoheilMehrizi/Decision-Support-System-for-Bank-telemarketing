from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from configs.config_repository import ConfigRepository
from src.models.model_repository import ModelRepository
from sklearn.preprocessing import FunctionTransformer

from src.data_preprocessing import create_preprocessor

def train_svc(X_train, y_train, 
              num_columns, cat_columns,
               cv_splits=5, random_state=42):

   """
    Train a Support Vector Classifier (SVC) model using randomized hyperparameter search with 
    halving search CV, while applying preprocessing and class balancing within a scikit-learn pipeline.

    This function constructs a machine learning pipeline that includes:
    - Preprocessing of numeric and categorical features via `create_preprocessor`.
    - Oversampling of the minority class using `RandomOverSampler`.
    - Hyperparameter tuning using `HalvingRandomSearchCV`.
    - Model logging and registration for reproducibility and versioning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features. Should include both numeric and categorical columns.
    
    y_train : pd.Series
        Training target labels (binary classification).
    
    num_columns : list of str
        List of column names in `X_train` representing numerical features.
    
    cat_columns : list of str
        List of column names in `X_train` representing categorical features.
    
    cv_splits : int, default=5
        Number of folds for stratified cross-validation during hyperparameter search.
    
    random_state : int, default=42
        Random seed for reproducibility in cross-validation, model, and resampling.

    Returns
    -------
    refernce_pipeline : ImbPipeline
        A fitted pipeline including preprocessing and the best SVC model, ready for prediction or deployment.
    
    best_params : dict
        The best hyperparameter configuration found during the search.
    
    metrics : dict
        Dictionary containing evaluation metrics (e.g., cross-validated ROC AUC).
   """
   
   config_repo = ConfigRepository(config_path="../configs/models_config.json")
   param_dist = config_repo.get_config("svc")

   cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

   svc_model = SVC(probability=True, random_state=random_state)

   preprocessing_pipeline = create_preprocessor(num_features=num_columns,
                                                 cat_features=cat_columns)

   pipeline = ImbPipeline([
      ('preprocess', preprocessing_pipeline),
      ("smote", RandomOverSampler(random_state=42)),
      ("clf", svc_model) 
   ])

   halving_search = HalvingRandomSearchCV(
       estimator=pipeline,
       param_distributions=param_dist,
       factor=3,
       resource="n_samples",
       cv=cv,
       scoring='roc_auc',
       n_jobs=-1,
       verbose=1,
       random_state=random_state
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
   run_id = repo.log_model(refernce_pipeline, model_name="svm_model", params=best_params, metrics=metrics)
   registered_model = repo.register_model(run_id, model_name="svm_model", registered_name="BankMarketing_SVC")

   return refernce_pipeline, best_params, metrics
