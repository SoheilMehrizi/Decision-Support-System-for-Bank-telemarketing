
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold

from scipy.stats import loguniform

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from src.models.model_repository import ModelRepository
from configs.config_repository import ConfigRepository

def train_logistic_regression(X_train, y_train, cv_splits= 5, random_state=42):

    repo = ConfigRepository(config_path="../configs/models_config.json")
    param_dist = repo.get_config("logistic_regression")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    
    pipeline = Pipeline([
    ("smote", RandomOverSampler(random_state=42)),
    ("clf", lr_model)
    ])

    halving_search = HalvingRandomSearchCV(
        estimator = lr_model,
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
    
    repo = ModelRepository(experiment_name="Bank_Marketing_Models")
    run_id = repo.log_model(best_model, model_name="logistic_regression_model", params=best_params, metrics=metrics)
    registered_model = repo.register_model(run_id, model_name="logistic_regression_model", registered_name="BankMarketing_LogReg")
    
    return best_model, best_params, metrics