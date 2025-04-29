from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from configs.config_repository import ConfigRepository
from src.models.model_repository import ModelRepository

def train_svm(X_train, y_train, cv_splits=5, random_state=42):

    config_repo = ConfigRepository(config_path="../configs/models_config.json")
    param_dist = config_repo.get_config("svm")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    svc_model = SVC(probability=True, random_state=random_state)


    pipeline = Pipeline([
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

    repo = ModelRepository(experiment_name="Bank_Marketing_Models")
    run_id = repo.log_model(best_model, model_name="svm_model", params=best_params, metrics=metrics)
    registered_model = repo.register_model(run_id, model_name="svm_model", registered_name="BankMarketing_SVM")

    return best_model, best_params, metrics
