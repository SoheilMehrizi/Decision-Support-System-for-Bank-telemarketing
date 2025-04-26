import numpy as np
import pytest
from src.models.random_forest import train_random_forest


class DummyEstimator:
    """Stub estimator providing predict_proba method for ROC AUC computation"""
    def predict_proba(self, X):

        arr = np.asarray(X).flatten().astype(float)

        norm = (arr - arr.min()) / (arr.max() - arr.min())

        return np.vstack([1 - norm, norm]).T

class DummySearch:
    """Stub for HalvingRandomSearchCV"""
    def __init__(self, *args, **kwargs):

        self.param_distributions = kwargs.get('param_distributions') or (args[1] if len(args) > 1 else None)

        assert self.param_distributions == {"n_estimators": [100, 200], "min_samples_split": [2, 5]}
        assert kwargs.get('resource') == 'n_estimators'
        assert kwargs.get('max_resources') == 500
        assert kwargs.get('scoring') == 'roc_auc'

        self.best_estimator_ = DummyEstimator()
        self.best_params_ = {k: v[0] for k, v in self.param_distributions.items()}
        self.best_score_ = 0.75

    def fit(self, X, y):

        assert hasattr(X, 'shape') and hasattr(y, 'shape')
        return self

class DummyConfigRepo:
    def __init__(self, path):

        assert path == "dummy_configs.json"

    def get_config(self, name):

        assert name == "random_forest"
        return {"n_estimators": [100, 200], "min_samples_split": [2, 5]}

class DummyModelRepo:
    def __init__(self, experiment_name):

        assert experiment_name == "Bank_Marketing_Models"

    def log_model(self, model, model_name, params, metrics):

        assert isinstance(model, DummyEstimator)
        assert model_name == "Decision_Tree_model"
        assert isinstance(params, dict)
        assert "cv_roc_auc" in metrics
        return "dummy_run_id"

    def register_model(self, run_id, model_name, registered_name):

        assert run_id == "dummy_run_id"
        assert model_name == "Decision_Tree_model"
        assert registered_name == "BankMarketing_DT"
        return "dummy_registered_model"

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):

    monkeypatch.setattr('src.models.random_forest.ConfigRepository', DummyConfigRepo)
    monkeypatch.setattr('src.models.random_forest.HalvingRandomSearchCV', DummySearch)
    monkeypatch.setattr('src.models.random_forest.ModelRepository', DummyModelRepo)
    yield

def test_train_random_forest_returns_expected():
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    best_model, best_params, metrics = train_random_forest(
        X_train, y_train,
        cv_splits=2,
        config_path="dummy_configs.json",
        random_state=123
    )

    assert isinstance(best_model, DummyEstimator)
    assert best_params == {"n_estimators": 100, "min_samples_split": 2}
    assert metrics == {"cv_roc_auc": 0.75}