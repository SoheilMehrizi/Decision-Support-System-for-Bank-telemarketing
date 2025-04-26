import numpy as np
import pytest

from src.models.svm import train_svm


class DummySearch:
    """Stub for HalvingRandomSearchCV"""
    def __init__(
        self, estimator, param_distributions, factor, resource,
        min_resources, max_resources, cv, scoring, n_jobs, verbose, random_state
    ):
        
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_estimator_ = "best_svc"
        self.best_params_ = {k: v[0] for k, v in param_distributions.items()}
        self.best_score_ = 0.75

    def fit(self, X, y):
        
        assert hasattr(X, 'shape') and hasattr(y, 'shape')
        return self

class DummyConfigRepo:
    def __init__(self, path):
        
        assert path == "dummy_configs.json"
    def get_config(self, name):
        
        assert name == "svm"
        return {"C": [0.1, 1], "gamma": [0.01, 0.1]}

class DummyModelRepo:
    def __init__(self, experiment_name):
        
        assert experiment_name == "Bank_Marketing_Models"
    def log_model(self, model, model_name, params, metrics):
        
        assert model == "best_svc"
        assert model_name == "svm_model"
        assert isinstance(params, dict)
        assert "cv_roc_auc" in metrics
        return "dummy_run_id"
    def register_model(self, run_id, model_name, registered_name):
        
        assert run_id == "dummy_run_id"
        assert model_name == "svm_model"
        assert registered_name == "BankMarketing_SVM"
        return "dummy_registered_model"

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    
    monkeypatch.setattr('src.models.svm.ConfigRepository', DummyConfigRepo)  
    monkeypatch.setattr('src.models.svm.HalvingRandomSearchCV', DummySearch)  
    monkeypatch.setattr('src.models.svm.ModelRepository', DummyModelRepo)    
    yield


def test_train_svm_returns_expected():
    # Create minimal dummy data
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    # Call train_svm with the dummy config path
    best_model, best_params, metrics = train_svm(
        X_train, y_train,
        config_path="dummy_configs.json",
        cv_splits=2,
        random_state=123
    )

    # Assert the outputs match our dummy search's attributes
    assert best_model == "best_svc"
    assert best_params == {"C": 0.1, "gamma": 0.01}
    assert metrics == {"cv_roc_auc": 0.75}
