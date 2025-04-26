import numpy as np
import pytest

# Import the train_logistic_regression function from its module
from src.models.logistic_regression import train_logistic_regression
from scipy.stats._distn_infrastructure import rv_frozen

# Dummy estimator to support predict_proba for AUC computation
class DummyEstimator:
    """Stub estimator providing predict_proba method for ROC AUC computation"""
    def predict_proba(self, X):
        arr = np.asarray(X).flatten().astype(float)
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        return np.vstack([1 - norm, norm]).T

class DummySearch:
    """Stub for HalvingRandomSearchCV"""
    def __init__(self, *args, **kwargs):
        # Extract the param distributions
        pdist = kwargs.get('param_distributions') or (args[1] if len(args) > 1 else None)
        # Ensure correct keys
        assert set(pdist.keys()) == {'C', 'penalty'}
        # Check that C is a frozen loguniform distribution
        C_dist = pdist['C']
        assert isinstance(C_dist, rv_frozen)
        assert C_dist.args == (1e-4, 10)
        # Check penalty options list
        assert isinstance(pdist['penalty'], list)
        assert pdist['penalty'] == ['l1', 'l2']
        # Check other HalvingRandomSearchCV settings
        assert kwargs.get('resource') == 'max_iter'
        assert kwargs.get('min_resources') == 100
        assert kwargs.get('max_resources') == 500
        assert kwargs.get('scoring') == 'roc_auc'

        # Provide dummy search outcomes
        self.best_estimator_ = DummyEstimator()
        self.best_params_ = {'C': C_dist.args, 'penalty': pdist['penalty'][0]}
        self.best_score_ = 0.85

    def fit(self, X, y):
        assert hasattr(X, 'shape') and hasattr(y, 'shape')
        return self

class DummyConfigRepo:
    def __init__(self, path=None):
        # Config path provided by function
        assert path is None or isinstance(path, str)

    def get_config(self, name):
        assert name == "logistic_regression"
        from scipy.stats import loguniform
        return {"C": loguniform(1e-4, 10), "penalty": ["l1", "l2"]}

class DummyModelRepo:
    def __init__(self, experiment_name):
        assert experiment_name == "Bank_Marketing_Models"

    def log_model(self, model, model_name, params, metrics):
        assert isinstance(model, DummyEstimator)
        assert model_name == "logistic_regression_model"
        assert isinstance(params, dict)
        assert "cv_roc_auc" in metrics
        return "dummy_run_id"

    def register_model(self, run_id, model_name, registered_name):
        assert run_id == "dummy_run_id"
        assert model_name == "logistic_regression_model"
        assert registered_name == "BankMarketing_LogReg"
        return "dummy_registered_model"

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch ConfigRepository, HalvingRandomSearchCV, and ModelRepository in the target module
    monkeypatch.setattr('src.models.logistic_regression.ConfigRepository', DummyConfigRepo)
    monkeypatch.setattr('src.models.logistic_regression.HalvingRandomSearchCV', DummySearch)
    monkeypatch.setattr('src.models.logistic_regression.ModelRepository', DummyModelRepo)
    yield


def test_train_logistic_regression_returns_expected():
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([0, 0, 1, 1])

    best_model, best_params, metrics = train_logistic_regression(
        X_train, y_train,
        cv_splits=2,
        random_state=123
    )

    assert isinstance(best_model, DummyEstimator)
    # C parameter should be the args tuple of the loguniform
    assert best_params['C'] == (1e-4, 10)
    assert best_params['penalty'] == 'l1'
    assert metrics == {'cv_roc_auc': 0.85}
