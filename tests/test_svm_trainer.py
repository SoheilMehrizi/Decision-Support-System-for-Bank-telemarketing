import numpy as np
import pytest
from scipy.stats._distn_infrastructure import rv_frozen

# Stub classes for dependencies
class DummySearch:
    """Stub for HalvingRandomSearchCV"""
    def __init__(self, *args, **kwargs):
        # Extract estimator (first positional or via kw)
        estimator = kwargs.get('estimator') if 'estimator' in kwargs else (args[0] if len(args) > 0 else None)
        # Ensure we receive the imblearn Pipeline
        assert estimator.__class__.__name__ == 'Pipeline'
        # Param distributions must be provided
        pdist = kwargs.get('param_distributions') if 'param_distributions' in kwargs else (args[1] if len(args) > 1 else None)
        assert isinstance(pdist, dict)
        assert set(pdist.keys()) == {'C', 'gamma'}
        # C and gamma values as lists
        C_dist = pdist['C']
        gamma_list = pdist['gamma']
        assert isinstance(C_dist, list) or isinstance(C_dist, rv_frozen) or isinstance(C_dist, list)
        # If frozen, check back-conversion
        if isinstance(C_dist, rv_frozen):
            # if config returns frozen, unwrap args
            assert C_dist.args == (0.1, 1)
        else:
            assert C_dist == [0.1, 1]
        assert gamma_list == [0.01, 0.1]
        # Check Hyperband keyword args
        assert kwargs.get('factor') == 3
        assert kwargs.get('resource') == 'n_samples'
        assert isinstance(kwargs.get('cv'), object)  # StratifiedKFold instance
        assert kwargs.get('scoring') == 'roc_auc'
        assert kwargs.get('n_jobs') == -1
        assert kwargs.get('verbose') == 2
        assert kwargs.get('random_state') == 123
        # Provide dummy best results
        self.best_estimator_ = 'best_svc'
        # best_params_ picks first in lists or frozen args
        if isinstance(C_dist, rv_frozen):
            self.best_params_ = {'C': C_dist.args[0] if hasattr(C_dist, 'args') else C_dist, 'gamma': gamma_list[0]}
        else:
            self.best_params_ = {'C': C_dist[0], 'gamma': gamma_list[0]}
        self.best_score_ = 0.75

    def fit(self, X, y):
        # Ensure arrays are passed
        assert hasattr(X, 'shape') and hasattr(y, 'shape')
        return self

class DummyConfigRepo:
    def __init__(self, config_path=None):
        assert isinstance(config_path, str)

    def get_config(self, name):
        assert name == 'svm'
        # Return simple list-based grid
        return {'C': [0.1, 1], 'gamma': [0.01, 0.1]}

class DummyModelRepo:
    def __init__(self, experiment_name=None):
        assert experiment_name == 'Bank_Marketing_Models'

    def log_model(self, model, model_name, params, metrics):
        # Validate logging contract
        assert model == 'best_svc'
        assert model_name == 'svm_model'
        assert isinstance(params, dict)
        assert 'cv_roc_auc' in metrics and metrics['cv_roc_auc'] == 0.75
        return 'dummy_run_id'

    def register_model(self, run_id, model_name, registered_name):
        assert run_id == 'dummy_run_id'
        assert model_name == 'svm_model'
        assert registered_name == 'BankMarketing_SVM'
        return 'dummy_registered_model'

# Monkeypatch dependencies
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr('src.models.svm.ConfigRepository', DummyConfigRepo)
    monkeypatch.setattr('src.models.svm.HalvingRandomSearchCV', DummySearch)
    monkeypatch.setattr('src.models.svm.ModelRepository', DummyModelRepo)
    yield


def test_train_svm_returns_expected():
    from src.models.svm import train_svm

    # Dummy training data
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    # Invoke training
    best_model, best_params, metrics = train_svm(
        X_train, y_train,
        cv_splits=2,
        random_state=123
    )

    # Assertions
    assert best_model == 'best_svc'
    # best_params picks first list values
    assert best_params == {'C': 0.1, 'gamma': 0.01}
    # metrics returned correctly
    assert metrics == {'cv_roc_auc': 0.75}
