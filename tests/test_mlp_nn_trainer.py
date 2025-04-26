import numpy as np
import pytest
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_frozen

# Import the function under test
from src.models.neural_network import train_mlp_nn

# Dummy Keras model stub with predict
class DummyKerasModel:
    """Stub estimator providing predict method for ROC AUC computation"""
    def predict(self, X):
        arr = np.asarray(X).flatten().astype(float)
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        return norm.reshape(-1, 1)

# Dummy HalvingRandomSearchCV stub
class DummySearch:
    """Stub for HalvingRandomSearchCV"""
    def __init__(self, *args, **kwargs):
        pdist = kwargs.get('param_distributions') or (args[1] if len(args) > 1 else {})
        assert 'epochs' in pdist
        assert isinstance(pdist['epochs'], rv_frozen)
        assert kwargs.get('resource') == 'epochs'
        assert kwargs.get('max_resources') == pdist['epochs'].a
        assert kwargs.get('scoring') == 'roc_auc'
        # container with .model_ attribute for KerasClassifier
        container = type('C', (), {})()
        container.model_ = DummyKerasModel()
        self.best_estimator_ = container
        dropout_vals = pdist.get('dropout', [0.5])
        self.best_params_ = {'epochs': pdist['epochs'].a, 'dropout': dropout_vals[0]}
        self.best_score_ = 0.9

    def fit(self, X, y):
        assert hasattr(X, 'shape') and hasattr(y, 'shape')
        return self

# Dummy ConfigRepository stub
class DummyConfigRepo:
    def __init__(self, path=None):
        assert path == "dummy_config.json"
    def get_config(self, name):
        assert name == "mlp"
        return {'epochs': randint(1, 5), 'dropout': [0.3, 0.5]}

# Dummy ModelRepository stub
class DummyModelRepo:
    def __init__(self, experiment_name):
        assert experiment_name == "Bank_Marketing_Models"
    def log_model(self, model, model_name, params, metrics):
        assert isinstance(model, DummyKerasModel)
        assert model_name == "MLP_model"
        assert 'cv_roc_auc' in metrics and 'train_roc_auc' in metrics
        return "run_id"
    def register_model(self, run_id, model_name, registered_name):
        assert run_id == "run_id"
        assert model_name == "MLP_model"
        assert registered_name == "BankMarketing_MLP"
        return "reg_model"

@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    # Patch dependencies in mlp_nn module
    monkeypatch.setattr('src.models.neural_network.ConfigRepository', DummyConfigRepo)
    monkeypatch.setattr('src.models.neural_network.HalvingRandomSearchCV', DummySearch)
    monkeypatch.setattr('src.models.neural_network.ModelRepository', DummyModelRepo)
    yield


def test_train_mlp_nn_returns_expected():
    X_train = np.array([[0], [1], [2], [3], [4]])
    y_train = np.array([0, 0, 1, 1, 1])

    best_model, best_params, metrics = train_mlp_nn(
        X_train, y_train,
        cv_splits=2,
        random_state=123,
        config_path="dummy_config.json",
        log_dir="dummy_log",
        validation_split=0.2
    )

    assert isinstance(best_model, DummyKerasModel)
    assert best_params['dropout'] == 0.3
    assert metrics['cv_roc_auc'] == 0.9
    assert 0.0 <= metrics['train_roc_auc'] <= 1.0