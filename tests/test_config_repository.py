import threading
import pytest
from pathlib import Path
from scipy.stats import randint, loguniform

from configs.config_repository import ConfigRepository


def setup_tmp_repo(tmp_path, monkeypatch):
    """
    Create a temporary config file in the tmp_path and initialize ConfigRepository
    with that path to isolate tests from the real config file location.
    """
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "models_config.json"
    config_file.write_text("{}", encoding='utf-8')

    return ConfigRepository(config_path=str(config_file))


def test_add_and_get_simple_config(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    model_name = "simple_model"
    config = {"param1": 10, "param2": [1, 2, 3]}
    repo.add_config(model_name, config)
    loaded = repo.get_config(model_name)
    assert loaded == config


def test_add_and_get_complex_config(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    model_name = "complex_model"

    orig_int_rv = randint(0, 5)
    orig_float_rv = loguniform(1e-3, 1)

    config = {
        "int_range": orig_int_rv,
        "float_range": orig_float_rv
    }

    repo.add_config(model_name, config)
    loaded = repo.get_config(model_name)

    assert isinstance(loaded["int_range"], type(orig_int_rv))
    assert loaded["int_range"].args == (0, 5)

    assert isinstance(loaded["float_range"], type(orig_float_rv))
    assert loaded["float_range"].a == 1e-3
    assert loaded["float_range"].b == 1


def test_remove_config(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    model_name = "model_to_remove"
    repo.add_config(model_name, {"x": 1})
    repo.remove_config(model_name)
    with pytest.raises(KeyError):
        repo.get_config(model_name)


def test_list_models(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    names = ["m1", "m2", "m3"]
    for name in names:
        repo.add_config(name, {"x": name})
    listed = repo.list_models()
    assert set(listed) == set(names)


def test_thread_safety(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    errors = []

    def worker(name, config):
        try:
            repo.add_config(name, config)
            assert repo.get_config(name) == config
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(f"model_{i}", {"i": i}))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors occurred in threads: {errors}"
