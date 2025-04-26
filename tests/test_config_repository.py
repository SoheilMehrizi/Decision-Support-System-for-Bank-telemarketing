# tests/test_config_repository.py

import threading
import pytest
from pathlib import Path
from scipy.stats import randint, loguniform

from configs.config_repository import ConfigRepository

def setup_tmp_repo(tmp_path, monkeypatch):
    # tmp_path is a unique pathlib.Path per test function :contentReference[oaicite:2]{index=2}
    # monkeypatch.chdir isolates CWD so models_config.json lives only in tmp_path :contentReference[oaicite:3]{index=3}
    monkeypatch.chdir(tmp_path)
    return ConfigRepository()

def test_add_and_get_simple_config(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    model_name = "simple_model"
    config = {"param1": 10, "param2": [1, 2, 3]}
    repo.add_config(model_name, config)
    loaded = repo.get_config(model_name)
    assert loaded == config  # simple dict equality, no type checks here

def test_add_and_get_complex_config(tmp_path, monkeypatch):
    repo = setup_tmp_repo(tmp_path, monkeypatch)
    model_name = "complex_model"

    # Create original frozen distributions
    orig_int_rv = randint(0, 5)
    orig_float_rv = loguniform(1e-3, 1)

    config = {
        "int_range": orig_int_rv,
        "float_range": orig_float_rv
    }

    repo.add_config(model_name, config)
    loaded = repo.get_config(model_name)

    # Use each original object’s type for isinstance checks :contentReference[oaicite:4]{index=4}
    assert isinstance(loaded["int_range"], type(orig_int_rv))
    assert loaded["int_range"].args == (0, 5)

    assert isinstance(loaded["float_range"], type(orig_float_rv))
    # loguniform stores its bounds in .a and .b :contentReference[oaicite:5]{index=5}
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

    # No thread should have errored :contentReference[oaicite:6]{index=6}
    assert not errors, f"Errors occurred in threads: {errors}"
