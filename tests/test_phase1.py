import json
import numpy as np
from src.utils import (load_config, set_global_seed, deep_update, config_hash,
                       save_json, environment_report)


def test_config_loads_and_has_required_sections():
    cfg = load_config()
    for key in ["data", "target_model", "uncertainty", "validator", "search",
                "gp_bo", "cmaes", "quantum", "ablations", "reproducibility"]:
        assert key in cfg
    assert cfg["reproducibility"]["seeds"] == list(range(1, 11))


def test_seed_determinism():
    a = set_global_seed(7).random(5)
    b = set_global_seed(7).random(5)
    assert np.allclose(a, b)


def test_overrides_do_not_mutate_base_and_change_hash():
    base = load_config()
    mod = load_config(overrides={"quantum": {"n_qubits": 12}})
    assert base["quantum"]["n_qubits"] == 8 and mod["quantum"]["n_qubits"] == 12
    assert mod["quantum"]["n_layers"] == base["quantum"]["n_layers"]
    assert config_hash(base) != config_hash(mod)


def test_save_json_handles_numpy(tmp_path):
    p = save_json({"x": np.arange(3), "y": np.float32(1.5)}, tmp_path / "a.json")
    assert json.load(open(p)) == {"x": [0, 1, 2], "y": 1.5}


def test_environment_report_keys():
    env = environment_report()
    assert "python" in env and "numpy" in env["packages"]
