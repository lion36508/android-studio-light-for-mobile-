import json
import pytest

from qmini.config import QMiniConfig


def test_default_and_light_profile_context_are_512():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert QMiniConfig().max_context == 512
    light = QMiniConfig.from_json(root / "configs" / "qmini_light_0.7m.json")
    assert light.max_context == 512


def test_config_round_trip(tmp_path):
    original = QMiniConfig(
        vocab_size=512, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
    )
    path = original.save(tmp_path / "config.json")
    restored = QMiniConfig.from_json(path)
    assert restored.to_dict() == original.to_dict()


def test_unknown_config_keys_fail_loudly():
    with pytest.raises(ValueError, match="Unknown Q-Mini config keys"):
        QMiniConfig.from_dict({"vocab_size": 128, "hidden_szie": 256})


def test_dilation_count_must_match_layers():
    with pytest.raises(ValueError, match="dilations"):
        QMiniConfig.from_dict({"n_layers": 3, "dilations": [1, 2]})

@pytest.mark.parametrize(
    "values",
    [
        {"d_model": 32.5},
        {"dilations": [1.5, 2, 4, 8, 16, 32]},
        {"rms_norm_eps": float("nan")},
        {"seed": -1},
        {"max_context": 1},
    ],
)
def test_invalid_config_values_are_rejected_without_coercion(values):
    with pytest.raises((TypeError, ValueError)):
        QMiniConfig.from_dict(values)

from qmini.model import QMiniMoE


@pytest.mark.parametrize("use_state_space", [False, True])
def test_estimated_parameter_count_matches_live_model(use_state_space):
    config = QMiniConfig(
        vocab_size=128, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
        use_state_space=use_state_space,
    )
    model = QMiniMoE(config)
    assert config.estimated_parameter_count() == model.parameter_report()["total"]


def test_light_profile_parameter_count_matches_instantiated_model():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = QMiniConfig.from_json(root / "configs" / "qmini_light_0.7m.json")
    model = QMiniMoE(config)
    assert config.estimated_parameter_count() == 705_742
    assert model.parameter_report()["total"] == 705_742


def test_smoke_profile_remains_small_and_instantiable():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = QMiniConfig.from_json(root / "configs" / "qmini_smoke.json")
    model = QMiniMoE(config)
    assert config.estimated_parameter_count() == model.parameter_report()["total"]
