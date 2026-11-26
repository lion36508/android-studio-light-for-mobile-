import pytest
import torch

from qmini.config import QMiniConfig
from qmini.preflight import (
    check_training_memory_budget,
    run_architecture_preflight,
    training_memory_floor_bytes,
)


def test_training_memory_floor_is_two_times_adamw_lower_bound(tiny_config):
    assert training_memory_floor_bytes(tiny_config) == (
        tiny_config.estimated_adamw_training_state_bytes() * 2
    )


def test_memory_budget_passes_with_sufficient_measured_memory(monkeypatch, tiny_config):
    import qmini.preflight as preflight

    required = training_memory_floor_bytes(tiny_config)
    monkeypatch.setattr(
        preflight,
        "available_memory_info",
        lambda _device: {
            "available_bytes": required + 1,
            "total_bytes": required + 10,
            "source": "test-fixture",
        },
    )
    report = check_training_memory_budget(tiny_config, torch.device("cpu"))
    assert report["checked"] is True
    assert report["passed"] is True


def test_memory_budget_rejects_insufficient_memory_before_model_allocation(
    monkeypatch, tiny_config
):
    import qmini.preflight as preflight

    monkeypatch.setattr(
        preflight,
        "available_memory_info",
        lambda _device: {
            "available_bytes": 0,
            "total_bytes": 100,
            "source": "test-fixture",
        },
    )
    with pytest.raises(MemoryError, match="Training refused before model allocation"):
        check_training_memory_budget(tiny_config, torch.device("cpu"))


def test_architecture_preflight_checks_shapes_and_gradients_without_updating_weights(
    tiny_config,
):
    report = run_architecture_preflight(
        tiny_config,
        device="cpu",
        sequence_length=8,
        max_parameters=1_000_000,
        check_backward=True,
    )
    assert report["format"] == "qmini-architecture-preflight-v1"
    assert report["parameters_expected"] == tiny_config.estimated_parameter_count()
    assert report["parameters_materialized"] == tiny_config.estimated_parameter_count()
    assert report["input_shape"] == [1, 8]
    assert report["logits_shape"] == [1, 8, tiny_config.vocab_size]
    assert report["forward_finite"] is True
    assert report["backward_checked"] is True
    assert report["gradient_tensors_checked"] > 0
    assert report["gradient_tensors_nonzero"] > 0
    assert report["random_initialization_only"] is True
    assert report["pretrained_weights_loaded"] is False
    assert report["dataset_opened"] is False
    assert report["optimizer_step_performed"] is False
    assert report["weights_updated"] is False


def test_architecture_preflight_rejects_oversized_config_before_allocation():
    config = QMiniConfig(
        vocab_size=32768,
        d_model=1024,
        d_hidden=2048,
        n_layers=32,
        n_experts=8,
        top_k=1,
        max_context=512,
        kernel_size=10,
        dilations=(1,) * 32,
        use_state_space=True,
        gradient_checkpointing=True,
    )
    with pytest.raises(MemoryError, match="refused to materialize"):
        run_architecture_preflight(
            config,
            device="cpu",
            sequence_length=8,
            max_parameters=250_000_000,
        )
