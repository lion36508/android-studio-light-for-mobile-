from __future__ import annotations

import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.export import _quantized_parameter_names
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.quantization_audit import audit_model_quantization


def _tiny_model() -> AddSubLanguageModel:
    config = AddSubConfig(
        vocab_size=264,
        d_model=8,
        d_hidden=16,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=8,
    )
    return AddSubLanguageModel(config)


def test_quantization_audit_covers_every_exported_weight_and_all_elements():
    model = _tiny_model()
    report = audit_model_quantization(model)

    assert report["format"] == "qmini-addsub-quantization-audit-v1"
    assert report["bits_per_weight"] == 3
    assert report["quantized_parameter_count"] > 1
    assert report["quantized_element_count"] == sum(
        item["element_count"] for item in report["tensors"]
    )
    assert sum(item["count"] for item in report["overall_symbols"]) == report["quantized_element_count"]
    assert 0.0 <= report["overall_zero_symbol_fraction"] <= 1.0
    assert all(
        sum(symbol["count"] for symbol in item["symbols"]) == item["element_count"]
        for item in report["tensors"]
    )


def test_quantization_audit_reports_real_all_zero_tensor_as_zero_symbol_only():
    model = _tiny_model()
    with torch.no_grad():
        model.token_embedding.weight.zero_()

    report = audit_model_quantization(model)
    embedding = next(item for item in report["tensors"] if item["name"] == "token_embedding.weight")

    assert embedding["scale"] > 0.0
    assert embedding["zero_symbol_fraction"] == 1.0
    assert embedding["used_symbol_count"] == 1
    assert [item["count"] for item in embedding["symbols"]] == [
        0, 0, model.token_embedding.weight.numel(), 0, 0
    ]


def test_quantization_audit_rejects_non_finite_quantized_parameters():
    model = _tiny_model()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="empty or non-finite"):
        audit_model_quantization(model)



def test_quantization_audit_labels_legacy_fixed_scale_models_correctly():
    model = _tiny_model()
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in _quantized_parameter_names(model):
            parameters[name]._qmini_fixed_scale = 1.0

    report = audit_model_quantization(model)
    assert report["quantization_scale_mode"] == "legacy-fixed"
    assert report["weight_format"] == "legacy-fixed-scale-five-symbol-q3"
