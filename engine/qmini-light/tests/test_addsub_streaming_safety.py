import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.model import AddSubLanguageModel


def _model():
    return AddSubLanguageModel(AddSubConfig(
        vocab_size=264, d_model=32, d_hidden=48, n_layers=1,
        n_experts=2, kernel_size=3, max_context=32, dropout=0.0,
    )).eval()


def test_forward_step_requires_inference_mode():
    model = _model()
    caches = model.init_cache(batch_size=1)
    with pytest.raises(RuntimeError, match="inference_mode"):
        model.forward_step(torch.tensor([5]), caches)


def test_forward_step_rejects_malformed_cache_shape():
    model = _model()
    caches = model.init_cache(batch_size=1)
    invalid = list(caches)
    invalid[0] = (torch.zeros((1, 1, 1)), caches[0][1])
    with torch.inference_mode(), pytest.raises(ValueError, match="Convolution cache shape"):
        model.forward_step(torch.tensor([5]), invalid)


def test_forward_step_rejects_wrong_device_and_cache_container():
    model = _model()
    caches = model.init_cache(batch_size=1)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="cache tuple"):
            model.forward_step(torch.tensor([5]), None)
        with pytest.raises(ValueError, match="model device"):
            model.forward_step(torch.tensor([5], device="meta"), caches)
