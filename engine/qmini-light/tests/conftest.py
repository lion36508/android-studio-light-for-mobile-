import pytest
from qmini.config import QMiniConfig
@pytest.fixture
def tiny_config():
    return QMiniConfig(vocab_size=128, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, top_k=1, max_context=64, kernel_size=5, dilations=(1, 2),
        max_thinking_steps=2, thinking_steps=1, seed=123)
