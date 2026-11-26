"""Q-Mini: quaternary causal-convolution sparse-MoE language model."""
__version__ = "0.1.0"
from .config import QMiniConfig
from .model import QMiniMoE
__all__ = ["QMiniConfig", "QMiniMoE", "__version__"]
