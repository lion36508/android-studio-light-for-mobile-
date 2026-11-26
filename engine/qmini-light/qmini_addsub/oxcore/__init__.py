"""Standalone native primitives for the experimental Q-Mini Add/Sub model.

Importing this package does not import PyTorch or load a native binary. Call
OxCoreNative.load() explicitly when a compiled OxCore shared library is ready.
"""

from .native import OxCoreError, OxCoreNative, StaticArena

__all__ = ["OxCoreError", "OxCoreNative", "StaticArena"]
